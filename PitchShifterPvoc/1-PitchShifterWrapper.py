import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
import numpy as np
import sys, os
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from scipy import signal 
from PQMF.pqmf import CachedPQMF
from typing import List, Tuple
import random
from PQMF.PitchShifterPvoc.VocoderPitchShifter import PhaseVocoderPitchShift


# Diretórios
# ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
audio_dir = os.path.join(project_root, "audio")
out_audio = os.path.join(script_dir, "audio")
out_torchscript = os.path.join(script_dir, "torchscript")
os.makedirs(audio_dir, exist_ok=True)
os.makedirs(out_audio, exist_ok=True)
os.makedirs(out_torchscript, exist_ok=True)



class PitchShifter(nn.Module):
    def __init__(self, n_steps: int, n_fft: int = 4096, hop_length: int = 128, win_length: int = 1024, window_type: str = "hann"):
        super().__init__()
        self.n_steps = int(n_steps)
        # instância do vocoder (torchscript-friendly)
        self.vocoder = PhaseVocoderPitchShift(n_fft=n_fft, hop_length=hop_length, win_length=win_length)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # chama vocoder com o n_steps armazenado
        return self.vocoder(x, self.n_steps)


# --- Pitch shift TorchScript-friendly (não estou usando!!!) ---
class ScriptablePitchShift(nn.Module):
    """
    Pitch shift simples por mudança de taxa (resample via F.interpolate).
    Não preserva perfeitamente a fase como phase-vocoder
    Entrada: tensor [B, T] ou [T] ; retorna tensor [B, T] (mesmo comprimento de entrada).
    n_steps: lista com número de semitons para cada sub-banda
    factor = 2**(n_steps/12)
    """
    def __init__(self, n_steps: int):
        super().__init__()
        # armazenar n_steps como int e fator como float
        self.n_steps = int(n_steps)
        # fator de pitch (playback speed)
        self.factor = float(2 ** (self.n_steps / 12.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x pode ser [B, T] ou [T]; normalizar para [B, T]
        squeeze_back = False
        if x.dim() == 1:
            x = x.unsqueeze(0)
            squeeze_back = True
        if x.dim() == 3 and x.shape[1] == 1:
            # caso venha [B,1,T] -> tornar [B,T]
            x = x.squeeze(1)
        if x.dim() != 2:
            raise RuntimeError("ScriptablePitchShift espera [B, T] ou [T]")

        B, T = x.shape
        # transformar para [B, 1, T] para usar interpolate (1D)
        x3 = x.unsqueeze(1)  # [B,1,T]

        # novo comprimento após mudança de velocidade
        new_len_f = float(T) / float(self.factor)
        new_len = int(round(new_len_f))
        if new_len < 1:
            new_len = 1

        # usar interpolate para reamostrar (modo linear)
        x_resampled = F.interpolate(x3, size=new_len, mode="linear", align_corners=False)

        # ajustar de volta para tamanho T (central crop ou pad)
        cur = x_resampled.shape[-1]
        if cur > T:
            start = (cur - T) // 2
            x_out = x_resampled[..., start:start + T]
        elif cur < T:
            pad = T - cur
            left = pad // 2
            right = pad - left
            x_out = F.pad(x_resampled, (left, right))
        else:
            x_out = x_resampled

        x_out = x_out.squeeze(1)  # [B, T]
        if squeeze_back:
            return x_out.squeeze(0)
        return x_out


# --- Wrapper PQMF -> sub-bands -> phase vocoder pitch Shifter -> signal reconstruction ---
class PQMFPitchShiftWrapper(nn.Module):
    def __init__(self, attenuation=100, n_band=16, m_buffer_size=8192, sample_rate: int = 44100, shifts_in_semitones=None):
        super().__init__()
        self.n_band = n_band
        self.attenuation = attenuation
        self.sample_rate = sample_rate
        self.pqmf = CachedPQMF(attenuation, n_band)

        self._methods = ["forward", "pitchshift"]
        self._attributes = ["n_band", "attenuation",
                            "forward_in_ch", "forward_out_ch",
                            "pitchshift_in_ch", "pitchshift_out_ch",
                            "m_buffer_size", "max_buffer_size"]
        
        self.forward_in_ch = 1
        self.forward_out_ch = 1
        self.pitchshift_in_ch = 1
        self.pitchshift_out_ch = 1
        self.m_buffer_size = m_buffer_size
        self.max_buffer_size = 16384

        # Calcula a taxa de amostragem de cada sub-banda
        sub_band_sample_rate = int(round(float(self.sample_rate) / float(max(1, self.n_band))))

        # Define os valores de transposição para cada sub-banda
        if shifts_in_semitones is None:
            # Se nenhum valor for fornecido, cria uma escala cromática ascendente
            self.shifts = list(range(n_band))
        else:
            self.shifts = shifts_in_semitones

        # Cria uma lista de módulos PitchShifter com a classe phasevocoder (um para cada sub-banda)
        self.pitch_shifters = nn.ModuleList()
        # Estima o tamanho de cada sub-banda
        sub_len_est = max(16, int(self.m_buffer_size // max(1, self.n_band)))
        # Definir win_length = tamanho da sub-banda
        win_len = int(max(16, min(sub_len_est, 4096)))
        hop_len = max(1, win_len // 4)
        # escolher n_fft como próxima potência de 2 >= win_len
        def _next_pow2(x: int) -> int:
            p = 1
            while p < x:
                p <<= 1
            return p
        n_fft_val = _next_pow2(win_len)
        # limitar n_fft
        n_fft_val = int(min(n_fft_val, 4096))
        if n_fft_val < win_len:
            n_fft_val = win_len
        
        # print("PQMFPitchShiftWrapper init: n_band=" + str(self.n_band) + " m_buffer_size=" + str(self.m_buffer_size) + " sub_len_est=" + str(sub_len_est))
        # print("PQMFPitchShiftWrapper init: win_len=" + str(win_len) + " hop_len=" + str(hop_len) + " n_fft_val=" + str(n_fft_val) + " sub_band_sr=" + str(sub_band_sample_rate))
        # print("PQMFPitchShiftWrapper init: shifts_count=" + str(len(self.shifts)))

        # criar pitch shifters
        for semis in self.shifts:
             n_steps = int(round(float(semis)))
             self.pitch_shifters.append(PitchShifter(n_steps=n_steps, n_fft=n_fft_val, hop_length=hop_len, win_length=win_len))
        # print("PQMFPitchShiftWrapper init: created pitch_shifters=" + str(len(self.pitch_shifters)))

        # preparar buffers para crossfade entre blocos (overlap-add) por sub-banda
        self.band_overlap = int(min(hop_len, max(0, win_len // 4)))
        # se band_overlap < 0, setar para 0
        if self.band_overlap < 0:
            self.band_overlap = 0
        # se band_overlap > 0, criar buffers
        if self.band_overlap > 0:
            # prev_tail: [n_band, band_overlap]
            self.register_buffer("prev_tail", torch.zeros(self.n_band, self.band_overlap, dtype=torch.float32))
            # criar janela Hann full e dividir em fade_out / fade_in (cada uma com length band_overlap)
            full = torch.hann_window(2 * self.band_overlap, dtype=torch.float32)
            self.register_buffer("fade_out", full[: self.band_overlap].unsqueeze(0))  # [1, L]
            self.register_buffer("fade_in", full[self.band_overlap :].unsqueeze(0))    # [1, L]
        else:
            self.register_buffer("prev_tail", torch.zeros(self.n_band, 0, dtype=torch.float32))
            self.register_buffer("fade_out", torch.zeros(1, 0, dtype=torch.float32))
            self.register_buffer("fade_in", torch.zeros(1, 0, dtype=torch.float32))


    @torch.jit.export
    def get_methods(self) -> List[str]:
        return self._methods

    @torch.jit.export
    def get_attributes(self) -> List[str]:
        return self._attributes
    
    
    def decompose(self, x: torch.Tensor) -> torch.Tensor:
        """
        Decomposição do sinal mono em sub-bandas.
        Args:
            x: Tensor 2D [1, m_buffer_size] ou 3D [batch, 1, buffer_size]
        retorna:
            Tensor [batch, n_band, buffer_size'] ou [1, n_band, buffer_size']
        """
        if x.dim() == 2:
            x = x.unsqueeze(0)  # [1, 1, T]
        if x.dim() == 3 and x.shape[1] == 1:
            return self.pqmf.forward(x)
        else:
            raise ValueError("Entrada deve ser [1, buffer_size] ou [batch, 1, buffer_size]")

    
    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        """
        Reconstrução do sinal mono a partir das sub-bandas.
        Args:
            x: Tensor [batch, n_band, buffer_size'] ou [1, n_band, buffer_size']
        retorna:
            Tensor [batch, 1, buffer_size] ou [1, 1, buffer_size]
        """
        if x.dim() == 3 and x.shape[1] == self.n_band:
            return self.pqmf.inverse(x)
        else:
            raise ValueError(f"Entrada deve ser [batch, {self.n_band}, buffer_size'] ou [1, {self.n_band}, buffer_size']")


    def processing(self, x: torch.Tensor) -> torch.Tensor:
        """
        Aplica pitch shifting em cada sub-banda e reconstrói o sinal.
        Args:
            x: Tensor 2D [1, m_buffer_size] ou 3D [batch, 1, buffer_size]
        retorna:
            Tensor [batch, 1, buffer_size] ou [1, 1, buffer_size]
        """
        # print("pitchshifter: input.shape=" + str(list(x.shape)))
        
        # 1 - decompõe o sinal em sub-bandas
        subbands = self.decompose(x)
        # print("pitchshifter: subbands.shape=" + str(list(subbands.shape)))

        # checa se o nº de pitch shifters está correto
        if len(self.pitch_shifters) < self.n_band:
            raise RuntimeError(f"Esperado {self.n_band} pitch_shifters, encontrado {len(self.pitch_shifters)}")

        processed_bands = []

        # converte subbands para lista de tensores por banda: cada item [B, T]
        bands = list(subbands.unbind(1))
        # print("pitchshifter: bands_count=" + str(len(bands)))
        if len(bands) != len(self.pitch_shifters):
            raise RuntimeError(f"Quantidade de bandas ({len(bands)}) diferente de pitch_shifters ({len(self.pitch_shifters)})")

        # 2 - aplica o pitch shifter em cada banda
        for idx, sh in enumerate(self.pitch_shifters):
            band_bt = bands[idx]  # [B, T]
            # print("pitchshifter: band idx=" + str(idx) + " band.shape=" + str(list(band_bt.shape)))
            # aplica o pitch shifter (espera [B, T])
            shifted_bt = sh(band_bt)  # [B, T_new]
            # print("pitchshifter: shifted idx=" + str(idx) + " shifted.shape=" + str(list(shifted_bt.shape)))
            shifted_band_i = shifted_bt.unsqueeze(1)  # [B,1,T_new]


            # --- crossfade usando prev_tail (por sub-banda) ---
            if self.band_overlap > 0:
                L = self.band_overlap
                # somente para batch==1 (PD)
                if shifted_band_i.dim() == 3 and shifted_band_i.size(0) == 1 and shifted_band_i.size(-1) >= L:
                    # prefixa e sufixo atuais
                    cur_prefix = shifted_band_i[:, :, :L].squeeze(1)   # [1, L]
                    cur_suffix = shifted_band_i[:, :, -L:].squeeze(1)  # [1, L]
                    prev = self.prev_tail[idx : idx + 1, :]             # [1, L]
                    # blended = prev * fade_out + cur_prefix * fade_in
                    blended = prev * self.fade_out + cur_prefix * self.fade_in
                    # escrever blended de volta no shifted_band_i
                    shifted_band_i[:, :, :L] = blended.unsqueeze(1)
                    # atualizar prev_tail[idx] com current suffix (para próxima chamada)
                    self.prev_tail[idx : idx + 1, :] = cur_suffix.clone()
                else:
                    # atualizar prev_tail mesmo se bloco curto ou B!=1 (tentar evitar stale state)
                    if shifted_band_i.size(-1) >= L and shifted_band_i.size(0) == 1:
                        self.prev_tail[idx : idx + 1, :] = shifted_band_i[:, 0, -L:].clone()

    
            target = subbands.shape[-1]
            cur = shifted_band_i.shape[-1]
            if cur != target:
                if cur > target:
                    start = (cur - target) // 2
                    shifted_band_i = shifted_band_i[..., start:start+target]
                else:
                   pad = target - cur
                   left = pad // 2
                   right = pad - left
                   shifted_band_i = F.pad(shifted_band_i, (left, right), mode='constant', value=0.0)
            # print("pitchshifter: after pad/trunc idx=" + str(idx) + " final.shape=" + str(list(shifted_band_i.shape)))

            processed_bands.append(shifted_band_i)

        # 2 - concatena as sub-bandas e reconstrói o sinal
        shifted_subbands = torch.cat(processed_bands, dim=1)  # [B, n_band, T']
        # print("pitchshifter: shifted_subbands.shape=" + str(list(shifted_subbands.shape)))
        reconstructed = self.inverse(shifted_subbands)  # [B, 1, T]
        # print("pitchshifter: reconstructed.shape=" + str(list(reconstructed.shape)))
        if reconstructed.dim() == 3 and reconstructed.size(1) == 1:
            reconstructed = reconstructed.squeeze(1)  # [B, T]
        return reconstructed

    @torch.jit.export
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """"
        Decomposição do sinal mono em sub-bandas.
        Args:
            x: Tensor 2D [1, m_buffer_size] ou 3D [batch, 1, buffer_size]
        retorna:
            Tensor [1, buffer_size]
        """
        sub = self.decompose(x)
        y = self.inverse(sub)
        if y.dim() == 3 and y.size(1) == 1:
            y = y.squeeze(1)               # [B, T]
        return y

    @torch.jit.export
    def pitchshift(self, x: torch.Tensor) -> torch.Tensor:
        """
        Aplica pitch shifting em cada sub-banda e reconstrói o sinal.
        """
        return self.processing(x)

    

# --- Exportação TorchScript --- 
if __name__ == "__main__":
    print("Exportando PQMFPitchShiftWrapper para TorchScript...")

    shifts = [random.uniform(-24.75, 12.43) for _ in range(16)]
    print(f"Usando shifts (semitons): {shifts}")
    wrapper = PQMFPitchShiftWrapper(attenuation=100, n_band=16, m_buffer_size=8192, sample_rate=44100, shifts_in_semitones=shifts)
    wrapper.eval()

    # Exporta para TorchScript
    scripted = torch.jit.script(wrapper)
    scripted.save(os.path.join(out_torchscript, "pqmfpvoc.ts"))
    print(f"Modelo salvo como {os.path.join(out_torchscript, 'pqmfpvoc.ts')}")

    # --- Carrega o modelo TorchScript e realiza o teste ---
    print("Carregando modelo TorchScript para teste...")
    loaded = torch.jit.load(os.path.join(out_torchscript, "pqmfpvoc.ts"))
    loaded.eval()

    # Testa o modelo
    wav_path = os.path.join(audio_dir, 'flute.wav')
    wav, sr = torchaudio.load(wav_path)
    if wav.dim() == 2 and wav.shape[0] > 1:
        wav = wav[0].unsqueeze(0)  # Usa apenas o primeiro canal

    buffer_size = wrapper.m_buffer_size
    total_size = wav.shape[-1]
    pad = (buffer_size - total_size % buffer_size) % buffer_size
    if pad > 0:
        wav = torch.nn.functional.pad(wav, (0, pad))

    print(f"Áudio carregado: shape={wav.shape}, sr={sr}")

    with torch.no_grad():
        reconstructed = loaded.forward(wav)
        shifter = loaded.pitchshift(wav)
        subbands = loaded.decompose(wav)
        
    print(f"Sub-bandas shape: {subbands.shape}")
    print(f"Reconstruído shape: {reconstructed.shape}")
    print(f"Pitchshifter output shapes: {[t.shape for t in shifter]}")

    # reconstructed_2d = shifter.squeeze(0).squeeze(0).unsqueeze(0)  # [1, samples]
    # torchaudio.save(os.path.join(out_audio, "phasevocoder.wav"), reconstructed_2d.cpu(), sr)
    # print(f"Áudio reconstruído salvo em {os.path.join(out_audio, 'phasevocoder.wav')}")
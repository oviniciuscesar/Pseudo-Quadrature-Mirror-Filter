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



# Diretórios
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
audio_dir = os.path.join(project_root, "audio")
out_audio = os.path.join(script_dir, "audio")
out_torchscript = os.path.join(script_dir, "torchscript")
os.makedirs(audio_dir, exist_ok=True)
os.makedirs(out_audio, exist_ok=True)
os.makedirs(out_torchscript, exist_ok=True)




class PQMFPitchShiftWrapper(nn.Module):
    def __init__(self, attenuation=100, n_band=16, m_buffer_size=512, sample_rate: int = 44100, shifts_in_semitones=None):
        super().__init__()
        self.n_band = n_band
        self.attenuation = attenuation
        self.sample_rate = sample_rate
        self.pqmf = CachedPQMF(attenuation, n_band)
        
        self._methods = ["forward", "inverse", "pitchshifter"]
        self._attributes = ["n_band", "attenuation",
                            "forward_in_ch", "forward_out_ch",
                            "inverse_in_ch", "inverse_out_ch",
                            "pitchshifter_in_ch", "pitchshifter_out_ch",
                            "m_buffer_size", "max_buffer_size"]
        
        self.forward_in_ch = 1
        self.forward_out_ch = 1
        self.inverse_in_ch = 1
        self.inverse_out_ch = 1
        self.pitchshifter_in_ch = 1
        self.pitchshifter_out_ch = 2
        self.m_buffer_size = m_buffer_size
        self.max_buffer_size = 8192

        # Calcula a taxa de amostragem de cada sub-banda
        # sub_band_sample_rate = int(self.sample_rate // max(1, self.n_band))
        sub_band_sample_rate = int(round(float(self.sample_rate) / float(max(1, self.n_band))))

        # Define os valores de transposição para cada bandaå
        if shifts_in_semitones is None:
            # Se nenhum valor for fornecido, cria uma escala cromática ascendente
            self.shifts = list(range(n_band))
        else:
            self.shifts = shifts_in_semitones
        
        # Cria uma lista de módulos PitchShift, um para cada banda.
        # Usar nn.ModuleList é a maneira correta de registrar uma lista de módulos em PyTorch.
        self.pitch_shifters = nn.ModuleList()
        for semis in self.shifts:
            n_steps = int(round(float(semis)))
            self.pitch_shifters.append(T.PitchShift(sample_rate=sub_band_sample_rate, n_steps=n_steps))
            # self.pitch_shifters.append(nn.Identity())



    @torch.jit.export
    def get_methods(self) -> List[str]:
        return self._methods

    @torch.jit.export
    def get_attributes(self) -> List[str]:
        return self._attributes
    
    @torch.jit.export
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Decomposição do sinal mono em sub-bandas.
        Args:
            x: Tensor 2D [1, m_buffer_size] ou 3D [batch, 1, buffer_size]
        Returns:
            Tensor [batch, n_band, buffer_size'] ou [1, n_band, buffer_size']
        """
        if x.dim() == 2:
            x = x.unsqueeze(0)  # [1, 1, T]
        if x.dim() == 3 and x.shape[1] == 1:
            return self.pqmf.forward(x)
        else:
            raise ValueError("Entrada deve ser [1, buffer_size] ou [batch, 1, buffer_size]")

    @torch.jit.export
    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        """
        Reconstrução do sinal mono a partir das sub-bandas.
        Args:
            x: Tensor [batch, n_band, buffer_size'] ou [1, n_band, buffer_size']
        Returns:
            Tensor [batch, 1, buffer_size] ou [1, 1, buffer_size]
        """
        if x.dim() == 3 and x.shape[1] == self.n_band:
            return self.pqmf.inverse(x)
        else:
            raise ValueError(f"Entrada deve ser [batch, {self.n_band}, buffer_size'] ou [1, {self.n_band}, buffer_size']")
    
    @torch.jit.export
    def pitchshifter(self, x: torch.Tensor) -> torch.Tensor:
        
        # 1 - decompõe o sinal em sub-bandas
        subbands = self.forward(x)

        if len(self.pitch_shifters) < self.n_band:
            raise RuntimeError(f"Esperado {self.n_band} pitch_shifters, encontrado {len(self.pitch_shifters)}")

        processed_bands = []

        # 2 - itera sobre cada banda e aplica o pitch shift
        for i in range(self.n_band):
            band_i = subbands[:, i:i+1, :]  # extrai a i-ésima banda  [B,1,T]
            band_bt = band_i.squeeze(1)  # [B,T]
            # aplica o pitch shifter correspondente a cada sub-banda
            shifted_bt = self.pitch_shifters[i](band_bt)  # [B,T_new]  (PitchShift aceita [B,T])
            shifted_band_i = shifted_bt.unsqueeze(1) 
            target = band_i.shape[-1]
            cur = shifted_band_i.shape[-1]
            if cur != target:
                if cur > target:
                    start = (cur - target) // 2
                    shifted_band_i = shifted_band_i[..., start:start+target]
                else:
                    pad = target - cur
                    left = pad // 2
                    right = pad - left
                    shifted_band_i = F.pad(shifted_band_i, (left, right), mode='reflect')

            processed_bands.append(shifted_band_i)

        
        # 2 - concatena e reconstrói
        shifted_subbands = torch.cat(processed_bands, dim=1)  # [B, n_band, T']
        reconstructed = self.inverse(shifted_subbands)  # [B, 1, T]
        return reconstructed

    
# --- Exportação TorchScript --- 
if __name__ == "__main__":
    print("Exportando PQMFPitchShiftWrapper para TorchScript...")

    shifts = [random.uniform(-48.53, 12.32) for _ in range(16)]
    print(f"Usando shifts (semitons): {shifts}")
    wrapper = PQMFPitchShiftWrapper(attenuation=100, n_band=16, m_buffer_size=8192, sample_rate=44100, shifts_in_semitones=shifts)
    wrapper.eval()


    # # Exporta para TorchScript
    # scripted = torch.jit.script(wrapper)
    # scripted.save(os.path.join(torchscript_dir, "pqmfps.ts"))
    # print(f"Modelo salvo como {os.path.join(torchscript_dir, 'pqmfps.ts')}")

    # # --- Carrega o modelo TorchScript e realiza o teste ---
    # print("Carregando modelo TorchScript para teste...")
    # loaded = torch.jit.load(os.path.join(torchscript_dir, "pqmfps.ts"))
    # loaded.eval()

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
        subbands = wrapper.forward(wav)
        reconstructed = wrapper.inverse(subbands)
        shifter = wrapper.pitchshifter(wav)
        print(f"Sub-bandas shape: {subbands.shape}")
        print(f"Reconstruído shape: {reconstructed.shape}")
        print(f"Process output shapes: {[t.shape for t in shifter]}")

    reconstructed_2d = shifter.squeeze(0).squeeze(0).unsqueeze(0)  # [1, samples]
    torchaudio.save(os.path.join(out_audio, "reconstruido.wav"), reconstructed_2d.cpu(), sr)
    print(f"Áudio reconstruído salvo em {os.path.join(out_audio, 'reconstruido.wav')}")
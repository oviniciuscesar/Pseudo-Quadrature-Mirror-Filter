"""
TorchScript-friendly Phase Vocoder pitch shifter (pure PyTorch).

This module implements a high-quality phase-vocoder-based pitch shifter fully
in PyTorch (no torchaudio dependency). It was written to be compatible with
`torch.jit.script` and to run inside environments like Pure Data's `torch.ts~`.

Usage:
    sh = PhaseVocoderPitchShift(n_fft=1024, hop_length=256, win_length=1024)
    y = sh(x, n_steps)

Inputs/outputs:
    x: Tensor [B, T] or [T]
    n_steps: int (semitones)
    returns: Tensor [B, T] (same length as input)

Notes:
- The algorithm: time-stretch via phase vocoder (rate = 1/factor), then resample
  back to original length. factor = 2**(n_steps/12).
- All computations use PyTorch ops and are (intentionally) written to be
  TorchScript-friendly.
- This implementation trades some micro-optimizations for readability and
  TorchScript compatibility.

References:
- Griffin-Lim / classic phase vocoder algorithm (implemented like librosa)
"""

from typing import Tuple
import math
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as T
import torchaudio.functional as taf
import numpy as np  

def _principal_angle(x: torch.Tensor) -> torch.Tensor:
    """Wrap angle to [-pi, pi]."""
    two_pi = 2.0 * math.pi
    # x may be any shape
    x = x + math.pi
    # torch.remainder works with tensors and scalars
    x = torch.remainder(x, two_pi)
    x = x - math.pi
    return x

#---- PhaseVocoderPitchShift class ----#
class PhaseVocoderPitchShift(nn.Module):
    def __init__(self, n_fft: int = 1024, hop_length: int = 256, win_length: int = 1024):
        super().__init__()
        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length)
        self.win_length = int(win_length)
        # window must be registered as buffer for TorchScript
        window = torch.hann_window(self.win_length)
        self.register_buffer("window", window)

    # STFT
    def _stft(self, x: torch.Tensor) -> torch.Tensor:
        """
        Short-time Fourier transform.
        - input: sub-band waveform
        - Return real/imag stft: shape [batch, freq, frames, 2]
        """

        # ensure input length >= win_length to avoid empty frames / padding errors
        if x.dim() == 1:
            x = x.unsqueeze(0)
        B, T = x.shape
        print("_stft: B=" + str(B) + " T=" + str(T) + " n_fft=" + str(self.n_fft) + " win_length=" + str(self.win_length) + " hop_length=" + str(self.hop_length))
        
        # se o tamanho da sub-banda for menor que n_fft, fazer padding à direita
        if T < max(1, self.n_fft):
            pad = int(max(1, self.n_fft) - T)
            print("_stft: padding input pad=" + str(pad))
            x = F.pad(x, (0, pad), mode="constant", value=0.0)

        window = self.window
        if window.device != x.device or window.dtype != x.dtype:
            window = torch.hann_window(self.win_length, device=x.device, dtype=x.dtype)

        # compute STFT
        return torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            return_complex=False,
            normalized=True,
            center=True,
            pad_mode="constant",
        )

    # Inverse STFT
    def _istft(self, spec: torch.Tensor) -> torch.Tensor:
        """
        Inverse STFT
        - input: real/imag stft [batch, freq, frames, 2]
        - output: time-domain waveform [batch, T]
        """
        # debugando
        print("_istft: entering: spec.shape=" + str(list(spec.shape)) + " numel=" + str(spec.numel()))

        # empty frames -> fallback zeros
        if spec.numel() == 0 or (spec.dim() >= 3 and spec.size(2) == 0):
            B = spec.size(0) if spec.dim() >= 1 else 1
            length = int(self.win_length)
            print("_istft: empty spec -> returning zeros B=" + str(B) + " len=" + str(length))
            return torch.zeros(B, length, device=spec.device, dtype=torch.float32)

        # converter stacked real/imag [batch, freq, frames, 2] para complexo [batch, freq, frames]
        spec_c = spec
        if spec.dim() >= 4 and spec.size(-1) == 2:
            spec_c = torch.view_as_complex(spec)
            print("_istft: converted to complex, spec_c.shape=" + str(list(spec_c.shape)) + " dtype=" + str(spec_c.dtype))

        # garantir dtype complexo
        if torch.is_complex(spec_c):
            spec_c = spec_c.to(torch.complex64)
        else:
            print("_istft: warning - spec_c not complex; shape=" + str(list(spec_c.shape)))

        # Se somente 1 frame freq x 1: usar irfft direto como fallback
        if spec_c.dim() >= 3 and spec_c.size(2) == 1:
            # spec_c[..., :, 0] -> [batch, freq]
            spec_frame = spec_c[..., 0]  # complex tensor shape [batch, freq]
            # irfft para reconstruir um quadro de comprimento n_fft
            y_frame = torch.fft.irfft(spec_frame, n=self.n_fft)  # [batch, n_fft], real
            # cortar/centralizar para win_length (retornar comprimento plausível)
            if y_frame.dim() == 2:
                out = y_frame[..., : int(self.win_length)]
            else:
                out = y_frame
            print("_istft: irfft fallback used, out.shape=" + str(list(out.shape)))
            return out.to(torch.float32)

        # caso geral: usar torch.istft (janela válida)
        if self.window.numel() == 0:
            self.window = torch.hann_window(int(self.win_length), device=spec.device, dtype=torch.float32)
            print("_istft: created fallback hann_window length=" + str(self.win_length))
        else:
            window = self.window

        print("_istft: calling istft n_fft=" + str(self.n_fft) + " hop=" + str(self.hop_length) + " win_len=" + str(self.win_length) + " window_numel=" + str(self.window.numel()))
        out = torch.istft(
            spec_c,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            normalized=True,
        )
        print("_istft: istft succeeded out.shape=" + str(list(out.shape)))
        return out

    def _magphase(self, spec: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert real/imag to magnitude and phase.
        spec: [B, freq, frames, 2]
        returns (magnitude [B,freq,frames], phase [B,freq,frames])
        """
        real = spec[..., 0]
        imag = spec[..., 1]
        mag = torch.sqrt(real * real + imag * imag + 1e-12)
        phase = torch.atan2(imag, real)
        return mag, phase

    def _recompose(self, mag: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
        """Compose real/imag tensor from magnitude and phase. returns [B,freq,frames,2]"""
        real = mag * torch.cos(phase)
        imag = mag * torch.sin(phase)
        return torch.stack([real, imag], dim=-1)

    def _phase_vocoder(self, mag: torch.Tensor, phase: torch.Tensor, rate: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perform phase vocoder time-stretching on magnitude/phase.
        mag, phase: [B, freq, frames]
        rate: >0.0  (e.g. rate = 1.5 speeds up 50%; frames_out = int(frames / rate))
        returns (mag_stretch, phase_stretch) with frames_out
        """
        B, freq, frames = mag.shape
        # time positions in original frames to sample from
        # frames_new = int(math.floor(frames / rate))  # target number of frames
        # We'll build positions t' = torch.arange(0, frames, step=rate)
        # But using float step in torch.arange might accumulate fp error; compute count
        frames_f = float(frames)
        frames_out = int(math.floor((frames_f) / rate))
        if frames_out < 1:
            frames_out = 1

        device = mag.device

        # omega: expected phase advance for each bin
        # omega shape: (freq, )
        k = torch.arange(0, freq, device=device, dtype=mag.dtype)
        omega = 2.0 * math.pi * k * float(self.hop_length) / float(self.n_fft)  # [freq]

        # prepare output tensors
        mag_stretch = torch.zeros((B, freq, frames_out), dtype=mag.dtype, device=device)
        phase_stretch = torch.zeros((B, freq, frames_out), dtype=phase.dtype, device=device)

        # precompute frame positions (float) and integer parts
        # t_prime = [i * rate for i in range(frames_out)] but that's sampling original frames at new positions
        # we want mapping new -> old: t = i * rate
        t_prime = (torch.arange(0, frames_out, device=device, dtype=mag.dtype) * rate)  # floats

        # floor indices
        t0 = torch.floor(t_prime).to(torch.long)  # [frames_out]
        t1 = t0 + 1
        t1 = torch.clamp(t1, max=frames - 1)
        alpha = (t_prime - t0.to(mag.dtype)).unsqueeze(0).unsqueeze(0)  # [1,1,frames_out]

        # For each output frame j, we interpolate magnitude and compute phase progression
        # We'll loop over frames_out (safe in TorchScript)
        for j in range(frames_out):
            i0 = int(t0[j].item())
            i1 = int(t1[j].item())
            a = float(alpha[0,0,j].item())
            # magnitude linear interpolation
            mag0 = mag[..., i0]
            mag1 = mag[..., i1]
            mag_interp = (1.0 - a) * mag0 + a * mag1  # [B, freq]

            # phase progression
            phi0 = phase[..., i0]
            phi1 = phase[..., i1]
            # delta phase
            dp = phi1 - phi0 - omega.unsqueeze(0)  # broadcast omega [freq] -> [1,freq]
            dp = _principal_angle(dp)
            # predicted phase for new frame
            phi = phi0 + omega.unsqueeze(0) + a * dp

            mag_stretch[..., j] = mag_interp
            phase_stretch[..., j] = phi

        return mag_stretch, phase_stretch

    def forward(self, x: torch.Tensor, n_steps: int) -> torch.Tensor:
        """
        x: [B, T] or [T]
        n_steps: integer number of semitones (can be negative)
        returns: [B, T]
        """
        squeeze_back = False
        if x.dim() == 1:
            x = x.unsqueeze(0)
            squeeze_back = True
        if x.dim() == 3 and x.shape[1] == 1:
            x = x.squeeze(1)

        if x.dim() != 2:
            raise RuntimeError("Input must be [B,T] or [T]")

        B, T = x.shape

        # compute factor and rate
        factor = float(2 ** (float(int(n_steps)) / 12.0))  # playback speed factor
        # to change pitch by n_steps while keeping duration, we time-stretch by 1/factor, then resample
        rate = 1.0 / factor

        # STFT
        spec = self._stft(x)  # [B, freq, frames, 2]
        if spec.numel() == 0 or spec.shape[2] == 0:
            if squeeze_back:
                return x.squeeze(0)
            return x
        mag, phase = self._magphase(spec)  # [B,freq,frames]

        # phase vocoder time-stretch
        mag_stretch, phase_stretch = self._phase_vocoder(mag, phase, rate)

        # recompose and istft
        spec_stretch = self._recompose(mag_stretch, phase_stretch)  # [B,freq,frames_out,2]

        # estimate output length after stretch (approx)
        frames_out = spec_stretch.shape[2]
        # length estimation: (frames_out-1)*hop + n_fft
        length_stretch = int((frames_out - 1) * self.hop_length + self.n_fft)
        if length_stretch < 1:
            length_stretch = 1

        # inverse STFT
        y = self._istft(spec_stretch)
        
        L = y.shape[-1]
        if L != length_stretch:
            if L < length_stretch:
                pad = length_stretch - L
                left = pad // 2
                right = pad - left
                y = F.pad(y.unsqueeze(1), (left, right), mode='constant', value=0.0).squeeze(1)
            else:
                # center-truncate
                start = (L - length_stretch) // 2
                y = y[..., start:start + length_stretch]

       # now resample y to original length T using linear interpolation
        y3 = y.unsqueeze(1)  # [B,1,L]
        y_resampled = F.interpolate(y3, size=T, mode="linear", align_corners=False)
        y_out = y_resampled.squeeze(1)

        if squeeze_back:
            return y_out.squeeze(0)
        return y_out


try:
    import soundfile as sf  # pysoundfile
    HAS_SF = True
except Exception:
    HAS_SF = False
    try:
        import torchaudio
        HAS_TORCHAUDIO = True
    except Exception:
        HAS_TORCHAUDIO = False

# loads audio using soundfile or torchaudio
def load_audio(path):
    if HAS_SF:
        data, sr = sf.read(path, dtype='float32')
        # soundfile returns (T,) for mono or (T, C)
        if data.ndim == 2:
            data = data.mean(axis=1)  # converte para mono
        return data.astype(np.float32), sr
    elif HAS_TORCHAUDIO:
        wav, sr = torchaudio.load(path)  # wav: [C, T]
        if wav.size(0) > 1:
            wav = wav.mean(dim=0, keepdim=True)
        return wav.squeeze(0).cpu().numpy().astype(np.float32), sr
    else:
        raise RuntimeError("Nenhum backend de áudio disponível (instale soundfile ou torchaudio)")

# saves audio using soundfile or torchaudio
def save_audio(path, data, sr):
    if HAS_SF:
        sf.write(path, data.astype(np.float32), sr, subtype='PCM_16')
    elif HAS_TORCHAUDIO:
        tensor = torch.from_numpy(data).unsqueeze(0)  # [1, T]
        torchaudio.save(path, tensor, sr)
    else:
        raise RuntimeError("Nenhum backend de áudio disponível (instale soundfile ou torchaudio)")





def main():
    p = argparse.ArgumentParser(description="Teste PhaseVocoderPitchShift")
    p.add_argument("input", help="arquivo de entrada (wav)")
    p.add_argument("output", help="arquivo de saída (wav)")
    p.add_argument("--n_steps", type=float, default=4.0, help="semitons")
    p.add_argument("--n_fft", type=int, default=1024)
    p.add_argument("--hop_length", type=int, default=256)
    p.add_argument("--win_length", type=int, default=1024)
    args = p.parse_args()

    x_np, sr = load_audio(args.input)
    print(f"Loaded {args.input}: {x_np.shape}, sr={sr}")

    # prepara tensor [batch, T]
    x_t = torch.from_numpy(x_np).unsqueeze(0)  # [1, T]
    sh = PhaseVocoderPitchShift(n_fft=args.n_fft, hop_length=args.hop_length, win_length=args.win_length)
    sh.eval()

    n_steps_int = int(round(float(args.n_steps)))

    with torch.no_grad():
        y_t = sh(x_t, n_steps=n_steps_int)  # [B, T]
    y = y_t.squeeze(0).cpu().numpy()

    # normaliza se necessário (opcional): evita clipping quando salvar PCM16
    maxv = np.max(np.abs(y))
    if maxv > 1.0:
        y = y / maxv

    save_audio(args.output, y, sr)
    print(f"Saved {args.output}: {y.shape}, sr={sr}")

if __name__ == "__main__":
    main()
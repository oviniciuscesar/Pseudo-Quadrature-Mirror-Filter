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


class PhaseVocoderPitchShift(nn.Module):
    def __init__(self, n_fft: int = 1024, hop_length: int = 256, win_length: int = 1024):
        super().__init__()
        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length)
        self.win_length = int(win_length)
        # window must be registered as buffer for TorchScript
        win = torch.hann_window(self.win_length)
        self.register_buffer("window", win)

        # state for phase vocoder
        freq = self.n_fft // 2 + 1
        self.register_buffer("last_phase", torch.zeros(1, freq, 1))


    def _stft(self, x: torch.Tensor) -> torch.Tensor:
        """Return real/imag stft: shape [B, freq, frames, 2]"""

        # defensive: ensure input length >= win_length to avoid empty frames / padding errors
        if x.dim() == 1:
            x = x.unsqueeze(0)
        B, T = x.shape
        print("_stft: B=" + str(B) + " T=" + str(T) + " n_fft=" + str(self.n_fft) + " win_length=" + str(self.win_length) + " hop_length=" + str(self.hop_length))
        
        if T < max(1, self.n_fft):
            pad = int(max(1, self.n_fft) - T)
            print("_stft: padding input pad=" + str(pad))
            x = F.pad(x, (0, pad), mode="constant", value=0.0)

        return torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            return_complex=False,
            normalized=False,
            center=True,
            pad_mode="constant",
        )

    def _istft(self, spec: torch.Tensor) -> torch.Tensor:
        """Inverse stft. spec shape: [B, freq, frames, 2]"""
        # debug
        print("_istft: entering: spec.shape=" + str(list(spec.shape)) + " numel=" + str(spec.numel()))

        # empty frames -> fallback zeros
        if spec.numel() == 0 or (spec.dim() >= 3 and spec.size(2) == 0):
            B = spec.size(0) if spec.dim() >= 1 else 1
            length = int(self.win_length)
            print("_istft: empty spec -> returning zeros B=" + str(B) + " len=" + str(length))
            return torch.zeros(B, length, device=spec.device, dtype=torch.float32)

        # converter stacked real/imag [B,F,T,2] para complexo [B,F,T]
        spec_c = spec
        if spec.dim() >= 4 and spec.size(-1) == 2:
            spec_c = torch.view_as_complex(spec)
            print("_istft: converted to complex, spec_c.shape=" + str(list(spec_c.shape)) + " dtype=" + str(spec_c.dtype))

        # garantir dtype complexo
        if torch.is_complex(spec_c):
            spec_c = spec_c.to(torch.complex64)
        else:
            print("_istft: warning - spec_c not complex; shape=" + str(list(spec_c.shape)))

        # Se somente 1 frame freq x 1, usar irfft direto como fallback (evita caminho interno problemático do istft)
        if spec_c.dim() >= 3 and spec_c.size(2) == 1:
            # spec_c[..., :, 0] -> [B, F]
            spec_frame = spec_c[..., 0]  # complex tensor shape [B, F]
            # irfft para reconstruir um quadro de comprimento n_fft
            y_frame = torch.fft.irfft(spec_frame, n=self.n_fft)  # [B, n_fft], real
            # cortar/centralizar para win_length (retornar comprimento plausível)
            if y_frame.dim() == 2:
                out = y_frame[..., : int(self.win_length)]
            else:
                out = y_frame
            print("_istft: irfft fallback used, out.shape=" + str(list(out.shape)))
            return out.to(torch.float32)

        # caso geral: usar torch.istft (janela válida)
        if self.window.numel() == 0:
            window = torch.hann_window(int(self.win_length), device=spec.device, dtype=torch.float32)
            print("_istft: created fallback hann_window length=" + str(self.win_length))
        else:
            window = self.window

        print("_istft: calling istft n_fft=" + str(self.n_fft) + " hop=" + str(self.hop_length) + " win_len=" + str(self.win_length) + " window_numel=" + str(window.numel()))
        out = torch.istft(
            spec_c,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window,
            normalized=False,
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
        Stateful phase vocoder.
        """
        B, freq, frames = mag.shape
        if frames == 0:
            return mag, phase
            
        frames_out = int(round(frames / rate))
        if frames_out == 0:
            return torch.zeros_like(mag[:,:,:0]), torch.zeros_like(phase[:,:,:0])

        if frames <= 1:
            mag_out = mag.repeat(1,1,frames_out)
            phase_out = torch.zeros_like(mag_out)
            
            if self.last_phase.numel() > 1 and self.last_phase.shape == (B, freq, 1):
                phase_accum = self.last_phase.squeeze(-1) # [B, freq]
            else: # first block
                phase_accum = phase[..., 0]

            k = torch.arange(0, freq, device=mag.device, dtype=mag.dtype)
            omega = 2.0 * math.pi * k * float(self.hop_length) / float(self.n_fft) # [F]

            for j in range(frames_out):
                phase_out[..., j] = phase_accum
                phase_accum += omega
            
            self.last_phase = phase_accum.unsqueeze(-1).clone()
            return mag_out, phase_out

        # expected phase advance
        k = torch.arange(0, freq, device=mag.device, dtype=mag.dtype)
        omega = 2.0 * math.pi * k * float(self.hop_length) / float(self.n_fft) # [F]
        
        # phase derivative (instantaneous frequency)
        phase_d = phase[..., 1:] - phase[..., :-1]
        phase_d = _principal_angle(phase_d - omega.unsqueeze(0).unsqueeze(-1)) # [B, F, T-1]

        # true frequency (phase advance per hop)
        true_freq = omega.unsqueeze(0).unsqueeze(-1) + phase_d # [B, F, T-1]
        
        # output tensors
        mag_stretch = torch.zeros((B, freq, frames_out), dtype=mag.dtype, device=mag.device)
        phase_stretch = torch.zeros((B, freq, frames_out), dtype=phase.dtype, device=mag.device)

        # use last_phase if available
        if self.last_phase.numel() > 1 and self.last_phase.shape == (B, freq, 1):
            phase_accum = self.last_phase.squeeze(-1) # [B, freq]
        else: # first block
            phase_accum = phase[..., 0]

        # mapping from output time to input time
        t_out = torch.arange(frames_out, device=mag.device, dtype=mag.dtype) * rate
        
        for j in range(frames_out):
            t = t_out[j]
            t_int = int(torch.floor(t))
            alpha = t - t_int
            
            # magnitude interpolation
            if t_int < frames - 1:
                mag_stretch[..., j] = (1-alpha) * mag[..., t_int] + alpha * mag[..., t_int+1]
            else:
                mag_stretch[..., j] = mag[..., -1]

            # phase accumulation
            phase_stretch[..., j] = phase_accum

            # update phase accumulator by interpolating true frequency
            if t_int < frames - 2:
                freq_interp = (1-alpha) * true_freq[..., t_int] + alpha * true_freq[..., t_int+1]
            else:
                freq_interp = true_freq[..., -1]
            phase_accum += freq_interp

        self.last_phase = phase_accum.unsqueeze(-1).clone()

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


# tentativa de I/O: soundfile preferido, fallback para torchaudio
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

    # prepara tensor [B, T]
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



# # Example quick test helper (not executed here, for user's local testing)

# if __name__ == '__main__':
#     import torch
#     sh = PhaseVocoderPitchShift(n_fft=1024, hop_length=256, win_length=1024)
#     x = torch.randn(1, 16000)
#     with torch.no_grad():
#         y = sh(x, n_steps=4)
#     print(x.shape, y.shape)


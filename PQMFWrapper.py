import torch
import torchaudio
import torch.nn as nn
from typing import List, Tuple
from pqmf import CachedPQMF
import os

# Diretórios
script_dir = os.path.dirname(os.path.abspath(__file__))
audio_dir = os.path.join(script_dir, "audio")
os.makedirs(audio_dir, exist_ok=True)
torchscript_dir = os.path.join(script_dir, "torchscript")
os.makedirs(torchscript_dir, exist_ok=True)



class PQMFWrapper(nn.Module):
    """
    Wrapper para PQMF compatível com TorchScript (.ts) e torch.ts~
    Permite decompor e reconstruir sinais mono.
    """
    def __init__(self, attenuation=100, n_band=16, m_buffer_size=512):
        super().__init__()
        self.n_band = n_band
        self.attenuation = attenuation
        self.pqmf = CachedPQMF(attenuation, n_band)
        self._methods = ["forward", "inverse", "process"]
        self._attributes = ["n_band", "attenuation",
                            "forward_in_ch", "forward_out_ch",
                            "inverse_in_ch", "inverse_out_ch",
                            "process_in_ch", "process_out_ch",
                            "m_buffer_size", "max_buffer_size"]
        
        self.forward_in_ch = 1
        self.forward_out_ch = 1
        self.inverse_in_ch = 1
        self.inverse_out_ch = 1
        self.process_in_ch = 1
        self.process_out_ch = 2
        self.m_buffer_size = m_buffer_size
        self.max_buffer_size = 16384

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
    def process(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Decompõe e reconstrói o sinal mono em um único método.
        Args:
            x: Tensor [1, buffer_size] ou [batch, 1, buffer_size]
        Returns:
            Tuple: (subbands, reconstructed)
        """
        subbands = self.forward(x)
        reconstructed = self.inverse(subbands)
        return reconstructed, subbands


# --- Exportação TorchScript ---
if __name__ == "__main__":
    print("Exportando PQMFWrapper para TorchScript...")
    wrapper = PQMFWrapper(attenuation=100, n_band=16, m_buffer_size=8192)
    wrapper.eval()

    # Exporta para TorchScript
    scripted = torch.jit.script(wrapper)
    scripted.save(os.path.join(torchscript_dir, "pqmf.ts"))
    print(f"Modelo salvo como {os.path.join(torchscript_dir, 'pqmf.ts')}")

    # --- Carrega o modelo TorchScript e realiza o teste ---
    print("Carregando modelo TorchScript para teste...")
    loaded = torch.jit.load(os.path.join(torchscript_dir, "pqmf.ts"))
    loaded.eval()

    # Testa o modelo
    wav_path = os.path.join(audio_dir, 'violin_bow_nonvib_f4_44100.wav')
    wav, sr = torchaudio.load(wav_path)
    if wav.dim() == 2 and wav.shape[0] > 1:
        wav = wav[0].unsqueeze(0)  # Usa apenas o primeiro canal

    buffer_size = loaded.m_buffer_size
    total_size = wav.shape[-1]
    pad = (buffer_size - total_size % buffer_size) % buffer_size
    if pad > 0:
        wav = torch.nn.functional.pad(wav, (0, pad))

    print(f"Áudio carregado: shape={wav.shape}, sr={sr}")

    with torch.no_grad():
        subbands = loaded.forward(wav)
        reconstructed = loaded.inverse(subbands)
        process = loaded.process(wav)
        print(f"Sub-bandas shape: {subbands.shape}")
        print(f"Reconstruído shape: {reconstructed.shape}")
        print(f"Process output shapes: {[t.shape for t in process]}")

    reconstructed_2d = reconstructed.squeeze(0).squeeze(0).unsqueeze(0)  # [1, samples]
    torchaudio.save(os.path.join(audio_dir, "reconstruido.wav"), reconstructed_2d.cpu(), sr)
    print(f"Áudio reconstruído salvo em {os.path.join(audio_dir, 'reconstruido.wav')}")
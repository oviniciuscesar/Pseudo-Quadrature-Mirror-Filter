import math

import cached_conv as cc
import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from scipy.optimize import fmin
from scipy.signal import firwin, kaiser_beta, kaiserord



def reverse_half(x):
    """"
    cria uma máscara do mesmo tamanho que o tensor de entrada, onde todos os elementos são 1, 
    exceto os elementos nas posições ímpares da segunda metade, que são -1.
    Multiplica o tensor de entrada pela máscara, invertendo  o sinal dos elementos nas posições ímpares da segunda metade.
    """
    mask = torch.ones_like(x)
    mask[..., 1::2, ::2] = -1

    return x * mask



def center_pad_next_pow_2(x):
    """
    Adiciona padding ao tensor de entrada para que seu comprimento seja a potência de 2 mais próxima.
    """
    next_2 = 2**math.ceil(math.log2(x.shape[-1]))
    pad = next_2 - x.shape[-1]
    return nn.functional.pad(x, (pad // 2, pad // 2 + int(pad % 2)))


def make_odd(x):
    """
    Garante que o comprimento do tensor de entrada seja ímpar, adicionando padding se necessário.
    """
    if not x.shape[-1] % 2:
        x = nn.functional.pad(x, (0, 1))
    return x


def get_qmf_bank(h, n_band):
    """
    Modula o filtro protótipo para criar o banco de filtros QMF
    Parametros
    ----------
    h: torch.Tensor
        filtro protótipo
    n_band: int
        número de sub-bandas
    """
    k = torch.arange(n_band).reshape(-1, 1)
    N = h.shape[-1]
    t = torch.arange(-(N // 2), N // 2 + 1)

    p = (-1)**k * math.pi / 4

    mod = torch.cos((2 * k + 1) * math.pi / (2 * n_band) * t + p)
    hk = 2 * h * mod

    return hk


def kaiser_filter(wc, atten, N=None):
    """
    Calcula um filtro passa-baixa usando a janela de Kaiser
    Parameters
    ----------
    wc: float
        Frequência angular

    atten: float
        Atenuação (dB, positivo)

    N: int
        Tamanho do filtro (deve ser ímpar). Se None, o tamanho mínimo
        que satisfaz a atenuação será usado.
    """
    N_, beta = kaiserord(atten, wc / np.pi)
    N_ = 2 * (N_ // 2) + 1
    N = N if N is not None else N_
    h = firwin(N, wc, window=('kaiser', beta), scale=False, fs=2 * np.pi)
    return h


def loss_wc(wc, atten, M, N):
    """
    Computes the objective described in https://ieeexplore.ieee.org/document/681427
    """
    h = kaiser_filter(wc, atten, N)
    g = np.convolve(h, h[::-1], "full")
    g = abs(g[g.shape[-1] // 2::2 * M][1:])
    return np.max(g)


def get_prototype(atten, M, N=None):
    """
    Dado a atenuação e o número de sub-bandas, calcula o filtro protótipo
    Parameters
    ----------
    atten: float
        Atenuação (dB, positivo)
    M: int
        Número de sub-bandas
    N: int
        Tamanho do filtro (deve ser ímpar). Se None, o tamanho mínimo
        que satisfaz a atenuação será usado.
    """
    wc = fmin(lambda w: loss_wc(w, atten, M, N), 1 / M, disp=0)[0]
    return kaiser_filter(wc, atten, N)


def polyphase_forward(x, hk, rearrange_filter=True):
    """
    Implementação do processo de análise (rápido) usando a estrutura polifásica
    Parameters
    ----------
    x: torch.Tensor
        sinal a ser analisado ( B x 1 x T )

    hk: torch.Tensor
        banco de filtros ( M x T )
    """
    x = rearrange(x, "b c (t m) -> b (c m) t", m=hk.shape[0])
    if rearrange_filter:
        hk = rearrange(hk, "c (t m) -> c m t", m=hk.shape[0])
    x = nn.functional.conv1d(x, hk, padding=hk.shape[-1] // 2)[..., :-1]
    return x


def polyphase_inverse(x, hk, rearrange_filter=True):
    """
    Implementação do processo de síntese (rápido) usando a estrutura polifásica
    Parameters
    ----------
    x: torch.Tensor
        sinal a ser sintetizado ( B x 1 x T )

    hk: torch.Tensor
        banco de filtros ( M x T )
    """

    m = hk.shape[0]

    if rearrange_filter:
        hk = hk.flip(-1)
        hk = rearrange(hk, "c (t m) -> m c t", m=m)  # polyphase

    pad = hk.shape[-1] // 2 + 1
    x = nn.functional.conv1d(x, hk, padding=int(pad))[..., :-1] * m

    x = x.flip(1)
    x = rearrange(x, "b (c m) t -> b c (t m)", m=m)
    x = x[..., 2 * hk.shape[1]:]
    return x


def classic_forward(x, hk):
    """
    Implementação simples do processo de análise (lento)
    Parameters
    ----------
    x: torch.Tensor
        sinal a ser analisado ( B x 1 x T )

    hk: torch.Tensor
        banco de filtros ( M x T )
    """
    x = nn.functional.conv1d(
        x,
        hk.unsqueeze(1),
        stride=hk.shape[0],
        padding=hk.shape[-1] // 2,
    )[..., :-1]
    return x


def classic_inverse(x, hk):
    """
    Implementação simples do processo de síntese (lento)
    Parameters
    ----------
    x: torch.Tensor
        sinal a ser sintetizado ( B x 1 x T )

    hk: torch.Tensor
        banco de filtros ( M x T )
    """
    hk = hk.flip(-1)
    y = torch.zeros(*x.shape[:2], hk.shape[0] * x.shape[-1]).to(x)
    y[..., ::hk.shape[0]] = x * hk.shape[0]
    y = nn.functional.conv1d(
        y,
        hk.unsqueeze(0),
        padding=hk.shape[-1] // 2,
    )[..., 1:]
    return y


@torch.fx.wrap
class PQMF(nn.Module):
    """
    Decomposição e reconstrução de sinais usando um banco de filtros QMF
    Implementa tanto a versão clássica (lenta) quanto a polifásica (rápida)
    Parameters
    ----------
    attenuation: int
        Atenuação das bandas rejeitadas (dB, 80 - 120)
    n_band: int
        Número de bandas, deve ser uma potência de 2 se a implementação polifásica
        for necessária
    """

    def __init__(self, attenuation, n_band, polyphase=True, n_channels = 1):
        super().__init__()
        h = get_prototype(attenuation, n_band)

        if polyphase:
            power = math.log2(n_band)
            assert power == math.floor(
                power
            ), "quando usar o algoritmo polifásico, n_band deve ser uma potência de 2"

        h = torch.from_numpy(h).float()
        hk = get_qmf_bank(h, n_band)
        hk = center_pad_next_pow_2(hk)

        self.register_buffer("hk", hk)
        self.register_buffer("h", h)
        self.n_band = n_band
        self.polyphase = polyphase
        self.n_channels = n_channels

    """"
    Decomposição do sinal de entrada em n_band sub-bandas
    Parameters
    ----------
    x: torch.Tensor
        sinal a ser analisado ( B x 1 x T ) ou ( B x n_channels x T )
    Returns
    -------
    torch.Tensor
        Sinal analisado ( B x n_band x T' ) ou ( B x n_channels*n_band x T' )
    """
    def forward(self, x):
        if x.ndim == 2:
            return torch.stack([self.forward(x[i]) for i in range(x.shape[0])])
        if self.n_band == 1:
            return x
        elif self.polyphase:
            x = polyphase_forward(x, self.hk)
        else:
            x = classic_forward(x, self.hk)

        x = reverse_half(x)

        return x

    """
    Reconstrução do sinal original a partir das sub-bandas
    Parameters
    ----------
    x: torch.Tensor
        sinal a ser sintetizado ( B x n_band x T' ) ou ( B x n_channels*n_band x T' )
    Returns
    -------
    torch.Tensor
        Sinal sintetizado ( B x 1 x T ) ou ( B x n_channels x T )
    """
    def inverse(self, x):
        if x.ndim == 2:
            if self.n_channels == 1:
                return self.inverse(x[0]).unsqueeze(0)
            else:
                x = x.split(self.n_channels, -2)
                return torch.stack([self.inverse(x[i]) for i in len(x)])

        if self.n_band == 1:
            return x

        x = reverse_half(x)

        if self.polyphase:
            return polyphase_inverse(x, self.hk)
        else:
            return classic_inverse(x, self.hk)



""""
    Versão com convoluções cacheadas para acelerar a análise e síntese
    Note que a convolução cacheada não suporta múltiplos canais
    Parameters
    ----------
    attenuation: int
        Atenuação das bandas rejeitadas (dB, 80 - 120)
    n_band: int
        Número de bandas, deve ser uma potência de 2 se a implementação polifásica
        for necessária
    ----------------------------------------------------------------------
    Atenção: a convolução cacheada não suporta múltiplos canais
    ----------------------------------------------------------------------
"""
class CachedPQMF(PQMF):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        hkf = make_odd(self.hk).unsqueeze(1)

        hki = self.hk.flip(-1)
        hki = rearrange(hki, "c (t m) -> m c t", m=self.hk.shape[0])
        hki = make_odd(hki)

        self.forward_conv = cc.Conv1d(
            hkf.shape[1],
            hkf.shape[0],
            hkf.shape[2],
            padding=cc.get_padding(hkf.shape[-1]),
            stride=hkf.shape[0],
            bias=False,
        )
        self.forward_conv.weight.data.copy_(hkf)
 
        self.inverse_conv = cc.Conv1d(
            hki.shape[1],
            hki.shape[0],
            hki.shape[-1],
            padding=cc.get_padding(hki.shape[-1]),
            bias=False,
        )
        self.inverse_conv.weight.data.copy_(hki)

    def script_cache(self):
        self.forward_conv.script_cache()
        self.inverse_conv.script_cache()

    def forward(self, x):
        if self.n_band == 1: return x
        x = self.forward_conv(x)
        x = reverse_half(x)
        return x

    def inverse(self, x):
        if self.n_band == 1: return x
        x = reverse_half(x)
        m = self.hk.shape[0]
        x = self.inverse_conv(x) * m
        x = x.flip(1)
        x = x.permute(0, 2, 1)
        x = x.reshape(x.shape[0], x.shape[1], -1, m).permute(0, 2, 1, 3)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        return x 
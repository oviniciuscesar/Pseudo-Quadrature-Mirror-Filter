import os
import torch
import torchaudio
import argparse
import math

script_dir = os.path.dirname(os.path.abspath(__file__))
audio_dir = os.path.join(script_dir, "audio")
torchscript_dir = os.path.join(script_dir, "torchscript")

def load_mono(path):
    wav, sr = torchaudio.load(path)  # [C, T]
    if wav.dim() == 2 and wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    return wav.float(), sr

def pad_to_multiple(x, block):
    L = x.shape[-1]
    pad = (block - (L % block)) % block
    if pad > 0:
        x = torch.nn.functional.pad(x, (0, pad))
    return x, pad

def main():
    p = argparse.ArgumentParser()
    p.add_argument("input", help="arquivo de entrada (wav) em audio/")
    p.add_argument("--block", type=int, default=4096, help="tamanho do bloco (PD)")
    p.add_argument("--overlap", type=int, default=None, help="número de amostras de overlap entre blocos (default block//2)")
    p.add_argument("--ts", type=str, default=os.path.join(torchscript_dir, "pqmfpvoc.ts"), help="caminho do .ts")
    p.add_argument("--out_prefix", type=str, default="_pdstream", help="prefixo de saída em audio/")
    args = p.parse_args()

    wav, sr = load_mono(args.input)
    # wav, pad = pad_to_multiple(wav, args.block)
    # window = torch.hann_window(args.block, dtype=wav.dtype, device=wav.device).unsqueeze(0)  # shape [1, block]
    # total_len = wav.shape[-1]
    overlap = args.overlap if args.overlap is not None else (args.block // 2)
    if overlap < 0 or overlap >= args.block:
        raise ValueError("overlap deve estar em [0, block-1]")
    hop = args.block - overlap

    L = wav.shape[-1]
    if L <= args.block:
        n_frames = 1
    else:
        n_frames = int(math.ceil((L - args.block) / float(hop))) + 1
    total_needed = (n_frames - 1) * hop + args.block
    pad = total_needed - L
    if pad > 0:
        wav = torch.nn.functional.pad(wav, (0, pad))

    # janela Hann do tamanho do bloco
    window = torch.hann_window(args.block, dtype=wav.dtype, device=wav.device).unsqueeze(0)  # shape [1, block]
    total_len = wav.shape[-1]
    print(f"Loaded {args.input}: shape={wav.shape}, sr={sr}, pad={pad}")

    # carrega modelo TorchScript
    loaded = torch.jit.load(args.ts)
    loaded.eval()

    # stream em blocos
    # out_blocks = []
    # recon_blocks = []

    out_accum = torch.zeros(1, total_len, dtype=wav.dtype, device=wav.device)
    norm_accum = torch.zeros_like(out_accum)
    recon_accum = torch.zeros_like(out_accum)

    # with torch.no_grad():
    #     for i in range(0, total_len, args.block):
    #         blk = wav[:, i:i+args.block]  # [1, block]
    #         blk = blk * window          # aplicar janela
    #         # chama pitchshifter (processamento por bloco)
    #         try:
    #             out = loaded.pitchshifter(blk)   # espera [B,1,T] output
    #         except Exception as e:
    #             # fallback: tentar forward+inverse como sanity check
    #             print(f"pitchshifter falhou no bloco {i}: {e}; tentando forward+inverse")
    #             sub = loaded.forward(blk)        # [1, n_band, T_sub]
    #             out = loaded.inverse(sub)       # [1,1,T]
    #         # garantir shape [1, T]
    #         if out.dim() == 3 and out.shape[1] == 1:
    #             out = out.squeeze(1)
    #         out_blocks.append(out)

    #         # opcional: testar apenas forward->inverse roundtrip
    #         subbands = loaded.forward(blk)
    #         rec = loaded.inverse(subbands)
    #         if rec.dim() == 3 and rec.shape[1] == 1:
    #             rec = rec.squeeze(1)
    #         recon_blocks.append(rec)

    # # concatena blocos e remove padding
    # pitch_stream = torch.cat(out_blocks, dim=-1)[:, : wav.shape[-1] - pad]
    # recon_stream = torch.cat(recon_blocks, dim=-1)[:, : wav.shape[-1] - pad]

    with torch.no_grad():
        for frame_idx in range(n_frames):
            i = frame_idx * hop
            blk = wav[:, i:i+args.block]  # [1, block]
            blk_win = blk * window        # aplicar janela de análise

            # chama pitchshifter (processamento por bloco)
            try:
                out = loaded.pitchshifter(blk_win)   # espera [B, T] ou [B,1,T]
            except Exception as e:
                print(f"pitchshifter falhou no bloco {i}: {e}; tentando forward+inverse")
                sub = loaded.forward(blk_win)
                out = loaded.inverse(sub)

            # normalizar shapes: queremos [1, block]
            if out.dim() == 3 and out.shape[1] == 1:
                out = out.squeeze(1)
            if out.dim() == 2 and out.size(1) != args.block:
                # trunc/pad centralizado para block (segurança)
                cur = out.size(1)
                if cur > args.block:
                    start = (cur - args.block) // 2
                    out = out[:, start:start+args.block]
                else:
                    pad_l = (args.block - cur) // 2
                    pad_r = args.block - cur - pad_l
                    out = torch.nn.functional.pad(out, (pad_l, pad_r))

            # acumular com janela de síntese (usar mesma janela)
            out_accum[:, i:i+args.block] += out * window
            norm_accum[:, i:i+args.block] += (window * window)

            # opcional: forward->inverse roundtrip para avaliação
            subbands = loaded.forward(blk_win)
            rec = loaded.inverse(subbands)
            if rec.dim() == 3 and rec.shape[1] == 1:
                rec = rec.squeeze(1)
            recon_accum[:, i:i+args.block] += rec * window
    # finalizar OLA: evitar divisão por zero
    eps = 1e-8
    pitch_stream_full = out_accum / (norm_accum + eps)
    recon_stream_full = recon_accum / (norm_accum + eps)

    # remover padding adicionado antes (pad calculado acima)
    pitch_stream = pitch_stream_full[:, : (total_len - pad)]
    recon_stream = recon_stream_full[:, : (total_len - pad)]

    # processamento full (não por blocos) para comparação
    with torch.no_grad():
        full_in = wav.clone()
        full_out = None
        try:
            full_out = loaded.pitchshifter(full_in)
            if full_out.dim() == 3 and full_out.shape[1] == 1:
                full_out = full_out.squeeze(1)
            full_out = full_out[:, : wav.shape[-1] - pad]
        except Exception as e:
            print("pitchshifter(full) falhou:", e)
            full_out = None

    # salvar resultados
    os.makedirs(audio_dir, exist_ok=True)
    torchaudio.save(os.path.join(audio_dir, f"{args.out_prefix}_stream_pitch.wav"), pitch_stream.cpu(), sr)
    torchaudio.save(os.path.join(audio_dir, f"{args.out_prefix}_stream_recon.wav"), recon_stream.cpu(), sr)
    print("Saved stream outputs to audio/")

    if full_out is not None:
        torchaudio.save(os.path.join(audio_dir, f"{args.out_prefix}_full_pitch.wav"), full_out.cpu(), sr)
        print("Saved full-file pitch output to audio/")

    # imprimir métricas simples
    def rms(t):
        return float(torch.sqrt(torch.mean(t**2)))
    orig = wav[:, : wav.shape[-1] - pad]
    print("RMS orig:", rms(orig))
    print("RMS stream_pitch:", rms(pitch_stream))
    print("RMS stream_recon:", rms(recon_stream))
    if full_out is not None:
        print("RMS full_pitch:", rms(full_out))

if __name__ == "__main__":
    main()
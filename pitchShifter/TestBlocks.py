import os
import torch
import torchaudio
import argparse

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
    p.add_argument("--ts", type=str, default=os.path.join(torchscript_dir, "pqmfpvoc.ts"), help="caminho do .ts")
    p.add_argument("--out_prefix", type=str, default="_pdstream", help="prefixo de saída em audio/")
    args = p.parse_args()

    wav, sr = load_mono(args.input)
    wav, pad = pad_to_multiple(wav, args.block)
    total_len = wav.shape[-1]
    print(f"Loaded {args.input}: shape={wav.shape}, sr={sr}, pad={pad}")

    # carrega modelo TorchScript
    loaded = torch.jit.load(args.ts)
    loaded.eval()

    # stream em blocos
    out_blocks = []
    recon_blocks = []
    with torch.no_grad():
        for i in range(0, total_len, args.block):
            blk = wav[:, i:i+args.block]  # [1, block]
            # chama pitchshifter (processamento por bloco)
            try:
                out = loaded.pitchshifter(blk)   # espera [B,1,T] output
            except Exception as e:
                # fallback: tentar forward+inverse como sanity check
                print(f"pitchshifter falhou no bloco {i}: {e}; tentando forward+inverse")
                sub = loaded.forward(blk)        # [1, n_band, T_sub]
                out = loaded.inverse(sub)       # [1,1,T]
            # garantir shape [1, T]
            if out.dim() == 3 and out.shape[1] == 1:
                out = out.squeeze(1)
            out_blocks.append(out)

            # opcional: testar apenas forward->inverse roundtrip
            subbands = loaded.forward(blk)
            rec = loaded.inverse(subbands)
            if rec.dim() == 3 and rec.shape[1] == 1:
                rec = rec.squeeze(1)
            recon_blocks.append(rec)

    # concatena blocos e remove padding
    pitch_stream = torch.cat(out_blocks, dim=-1)[:, : wav.shape[-1] - pad]
    recon_stream = torch.cat(recon_blocks, dim=-1)[:, : wav.shape[-1] - pad]

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
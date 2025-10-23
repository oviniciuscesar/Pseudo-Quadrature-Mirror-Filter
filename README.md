# Pseudo-Quadrature-Mirror-Filter

Implementation of the PQMF (Pseudo Quadrature Mirror Filter) filter, used to decompose signals into sub-bands and reconstruct them. It's primarily intended for digital audio signal processing and machine learning applications. The code is an adaptation of the PQMF used in [acids-ircam's RAVE models](https://github.com/acids-ircam/RAVE).

## Overview

The PQMF is implemented at `pqmf.py` which includes:

- **Standard Implementation:** A classic implementation of PQMF filter.
- **Polyphasic Implementation:** A more efficient polyphase version.
- **Optimized Real-Time Implementation:** Uses cached convolution layers, making it suitable for low-latency, real-time processing.

Other implementations and wrappers:

- **TorchScript Compatible:** Ready for serialized, production-ready models.
- **[conTorchionist](https://github.com/ecrisufmg/contorchionist) compatibles wrappers:** Allows for easy integration into interactive and experimental workflows.
- **Pitch Shifter implementations**: A phase-vocoder-based pitch shifter and a torchaudio implementation

### Pitch Shifter with Phase-vocoder:

Applies a different phase-vocoder-based pitch shifter to every PQMF sub-band and reconstruct the signal

##### `PitchShifterPvoc` directory

`1-PitchShifterWrapper.py` creates a wrapper and exports TorchScript model compatible with conTorchionist

`2-TestBlocks.py <audio file name> --block 4096 --overlap 2048` tests the exported model simulating a real-time processing scenario with overlapping blocks, and save the resulting audio files. You can change `--block` and `--overlap` to testing different scenarios (block processing and overlapping logic isn't working properly yet)

### Pitch shifter with torchaudio implementation:

##### `PitchShifterTorchaudio` directory

Applies a pitch shifter (torchaudio implementation) to every PQMF sub-band and reconstruct the signal

`PQMFPsWrapper.py` (creates a wrapper, runs a test and save the resulting audio file)

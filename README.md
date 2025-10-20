# Pseudo-Quadrature-Mirror-Filter

Implementation of the PQMF (Pseudo Quadrature Mirror Filter) filter, used for decomposition and reconstruction of signals into sub-bands, focusing on digital signal processing and machine learning applications. Includes versions optimized for real-time processing, as well as TorchScript and Pure Data-compatible wrappers, allowing direct integration into interactive and experimental workflows. The code is an adaptation of the PQMF used by [acids-ircam's RAVE models](https://github.com/acids-ircam/RAVE).

1 - Pitch Shifter with Phase-vocoder run:
1-PitchShifterWrapper.py (create the wrapper and export the torchscript model)

2-TestBlocks.py flute.wav --block 4096 --overlap 2048 (test the exported model simulating real-time processing)

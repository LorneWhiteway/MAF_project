# Material for presentation at UCL on MAF algorithm, 27 April 2020

For information on MAF see [Papamakarios et al., Masked Autoregressive Flow for Density Estimation](https://arxiv.org/abs/1705.07057).

This repository contains:

1. A presentation on MAF
2. A python3 program `maf.py` to illustrate how MAF works. This program was used to create the plots in the presentation.

Notes on the python program:
* Uses python3.
* Uses [ChainConsumer](https://samreay.github.io/ChainConsumer/) for plotting.
* While optimising it reports to the screen the current 'negative log probability', normalised to be near unity (would be exactly unity in the case of a perfect Gaussian).
* Currently no regularisation term in the loss function; this could be added.
* Can easily be amended to model alternative distributions.
* After each MAF step (i.e. one run of MADE) the training data are rotated through 1.0 radians.



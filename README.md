# Learning Linear-Quadratic Regulators without Stabilizing Parameter Sets via Thompson Sampling with Preconditioned Langevin Dynamics

This repository includes Matlab implementation of the experimental result of the paper titled above. 

1. Preparation

Generation of asymmetric noises (5 and 10 - dimensional ) should be preceded.
Root folder: asymmetric&gaussian_mixture
   * Run TSLD-LQ-10D_asymmetric - ULA_asymmetric_10D
   * Run TSLD-LQ-5D_asymmetric - ULA_asymmetric_5D
   * Run TSLD-LQ-3D_asymmetric - ULA_asymmetric_3D

Running (1), (2), and (3) will create csv file with artificial noises.

2. Main
Root folder: asymmetric&gaussian_mixture
(1) Gaussian mixture case
  * Run TSLD-LQ-10D_gaussian_mixture - gaussian_mixture_10D.m
  * Run TSLD-LQ-5D_gaussian_mixture - gaussian_mixture_5D.m
  * Run TSLD-LQ-3D_gaussian_mixture - gaussian_mixture_3D.m

(2) Asymmetric case
  * Run TSLD-LQ-10D_asymmetric - asymmetric_10D.m
  * Run TSLD-LQ-5D_asymmetric - asymmetric_5D.m
  * Run TSLD-LQ-3D_asymmetric - asymmetric_3D.m


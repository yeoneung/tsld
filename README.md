# Learning Linear-Quadratic Regulators without Stabilizing Parameter Sets via Thompson Sampling with Preconditioned Langevin Dynamics

This repository includes Matlab implementation of the experimental result of the paper titled above. 

## 1. Preparation

Generation of asymmetric noises (5 and 10 - 10-dimensional) should be preceded.

--Root folder: asymmetric&gaussian_mixture--

   * Run TSLD-LQ-10D_asymmetric - ULA_asymmetric_10D
   * Run TSLD-LQ-5D_asymmetric - ULA_asymmetric_5D
   * Run TSLD-LQ-3D_asymmetric - ULA_asymmetric_3D

Running them will create a csv file with artificial noises in the current folder associated.

## 2. Main

--Root folder: asymmetric&gaussian_mixture--

(1) Gaussian mixture case
  * Run TSLD-LQ-10D_gaussian_mixture - gaussian_mixture_10D.m
  * Run TSLD-LQ-5D_gaussian_mixture - gaussian_mixture_5D.m
  * Run TSLD-LQ-3D_gaussian_mixture - gaussian_mixture_3D.m

(2) Asymmetric case
  * Run TSLD-LQ-10D_asymmetric - asymmetric_10D.m
  * Run TSLD-LQ-5D_asymmetric - asymmetric_5D.m
  * Run TSLD-LQ-3D_asymmetric - asymmetric_3D.m

## 3. Comparison

(1) Comparison of Langevin iteration (ULA vs preconditioned ULA)
  * Run comparison - Comparison_Langevin_iteration.m
  * The result is
![iteration](https://github.com/yeoneung/tsld/assets/102267531/77255a86-0117-4ee8-90aa-7c39c0c7e644)


(2) Comparison between our method and PSRL-LQ (https://ieeexplore.ieee.org/document/8884712)
  * Run comparison - Comparison_regret.m
  * The result is
![regret](https://github.com/yeoneung/tsld/assets/102267531/f40bc00c-14e4-4b39-af08-ab9c0247c132)


# Approximate Thompson Sampling for Learning Linear Quadratic Regulators with $O(\sqrt{T})$ Regret

## 1. Requirement

The code is written based on MATLAB R2021a.

* [MATLAB](https://www.mathworks.com),

which is free for students and academics.

In addition to that, the following toolboxes are needed:

* [Statistics and Machine Learning Toolbox](https://mathworks.com/products/statistics.html)
* [Control System Toolbox](https://mathworks.com/products/control.html)

## 2. Regret of system with non-Gaussian noise

Add an 'auxiliary' folder to the path where you intend to run the file. 


### 2.1 Noise generation - Gaussian mixture noise

In the case of Gaussian mixture noise, run the file:
`gaussian_mixture_3D.m`, `gaussian_mixture_5D.m`, `gaussian_mixture_10D.m`.

### 2.2 Noise generation - Asymmetric noise

In the case of asymmetric noise, the generation of asymmetric noises (3,5,10-dimensional) should be preceded.
Following files will create a csv file with artificial noises in the current folder associated: 
`ULA_asymmetric_3D`, `ULA_asymmetric_5D`, `ULA_asymmetric_10D`

To simulate the regret of system with asymmetric noise, run following files: 
`asymmetric_3D.m`, `asymmetric_5D.m`, `asymmetric_10D.m`.


### 3. Comparison

### 3.1 Comparison of Langevin iteration (ULA vs preconditioned ULA)
To run the simulation using naive ULA, activate following codes in `Comparison_Langevin_iteration.m` :
```
%naive ULA
writematrix(iteration,'ULA_iteration.csv');
```

```
%naive ULA
step_size = (M*min(eig(preconditioner)))/(16*L^2*(max(eig(preconditioner)))^2);
step_iteration = ceil(64*(max(eig(preconditioner)))^2/(min(eig(preconditioner)))^2);
```

```
%In case of using ULA
scaled_grad_U = grad_U;
```


To run the simulation using preconditioned ULA, activate following codes in `Comparison_Langevin_iteration.m`:

```
%preconditioner
writematrix(iteration,'preconditioned_ULA_iteration.csv');
```

```
%preconditioned ULA
step_size = (M*min(eig(preconditioner)))/(16*L^2*max(t_k,min(eig(preconditioner))));
step_iteration = ceil(4*(log2(max(t_k,min(eig(preconditioner)))/min(eig(preconditioner)))/(M*step_size)));
```

```
%In case of using preconditioner
scaled_grad_U = inv_preconditioner*grad_U;
```

### 3.2 Comparison between our method and PSRL-LQ (https://ieeexplore.ieee.org/document/8884712)
To compare our method with PSRL-LQ, run `Comparison_regret.m`. You will get `PSRL.csv` for PSRL-LQ and `TSLD.csv` for our method.


### 4. Double oscillator

After running the files, two types of csv files remain:

`DIMENSION-time.csv` and `DIMENSION-iter.csv` are real computation time and number of step iteration for convergence of Langevin dynamics.


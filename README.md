# GR4_models

implement of (sub) daily rainfall-runoff model (GR4H)

- The python version is written by torch.
- The Julia version is written by torch (using DifferentialEquation.jl).

The GR4H model has been tested by the data from https://hydrogr.github.io/airGR/index.html, see examples
however, because the route func implementation is different from the original GR4H, thus there are some error in two predictions.

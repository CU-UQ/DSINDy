# DSINDy: Derivative-based sparse identification of nonlinear dynamics

This repository contains the code used to run DSINDy and other algorithms as described in: 

Wentz, Jacqueline, and Alireza Doostan. "Derivative-based SINDy (DSINDy): Addressing the challenge of discovering governing equations from noisy data." arXiv preprint arXiv:2211.05918 (2022).

## Setting up environment

This code uses the packages specified in [requirements.yml](requirements.yml) and the environment can be set up using conda as follows:

```
conda env create -f environment.yml
conda activate dsindy-env
``` 
Note that within the [requirements.yml](requirements.yml), the user must specify the path to the DSINDy package.

A mosek license is required to run the code. A personal academic license can be obtained from <https://www.mosek.com/products/academic-licenses/>. I place the license in the '/home/jacqui/mosek/' directory.

## Example script

An example of running DSINDy is given in the notebook [notebooks/example_run_of_DSINDy.py](notebooks/example_run_of_DSINDy.py).
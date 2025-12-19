This repository contains the reference implementation of FDA-AS used in the paper
"Fractal Decomposition Method for Comparison of Low-Dimensional Black-Box Optimization".

To reproduce the results reported in the article, simply run:
  python analyze_results.py

To reproduce the specific case studies presented in the paper, please find the corresponding code for each case in the study package and run the associated script.

To generate new experiments on the COCO/BBOB benchmark, run:
  python run_experiment.py

In run_experiment.py, users can select the optimization budget and the optimizers to benchmark.
 

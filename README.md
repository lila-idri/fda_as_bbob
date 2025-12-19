This repository contains the reference implementation of FDA-AS used in the paper
"Fractal Decomposition Method for Comparison of Low-Dimensional Black-Box Optimization".

To reproduce the results reported in the article, run:
python analyze_results.py

To reproduce the specific case studies presented in the paper, use the corresponding
scripts located in the study package.

To generate new experiments on the COCO/BBOB benchmark, run:
python run_experiment.py

In run_experiment.py, users can select the optimization budget and the optimizers
to benchmark.

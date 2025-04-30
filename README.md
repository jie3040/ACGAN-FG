# ACGAN-FG
The implementation of ACGAN-FG for zero-shot fault diagnosis.  

# Paper
W. Liao, L. Wu, S. Xu and S. Fujimura, "A Novel Zero-Shot Learning Method With Feature Generation for Intelligent Fault Diagnosis," in IEEE Transactions on Industrial Informatics, vol. 21, no. 4, pp. 3386-3395, April 2025, doi: 10.1109/TII.2025.3526478.

# Dataset
This example is based on the Tennessee-Eastman Process (TEP)  dataset.

# Requirements 
```
python = 3.9.20
Tensorflow = 2.10.0
pandas = 1.3.5
scikit-learn = 1.0.2                                                                                                     `
```
The codes was tested with tensorflow 2.10.0, CUDA driver 12.7, CUDA toolkit 11.8, Ubuntu 22.04.

# Quick strat

```
python ACGAN-FG.py
```
No need to download the whole dataset, we readed and saved in .npz files

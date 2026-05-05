# NNWFG_LSTM_Reproducibility

This repository provides a MATLAB workflow script for the LSTM-based adaptive weighting module used in the proposed multi-sensor weighted factor graph localization method.

The complete raw experimental dataset is not included in this repository due to project-related restrictions. This repository focuses on the implementation logic of the LSTM adaptive weighting module.

## Repository Structure

```text
NNWFG_LSTM_Reproducibility/
│
├── README.md
│
└── src/
    └── LSTM_adaptive_weighting.m
```

## Scope

This repository only includes the LSTM adaptive weighting module.

The following parts are not included:

- original ROS bag files;
- complete raw sensor data;
- full experimental trajectories;
- RTK satellite-number extraction;
- LiDAR point-cloud density extraction;
- complete weighted factor graph optimization.

The script assumes that residuals and quality indicators have already been obtained from the preprocessing stage.

## Required Input Variables

Before running `src/LSTM_adaptive_weighting.m`, the following variables should be prepared in MATLAB:

```text
err_IMU
err_LiDAR
err_RTK
N_RTK_raw
D_LiDAR_raw
t
```

where:

- `err_IMU`, `err_LiDAR`, and `err_RTK` are the residual sequences of IMU, LiDAR, and RTK.
- `N_RTK_raw` is the synchronized RTK satellite number.
- `D_LiDAR_raw` is the synchronized LiDAR point-cloud density.
- `t` is the time sequence.

## LSTM Input

The LSTM input feature vector is:

```text
u_t = [err_IMU, err_LiDAR, err_RTK, N_RTK, D_LiDAR]^T
```

The first three terms describe sensor residuals, while `N_RTK` and `D_LiDAR` describe RTK and LiDAR observation quality.

## Main Workflow

The MATLAB script performs the following steps:

1. aligns the lengths of all input features;
2. smooths the residual sequences using a moving average window;
3. constructs the raw feature matrix;
4. applies min-max normalization using the training portion only;
5. constructs adaptive confidence labels from residuals;
6. combines RTK satellite number and LiDAR density into the labels;
7. builds LSTM sequence samples with `T = 10` and `stride = 1`;
8. splits the data chronologically into training / validation / testing sets with a ratio of `70% / 15% / 15%`;
9. trains the LSTM model;
10. predicts dynamic weights for IMU, RTK, and LiDAR;
11. evaluates the prediction using RMSE and MAE;
12. plots dynamic weight curves and normalized input features.

## LSTM Network

The network structure is:

```text
sequenceInputLayer(5)
→ lstmLayer(64, OutputMode = 'last')
→ fullyConnectedLayer(32)
→ reluLayer
→ dropoutLayer(0.2)
→ fullyConnectedLayer(3)
→ softmaxLayer
→ regressionLayer
```

The output order is:

```text
[w_IMU, w_RTK, w_LiDAR]
```

## Training Settings

```text
Optimizer: Adam
Initial learning rate: 1e-3
Maximum epochs: 100
Mini-batch size: 128
Validation patience: 10
Shuffle: every epoch
```

## Outputs

Typical outputs include:

```text
Fig_LSTM_dynamic_weight_outputs.png
Fig_LSTM_normalized_input_features.png
NNWFG_LSTM_dynamic_weight_result.mat
LSTM_dynamic_weight_outputs.csv
```

## Data Availability

The complete raw experimental dataset is not publicly released in this repository. The complete data supporting the findings of the study are available from the authors upon reasonable request.

This repository is intended to document the implementation workflow of the LSTM-based adaptive weighting module, rather than to release the full raw-data processing pipeline.

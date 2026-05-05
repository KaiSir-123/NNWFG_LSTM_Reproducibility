# NNWFG_LSTM_Reproducibility

This repository provides a MATLAB workflow example for the LSTM-based adaptive weighting module used in the proposed multi-sensor weighted factor graph localization method.

The complete raw experimental dataset is not included in this repository. Instead, this repository provides the source code structure and implementation workflow used to construct LSTM input features, train the adaptive weighting model, and generate dynamic weights for IMU, RTK, and LiDAR factors.

## Purpose of This Repository

The purpose of this repository is to clarify the implementation process of the LSTM-based adaptive weighting module, including:

- construction of LSTM input features;
- feature normalization;
- adaptive confidence label generation;
- sliding-window sequence construction;
- LSTM network training;
- dynamic weight prediction;
- testing evaluation and visualization.

The provided MATLAB script is intended as an implementation reference. To run it with actual data, users need to prepare the required input variables from their own sensor preprocessing pipeline.

## Repository Structure

```text
NNWFG_LSTM_Reproducibility/
│
├── README.md
│
└── src/
    └── LSTM_adaptive_weighting.m
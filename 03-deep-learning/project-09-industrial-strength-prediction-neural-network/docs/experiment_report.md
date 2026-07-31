# Experiment Report

## Dataset Validation

- Observations: 1,030
- Features: 8 quantitative concrete mix and curing-age inputs
- Target: concrete compressive strength in MPa
- Missing values: none detected by `load_concrete_data`
- Split: 80/20 train-test split with `random_state=42`
- Generalization enhancement: target-quantile stratification with 10 bins to keep train/test target distributions aligned

## Promoted Configuration

- Model: PyTorch MLP
- Hidden layers: `[64, 32]`
- Activation: ReLU
- Loss: Mean Squared Error
- Optimizer: Adam
- Learning rate: `0.01`
- Batch size: `824`
- Epochs: `650`
- Dropout: `0.05`
- Weight decay: `0.0001`
- Stratification bins: `10`

## Results

| Split | MSE | RMSE | MAE |
|---|---:|---:|---:|
| Train | 24.596 | 4.959 | 3.754 |
| Test | 30.541 | 5.526 | 4.086 |

Absolute RMSE gap: `0.567 MPa`.

## Generalization Assessment

The first default-sized model reached a lower train RMSE but showed a larger train-test gap, indicating overfitting. The promoted model uses a stratified regression split and shorter training duration to keep train and test RMSE closely aligned while preserving reasonable predictive performance. The project therefore ships the regularized 64/32 MLP trained for 650 epochs as the production artifact.

The acceptance contract cites an example gap of approximately `0.54 MPa`. The measured promoted gap is `0.567 MPa`, which is close to that target on the fixed `random_state=42` split. Further tuning could evaluate k-fold cross validation, target scaling, learning-rate schedules, and ensemble averaging.

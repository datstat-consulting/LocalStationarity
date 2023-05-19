# LocalStationarity

This package provides a set of tools for the analysis of locally stationary processes. It includes functionalities for estimating the evolutionary spectral density, performing hypothesis testing for stationarity, and selecting the best model for the process based on information criteria. The package aims to help researchers and practitioners in various fields, such as finance, engineering, and environmental science, to analyze time series data that exhibit local stationarity.

## Main Features

- Estimate the evolutionary spectral density of a locally stationary process.
- Perform hypothesis testing to determine the stationarity of a time series at each time point.
- Apply model selection techniques, such as the Akaike Information Criterion (AIC), to choose the best model for the process.
- Generate predictions for both training and testing data using the selected model.

## Installation

You can install the package using pip:
```
pip install git+https://github.com/datstat-consulting/LocalStationarity
```
This package relies on some classes from `EconmetPerceptron`:
```
pip install git+https://github.com/datstat-consulting/EconmetPerceptron
```

# Examples
## Bivariate time series data

Import needed libraries, and generate synthetic data.
```
import torch
import matplotlib.pyplot as plt
from scipy import signal

torch.manual_seed(42)

n = 1000
t = torch.linspace(0, 10, n)
x1 = torch.sin(2 * 3.14159265359 * t) + 0.5 * torch.randn(n)
x2 = torch.cos(2 * 3.14159265359 * t) + 0.5 * torch.randn(n)
data = torch.stack((x1, x2), dim=1)
```
Split the data into testing and training.
```
train_ratio = 0.8
n_train = int(n * train_ratio)
train_data = data[:n_train, :]
test_data = data[n_train:, :]
```
Perform model estimation
```
# Create an instance of the LocallyStationaryProcess class for training data
lsp_train = LocallyStationaryProcess(train_data)

# Estimate the evolutionary spectral density
window_size = 50
spectral_density = lsp_train.estimate_evolutionary_spectral_density(window_size, method='periodogram') #other option: 'welch'

# Perform hypothesis testing
hypothesis_testing_results = lsp_train.hypothesis_testing(window_size)

# Perform model selection with the Akaike Information Criterion (AIC)
max_n_lags = 10
best_model = lsp_train.model_selection(window_size, 
                                        max_n_lags, 
                                        criterion='aic') # 'bic' is another choice
n_lags = best_model.shape[1] // train_data.shape[1]

# Generate predictions from the best model
n_windows_train = n_train - window_size + 1
n_windows_test = n - n_train - window_size + 1
predictions = torch.empty_like(data)

# Generate predictions for the training data
for i in range(n_windows_train):
    windowed_data = train_data[i: i + window_size]
    X, _ = WorkhorseFunctions.create_input_output_pairs(windowed_data, n_lags)
    y_pred = X.mm(best_model[i].reshape(n_lags * train_data.shape[1], train_data.shape[1]))
    predictions[i: i + window_size - n_lags, :] = y_pred

# Generate predictions for the test data
for i in range(n_windows_test):
    windowed_data = test_data[i: i + window_size]
    X, _ = WorkhorseFunctions.create_input_output_pairs(windowed_data, n_lags)
    y_pred = X.mm(best_model[n_windows_train - 1].reshape(n_lags * test_data.shape[1], test_data.shape[1]))
    predictions[n_train + i: n_train + i + window_size - n_lags, :] = y_pred
```
Plot the data and the estimated model.
```
plt.figure()
plt.plot(t, data[:, 0], label='x1')
plt.plot(t, data[:, 1], label='x2')
plt.plot(t[n_lags:], predictions[n_lags:, 0], '--', label='x1_pred')
plt.plot(t[n_lags:], predictions[n_lags:, 1], '--', label='x2_pred')
plt.axvline(x=n_train / n * 10, color='r', linestyle='--', label='train-test split')
plt.legend()
plt.show()
```
Plot spectral density and hypothesis test results. The alternative hypothesis is that the time series process is nonstationary for the given point.
```
# Plot the estimated evolutionary spectral density for x1
plt.figure()
plt.imshow(spectral_density[0].T, origin='lower', aspect='auto', cmap='jet', extent=[0, n_train, 0, 25])
plt.colorbar()
plt.title('Estimated Evolutionary Spectral Density for x1')
plt.xlabel('Time')
plt.ylabel('Frequency')
plt.show()

# Plot the estimated evolutionary spectral density for x2
plt.figure()
plt.imshow(spectral_density[1].T, origin='lower', aspect='auto', cmap='jet', extent=[0, n_train, 0, 25])
plt.colorbar()
plt.title('Estimated Evolutionary Spectral Density for x2')
plt.xlabel('Time')
plt.ylabel('Frequency')
plt.show()

# Plot the hypothesis testing results (p-values)
plt.figure()
plt.plot(hypothesis_testing_results[:, 0], label='x1 p-value')
plt.plot(hypothesis_testing_results[:, 1], label='x2 p-value')
plt.axhline(y=0.05, color='r', linestyle='--', label='significance level')
plt.legend()
plt.title('Hypothesis Testing Results (p-values)')
plt.xlabel('Time')
plt.ylabel('p-value')
plt.show()
```
# References
- Dahlhaus, R. (2000). A likelihood approximation for locally stationary processes. The Annals of Statistics, 28(6), 1762-1794.
- Dahlhaus, R. (2012). Locally stationary processes. In Handbook of statistics (Vol. 30, pp. 351-413). Elsevier.

# LocalStationarity
Description WIP

# Examples

```
import torch
import matplotlib.pyplot as plt

torch.manual_seed(42)

# Generate a synthetic dataset
n = 1000
t = torch.linspace(0, 10, n)
x1 = torch.sin(2 * 3.14159265359 * t) + 0.5 * torch.randn(n)
x2 = torch.cos(2 * 3.14159265359 * t) + 0.5 * torch.randn(n)
data = torch.stack((x1, x2), dim=1)

# Create an instance of the LocallyStationaryProcess class
lsp = LocallyStationaryProcess(data)

# Estimate the model parameters with a specific window size and number of lags
window_size = 50
n_lags = 5
model = lsp.estimate_parameters(window_size, n_lags)

# Perform model selection with the Akaike Information Criterion (AIC)
best_model = lsp.model_selection(window_size, max_n_lags=10, criterion='aic')

# Plot the original data and the estimated model
plt.figure()
plt.plot(t, data[:, 0], label='x1')
plt.plot(t, data[:, 1], label='x2')
plt.legend()
plt.show()

```

# References
- Dahlhaus, R. (2000). A likelihood approximation for locally stationary processes. The Annals of Statistics, 28(6), 1762-1794.
- Dahlhaus, R. (2012). Locally stationary processes. In Handbook of statistics (Vol. 30, pp. 351-413). Elsevier.

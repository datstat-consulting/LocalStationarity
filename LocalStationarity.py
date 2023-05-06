import torch
import pandas as pd
from scipy import signal
from statsmodels.tsa.stattools import adfuller
from EconmetPerceptron import WorkhorseFunctions

class LocallyStationaryProcess:
    def __init__(self, time_series_data):
        if not isinstance(time_series_data, torch.Tensor):
            self.data = torch.tensor(time_series_data, dtype=torch.float32)
        else:
            self.data = time_series_data
        self.n = len(time_series_data)
        self.time_varying_parameters = None
        self.model = None

    def estimate_evolutionary_spectral_density(self, window_size, window_type='hann', method='periodogram'):
        n_windows = self.n - window_size + 1
        n_variables = self.data.shape[1]
        window_function = torch.tensor(signal.windows.get_window(window_type, window_size), dtype=torch.float32)
        spectral_density = torch.empty((n_windows, window_size // 2 + 1, n_variables))

        for t in range(n_windows):
            windowed_data = self.data[t : t + window_size] * window_function.unsqueeze(-1)
        
            if method == 'periodogram':
                periodogram = torch.abs(torch.fft.fft(windowed_data, dim=0))**2 / window_size
                spectral_density[t, :, :] = periodogram[:window_size // 2 + 1]

            # Add other estimation methods here, e.g., multitaper, wavelet-based methods, etc.

        self.time_varying_parameters = spectral_density
        return spectral_density

    def estimate_parameters(self, window_size, n_lags):
        n_windows = self.n - window_size + 1
        n_variables = self.data.shape[1]
        self.model = torch.empty((n_windows, n_lags * n_variables, n_variables))

        for t in range(n_windows):
            windowed_data = self.data[t : t + window_size]
            X, y = WorkhorseFunctions.create_input_output_pairs(windowed_data, n_lags)
            beta_hat = WorkhorseFunctions.ols_estimator_torch(X, y)
            self.model[t, :] = beta_hat.squeeze()

        return self.model

    def hypothesis_testing(self, window_size):
        n_windows = self.n - window_size + 1
        n_variables = self.data.shape[1]
        p_values = torch.empty((n_windows, n_variables))

        for t in range(n_windows):
            for var in range(n_variables):
                windowed_data = self.data[t : t + window_size, var].numpy()
                result = adfuller(windowed_data)
                p_values[t, var] = result[1]

        return p_values

    def model_selection(self, window_size, max_n_lags, criterion='aic'):
        n_windows = self.n - window_size + 1
        n_variables = self.data.shape[1]
        best_model = None
        best_criterion_value = float('inf')

        for n_lags in range(1, max_n_lags + 1):
            model = self.estimate_parameters(window_size, n_lags)
            log_likelihood = self.calculate_log_likelihood(window_size, n_lags, model)

            if criterion == 'aic':
                aic = 2 * n_lags * n_variables**2 - 2 * log_likelihood
                if aic < best_criterion_value:
                    best_criterion_value = aic
                    best_model = model

            # Add other model selection criteria here, e.g., BIC, etc.

        self.model = best_model
        return best_model

    def calculate_log_likelihood(self, window_size, n_lags, model):
        n_windows = self.n - window_size + 1
        n_variables = self.data.shape[1]
        log_likelihood = 0

        for t in range(n_windows):
            windowed_data = self.data[t : t + window_size]
            X, y = WorkhorseFunctions.create_input_output_pairs(windowed_data, n_lags)
            y_pred = X.mm(model[t])
            residuals = y - y_pred
            sigma_sq = torch.mean(residuals ** 2, axis=0)
            log_likelihood += -0.5 * window_size * torch.sum(torch.log(sigma_sq))

        return log_likelihood

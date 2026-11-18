# LocalStationarity

Estimate how the spectrum and dynamics of a time series change over time. LocalStationarity provides rolling spectral estimates, local unit-root tests and model selection, vector autoregression (VAR) forecasts, and retrospective and streaming change detection.

The NumPy APIs take a finite vector or a matrix with observations along rows and variables along columns. Spectral estimation assumes regular sampling. Pass the sampling frequency as `sample_rate` to report frequencies in the corresponding units.

## Installation

~~~bash
python -m pip install git+https://github.com/datstat-consulting/LocalStationarity.git
~~~

From a checkout, run `python -m pip install .`. Python 3.10 or later is required. The plotting example uses Matplotlib.

## Bivariate series

This example uses two noisy oscillating series. The second series shifts in level halfway through the record.

~~~python
import numpy as np

from LocalStationarity import (
    LocalVAR,
    LocallyStationaryProcess,
    regime_breaks,
    rolling_backtest,
    rolling_spectrum,
)

rng = np.random.default_rng(42)
sample_rate = 100
t = np.arange(1_000) / sample_rate
data = np.column_stack((
    np.sin(2 * np.pi * t) + 0.5 * rng.normal(size=t.size),
    np.cos(2 * np.pi * t) + 0.5 * rng.normal(size=t.size) + 1.5 * (t >= 5),
))

split = 800
window = 100
train = data[:split]
spectrum = rolling_spectrum(
    train, window, sample_rate=sample_rate, method="welch", step=5
)
breaks = regime_breaks(train, min_size=100, penalty=20)
~~~

`spectrum.power` has shape `(windows, frequencies, variables)`. Each end index is exclusive. Use `method="periodogram"` for a periodogram within each window. `regime_breaks` returns split indices, with each index denoting the first observation of a new segment. Its default cost detects changes in mean. `model="mean_variance"` also allows changes in variance. A larger penalty requires a larger improvement to accept a split.

### Lag selection and forecasts

Use `LocallyStationaryProcess` to compare lag orders by AIC or BIC across training windows. `LocalVAR` fits an intercept in each window and can apply ridge regularization to the lag coefficients.

~~~python
local = LocallyStationaryProcess(train)
coefficients = local.model_selection(window, max_n_lags=8, criterion="aic")
p_values = local.hypothesis_testing(window)

model = LocalVAR(window_size=window, lags=local.selected_lags_, ridge=0.1)
origins, predictions, actual = rolling_backtest(data, model, horizon=5, step=5)
holdout = origins >= split
mae_by_variable = np.abs(predictions[holdout] - actual[holdout]).mean(axis=(0, 1))
next_five = model.fit(data).forecast(steps=5)
~~~

Each forecast origin is both the exclusive end of the training window and the index of the first forecast target. `predictions` and `actual` have shape `(origins, horizon, variables)`. The backtest refits using the observations available at each origin, including earlier holdout observations as they become available. Lag selection uses the initial training portion.

For a fixed-origin evaluation, fit once on the training data and forecast the same five points without refitting. This comparison answers a different forecasting question from the rolling backtest, which incorporates newly observed data.

~~~python
fixed_model = LocalVAR(window_size=window, lags=local.selected_lags_, ridge=0.1)
fixed_forecast = fixed_model.fit(train).forecast(steps=5)
fixed_mae = np.abs(fixed_forecast - data[split:split + 5]).mean(axis=0)
~~~

The fitted model stores its exclusive window ends in `ends_`. To predict from an earlier fitted window, pass that end index as `origin=`. Each window's intercept and lag weights are stored in `coefficients_`.


### Spectral and diagnostic plots

The augmented Dickey–Fuller p-values test for a unit root separately for each variable within each window. Adjacent windows overlap, so their p-values are dependent. Constant windows have missing p-values.

~~~python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, constrained_layout=True)
for j, name in enumerate(("x1", "x2")):
    axes[0, 0].plot(t, data[:, j], alpha=0.6, label=name)
    axes[0, 0].plot(t[origins[holdout]], predictions[holdout, 0, j], "--")
    image = axes[1, j].pcolormesh(
        t[spectrum.ends - 1], spectrum.frequencies, spectrum.power[:, :, j].T,
        shading="auto",
    )
    axes[1, j].set(title=f"{name} power", xlabel="Time (s)", ylabel="Frequency (Hz)")
    fig.colorbar(image, ax=axes[1, j])

axes[0, 0].axvline(t[split], color="black", linestyle=":")
axes[0, 0].set_title("Observations and one-step holdout forecasts")
axes[0, 0].legend()
axes[0, 1].plot(t[window - 1:split], p_values)
axes[0, 1].axhline(0.05, color="black", linestyle=":")
axes[0, 1].set_title("Local unit-root p-values")
plt.show()
~~~

## Streaming regime alerts

~~~python
from LocalStationarity import OnlineRegimeDetector

detector = OnlineRegimeDetector(warmup=40, threshold=8, allowance=0.5)
stream = np.r_[rng.normal(0, 0.5, 160), rng.normal(2, 0.5, 160)]
for observation in stream:
    alert = detector.update(observation)
    if alert is not None:
        print(alert.index, alert.channel, alert.change)
~~~

The detector estimates a baseline from the warmup observations and scores subsequent observations with a two-sided CUSUM. Set `monitor_variance=True` to also track changes in standardized squared deviations. An alert records when the score crossed the threshold. By default, the detector collects a fresh baseline after an alert.

## Existing notebooks

`LocallyStationaryProcess` retains `estimate_evolutionary_spectral_density`, `estimate_parameters`, `hypothesis_testing`, and `model_selection`. It accepts NumPy arrays and floating-point PyTorch tensors. The spectral-density array is ordered by window, frequency, and variable. Use `rolling_spectrum` when the frequency grid and window indices are also needed.

## References

- Dahlhaus, R. (2000). A likelihood approximation for locally stationary processes. *The Annals of Statistics*, 28(6), 1762–1794.
- Dahlhaus, R. (2012). Locally stationary processes. *Handbook of Statistics*, 30, 351–413.

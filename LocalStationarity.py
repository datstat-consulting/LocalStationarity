"""Rolling estimates and piecewise-stationary analysis for time series."""

from dataclasses import dataclass
from numbers import Integral

import numpy as np
from scipy import signal
from statsmodels.tsa.stattools import adfuller

__all__ = ["LocalVAR", "OnlineRegimeDetector", "RegimeAlert", "LocallyStationaryProcess", "RollingSpectrum",
           "rolling_spectrum", "rolling_backtest", "regime_breaks"]


def _series(values):
    if hasattr(values, "detach"):
        values = values.detach().cpu().numpy()
    data = np.asarray(values, dtype=float)
    if data.ndim == 1:
        data = data[:, None]
    if data.ndim != 2 or not all(data.shape) or not np.all(np.isfinite(data)):
        raise ValueError("Provide a finite time-by-variable series.")
    return data


def _positive_integer(value, name):
    if not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer.")


def _windows(data, size, step):
    _positive_integer(size, "window_size")
    _positive_integer(step, "step")
    if size > len(data):
        raise ValueError("window_size exceeds the number of observations.")
    ends = np.arange(size, len(data) + 1, step)
    return ends, (data[end - size:end] for end in ends)


@dataclass(frozen=True)
class RollingSpectrum:
    frequencies: np.ndarray
    ends: np.ndarray
    power: np.ndarray


def rolling_spectrum(values, window_size, *, sample_rate=1.0, step=1,
                     method="periodogram", taper="hann", overlap=0.5):
    """One-sided PSD by window, with end indices exclusive."""
    data = _series(values)
    if not np.isfinite(sample_rate) or sample_rate <= 0 or not 0 <= overlap < 1:
        raise ValueError("Use a positive sample rate and overlap in [0, 1).")
    ends, windows = _windows(data, window_size, step)
    options = dict(fs=sample_rate, window=taper, axis=0, nfft=window_size)
    if method == "periodogram":
        estimate = signal.periodogram
    elif method == "welch":
        estimate = signal.welch
        segment_size = max(1, window_size // 2)
        options.update(nperseg=segment_size, noverlap=int(overlap * segment_size))
    else:
        raise ValueError("method must be 'periodogram' or 'welch'.")
    spectra = [estimate(window, **options) for window in windows]
    return RollingSpectrum(spectra[0][0], ends, np.stack([p for _, p in spectra]))


def _lagged(data, lags, start=None):
    start = lags if start is None else start
    rows = [data[t - lags:t].ravel() for t in range(start, len(data))]
    return np.stack(rows), data[start:]


class LocalVAR:
    """Rolling vector autoregression with an intercept and optional ridge penalty."""

    def __init__(self, window_size, lags=1, ridge=0.0):
        _positive_integer(window_size, "window_size")
        _positive_integer(lags, "lags")
        if window_size <= lags or not np.isfinite(ridge) or ridge < 0:
            raise ValueError("Require window_size > lags and a nonnegative ridge.")
        self.window_size = window_size
        self.lags = lags
        self.ridge = ridge

    def _fit_window(self, window):
        lagged, target = _lagged(window, self.lags)
        design = np.column_stack((np.ones(len(lagged)), lagged))
        if self.ridge:
            penalty = np.eye(design.shape[1]) * np.sqrt(self.ridge)
            penalty[0, 0] = 0
            design = np.vstack((design, penalty))
            target = np.vstack((target, np.zeros((len(penalty), target.shape[1]))))
        return np.linalg.lstsq(design, target, rcond=None)[0]

    def fit(self, values, *, step=1):
        data = _series(values)
        ends, windows = _windows(data, self.window_size, step)
        coefficients = np.stack([self._fit_window(window) for window in windows])
        self.ends_ = ends
        self.coefficients_ = coefficients
        self.history_ = data.copy()
        return self

    def forecast(self, steps=1, *, origin=None):
        """Forecast from a fitted origin using only observations through that origin."""
        _positive_integer(steps, "steps")
        if not hasattr(self, "coefficients_"):
            raise RuntimeError("Fit the model before forecasting.")
        origin = self.ends_[-1] if origin is None else origin
        index = np.searchsorted(self.ends_, origin)
        if index >= len(self.ends_) or self.ends_[index] != origin:
            raise ValueError("origin must be a fitted window end.")
        weights = self.coefficients_[index]
        history = list(self.history_[origin - self.lags:origin])
        for _ in range(steps):
            history.append(np.r_[1, np.asarray(history[-self.lags:]).ravel()] @ weights)
        return np.asarray(history[self.lags:])


def rolling_backtest(values, model, *, horizon=1, step=1):
    """Return forecast origins, forecasts and realized values from rolling fits."""
    _positive_integer(horizon, "horizon")
    data = _series(values)
    if len(data) < model.window_size + horizon:
        raise ValueError("Need at least one window with an observed forecast horizon.")
    model = LocalVAR(model.window_size, model.lags, model.ridge).fit(data[:-horizon], step=step)
    forecasts = np.stack([model.forecast(horizon, origin=end) for end in model.ends_])
    actuals = np.stack([data[end:end + horizon] for end in model.ends_])
    return model.ends_, forecasts, actuals


def regime_breaks(values, *, min_size=20, penalty=10.0, model="mean"):
    """Greedily split segments by within-segment mean or Gaussian variance cost.

    Return zero-based starting indices of new regimes. Standardization makes
    the penalty comparable across units. This is retrospective segmentation.
    """
    data = _series(values)
    _positive_integer(min_size, "min_size")
    if not np.isfinite(penalty) or penalty < 0:
        raise ValueError("penalty must be finite and nonnegative.")
    if model not in {"mean", "mean_variance"}:
        raise ValueError("model must be 'mean' or 'mean_variance'.")
    scale = data.std(axis=0)
    values = (data - data.mean(axis=0)) / np.where(scale > 0, scale, 1)
    totals = np.vstack((np.zeros(data.shape[1]), values.cumsum(axis=0)))
    squares = np.vstack((np.zeros(data.shape[1]), (values**2).cumsum(axis=0)))

    def cost(start, stop):
        count = np.asarray(stop - start)[..., None]
        sse = np.maximum(squares[stop] - squares[start]
                         - (totals[stop] - totals[start])**2 / count, 0)
        if model == "mean":
            return sse.sum(axis=-1)
        return (count * np.log(np.maximum(sse / count, 1e-8))).sum(axis=-1)

    breaks = []
    pending = [(0, len(data))]
    while pending:
        start, stop = pending.pop()
        candidates = np.arange(start + min_size, stop - min_size + 1)
        if not len(candidates):
            continue
        costs = cost(start, candidates) + cost(candidates, stop)
        best = np.argmin(costs)
        if cost(start, stop) - costs[best] > penalty:
            split = int(candidates[best])
            breaks.append(split)
            pending.extend(((start, split), (split, stop)))
    return np.array(sorted(breaks), dtype=int)


class LocallyStationaryProcess:
    """Rolling spectrum, local ADF tests and legacy rolling VAR estimates."""

    def __init__(self, time_series_data):
        self.data = _series(time_series_data)
        self.n = len(self.data)
        self.model = self.time_varying_parameters = None
        self._torch_input = time_series_data if hasattr(time_series_data, "new_tensor") else None

    def _output(self, values):
        if self._torch_input is not None:
            return self._torch_input.new_tensor(values)
        return values

    def estimate_evolutionary_spectral_density(self, window_size, window_type="hann",
                                               method="periodogram", overlap=0.5):
        spectrum = rolling_spectrum(self.data, window_size, method=method,
                                    taper=window_type, overlap=overlap)
        self.time_varying_parameters = self._output(spectrum.power)
        return self.time_varying_parameters

    def estimate_parameters(self, window_size, n_lags):
        _positive_integer(n_lags, "n_lags")
        ends, windows = _windows(self.data, window_size, 1)
        if window_size <= n_lags:
            raise ValueError("window_size must exceed n_lags.")
        coefficients = [np.linalg.lstsq(*_lagged(window, n_lags), rcond=None)[0]
                        for window in windows]
        self.model = self._output(np.stack(coefficients))
        return self.model

    def hypothesis_testing(self, window_size):
        if window_size < 8:
            raise ValueError("Local ADF tests require windows of at least eight observations.")
        _, windows = _windows(self.data, window_size, 1)
        values = [[adfuller(x)[1] if np.ptp(x) else np.nan for x in window.T]
                  for window in windows]
        return self._output(np.asarray(values))

    def model_selection(self, window_size, max_n_lags, criterion="aic"):
        _positive_integer(max_n_lags, "max_n_lags")
        if criterion not in {"aic", "bic"}:
            raise ValueError("criterion must be 'aic' or 'bic'.")
        count = window_size - max_n_lags
        n_variables = self.data.shape[1]
        if count <= max_n_lags * n_variables:
            raise ValueError("Need more observations per window than lag coefficients.")
        _, windows = _windows(self.data, window_size, 1)
        windows = list(windows)
        scores = []
        for lags in range(1, max_n_lags + 1):
            likelihood = 0.
            for window in windows:
                X, y = _lagged(window, lags, start=max_n_lags)
                beta = np.linalg.lstsq(X, y, rcond=None)[0]
                variance = np.mean((y - X @ beta)**2, axis=0)
                likelihood -= 0.5 * count * np.log(variance.clip(1e-15)).sum()
            penalty = 2 if criterion == "aic" else np.log(count)
            scores.append(len(windows) * penalty * lags * n_variables**2 - 2 * likelihood)
        self.selected_lags_ = int(np.argmin(scores) + 1)
        return self.estimate_parameters(window_size, self.selected_lags_)

    def calculate_log_likelihood(self, window_size, n_lags, model):
        _, windows = _windows(self.data, window_size, 1)
        likelihood = 0.
        for window, coefficients in zip(windows, _series_coefficients(model), strict=True):
            X, y = _lagged(window, n_lags)
            variance = np.mean((y - X @ coefficients)**2, axis=0).clip(1e-15)
            likelihood -= 0.5 * len(y) * (np.log(2 * np.pi * variance) + 1).sum()
        return likelihood


def _series_coefficients(values):
    if hasattr(values, "detach"):
        return values.detach().cpu().numpy()
    return np.asarray(values)


@dataclass(frozen=True)
class RegimeAlert:
    index: int
    regime: int
    channel: int
    change: str
    score: float


class OnlineRegimeDetector:
    """One-pass two-sided CUSUM against a baseline fitted during warmup.

    Call update once per observation. Alert indices indicate detection time.
    With reset_on_alarm, a fresh baseline starts at the following observation.
    """

    def __init__(self, *, warmup=40, threshold=8.0, allowance=0.5,
                 monitor_variance=False, min_scale=1e-6, reset_on_alarm=True):
        _positive_integer(warmup, "warmup")
        if warmup < 2:
            raise ValueError("warmup must be at least two observations.")
        if not all(np.isfinite(x) and x > 0 for x in (threshold, allowance, min_scale)):
            raise ValueError("threshold, allowance and min_scale must be finite and positive.")
        self.warmup = warmup
        self.threshold = threshold
        self.allowance = allowance
        self.monitor_variance = monitor_variance
        self.min_scale = min_scale
        self.reset_on_alarm = reset_on_alarm
        self.n_seen = self.regime = 0
        self._dimension = None
        self._baseline = []

    def update(self, observation):
        row = np.atleast_1d(np.asarray(observation, dtype=float))
        if row.ndim != 1 or row.size == 0 or not np.isfinite(row).all():
            raise ValueError("Provide a finite scalar or vector observation.")
        if self._dimension is None:
            self._dimension = len(row)
        if len(row) != self._dimension:
            raise ValueError("The number of channels must stay constant.")
        index = self.n_seen
        self.n_seen += 1
        if len(self._baseline) < self.warmup:
            self._baseline.append(row.copy())
            if len(self._baseline) == self.warmup:
                self._mean = np.mean(self._baseline, axis=0)
                self._scale = np.std(self._baseline, axis=0, ddof=1).clip(self.min_scale)
                shape = (2 if self.monitor_variance else 1, self._dimension)
                self._positive = np.zeros(shape)
                self._negative = np.zeros(shape)
            return None

        z = (row - self._mean) / self._scale
        statistics = np.stack((z, (z**2 - 1) / np.sqrt(2))) if self.monitor_variance else z[None, :]
        self._positive = np.maximum(0, self._positive + statistics - self.allowance)
        self._negative = np.maximum(0, self._negative - statistics - self.allowance)
        scores = np.stack((self._positive, self._negative))
        direction, kind, channel = np.unravel_index(np.argmax(scores), scores.shape)
        score = float(scores[direction, kind, channel])
        if score < self.threshold:
            return None
        self.regime += 1
        change = ("mean" if kind == 0 else "variance") + ("_up" if direction == 0 else "_down")
        if self.reset_on_alarm:
            self._baseline.clear()
        else:
            self._positive.fill(0)
            self._negative.fill(0)
        return RegimeAlert(index, self.regime, int(channel), change, score)

import numpy as np

from LocalStationarity import OnlineRegimeDetector, LocalVAR, regime_breaks, rolling_backtest, rolling_spectrum


def test_live_cusum_alerts_without_future_observations():
    rng = np.random.default_rng(9)
    baseline = rng.normal(size=(40, 2))
    updates = np.column_stack((np.zeros(4), np.full(4, 3.0)))
    detector = OnlineRegimeDetector(warmup=40, threshold=7, allowance=0.5)
    assert all(detector.update(row) is None for row in baseline)
    alerts = [alert for row in updates if (alert := detector.update(row)) is not None]
    assert len(alerts) == 1
    (alert,) = alerts
    assert 40 <= alert.index < 44 and alert.channel == 1
    assert alert.regime == 1 and alert.change == "mean_up"


def test_local_spectrum_forecast_and_retrospective_break():
    series = np.sin(2 * np.pi * np.arange(256) / 8)
    for method in ("periodogram", "welch"):
        spectrum = rolling_spectrum(series, 64, step=16, sample_rate=8, method=method)
        assert spectrum.power.shape == (13, 33, 1)
        assert spectrum.frequencies[np.argmax(spectrum.power[-1, :, 0])] == 1
    model = LocalVAR(48, 2)
    origins, forecasts, actuals = rolling_backtest(series, model, horizon=4, step=8)
    assert origins[0] == 48
    np.testing.assert_allclose(forecasts, actuals, atol=1e-12)
    np.testing.assert_array_equal(regime_breaks(np.r_[np.zeros(60), np.ones(60)]), [60])

import numpy as np

from feature_scaling import compare_scaling_methods, standardize, minmax_scale, robust_scale


def test_standardize_values():
    values = np.array([1.0, 2.0, 3.0, 4.0])
    result = standardize(values)
    assert np.isclose(result.mean(), 0.0)
    assert np.isclose(result.std(), 1.0)


def test_minmax_scale_values():
    values = np.array([1.0, 2.0, 3.0, 4.0])
    result = minmax_scale(values)
    assert np.allclose(result, np.array([0.0, 0.3333333333, 0.6666666667, 1.0]))


def test_robust_scale_values():
    values = np.array([1.0, 2.0, 3.0, 100.0])
    result = robust_scale(values)
    assert np.allclose(result, np.array([-1.5, -0.5, 0.5, 97.5]))


def test_compare_scaling_methods_returns_summary():
    values = np.array([1.0, 2.0, 3.0, 4.0])
    summary = compare_scaling_methods(values)

    assert summary["input_min"] == 1.0
    assert summary["input_max"] == 4.0
    assert summary["standardized_mean"] == 0.0
    assert summary["standardized_std"] == 1.0
    assert summary["minmax_range"] == (0.0, 1.0)
    assert summary["robust_median"] == 0.0

import numpy as np


def standardize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    mean = values.mean()
    std = values.std()
    if std == 0:
        return np.zeros_like(values, dtype=float)
    return (values - mean) / std


def minmax_scale(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    min_val = values.min()
    max_val = values.max()
    if max_val == min_val:
        return np.zeros_like(values, dtype=float)
    return (values - min_val) / (max_val - min_val)


def robust_scale(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    median = np.median(values)
    mad = np.median(np.abs(values - median))
    if mad == 0:
        return np.zeros_like(values, dtype=float)
    return (values - median) / mad


def compare_scaling_methods(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=float)
    standardized = standardize(values)
    minmax = minmax_scale(values)
    robust = robust_scale(values)

    return {
        "input_min": float(values.min()),
        "input_max": float(values.max()),
        "standardized_mean": float(standardized.mean()),
        "standardized_std": float(standardized.std()),
        "minmax_range": (float(minmax.min()), float(minmax.max())),
        "robust_median": float(np.median(robust)),
        "results": {
            "standardized": standardized,
            "minmax": minmax,
            "robust": robust,
        },
    }


if __name__ == "__main__":
    sample = np.array([1.0, 2.0, 3.0, 4.0])
    summary = compare_scaling_methods(sample)
    print("Feature scaling comparison")
    print(summary)

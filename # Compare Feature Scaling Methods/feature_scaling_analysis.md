# Feature Scaling Analysis

## What the script does

Loads the data, identifies the 13 numeric features, applies all three scalers to the same columns, plots before/after distributions for three deliberately-chosen features, prints outlier/skew diagnostics, and outputs a data-grounded recommendation. The three showcase features each stress a different scaler: alcohol (near-symmetric, 0% outliers), proline (huge 278–1680 range), and color_intensity (skewed, 2.2% outliers).

Key thing the plots demonstrate: scaling never changes a feature's shape — only its axis. The histograms are identical bin-for-bin; what moves is the range and centering. The overlay panel makes this vivid: MinMaxScaler crushes everything into [0,1] while the other two spread it out.

## When each scaler is most appropriate

### StandardScaler

Subtracts the mean, divides by standard deviation (→ mean 0, std 1). Best when features are roughly Gaussian/symmetric, because mean and std are only stable summaries when the distribution isn't dominated by a heavy tail. It's the expected input for PCA, SVM, kNN, and regularized linear/logistic regression. Weakness: a few large outliers distort both the mean and the std, dragging the whole feature.

### MinMaxScaler

Linearly maps each feature to a fixed [0,1] range. Best when you need bounded inputs (neural nets, image pipelines) or the feature has a known natural min/max. It preserves the exact shape and zero-sparsity. Weakness: it's the most outlier-sensitive of the three — one extreme value defines the max and compresses every normal point into a thin sliver near zero.

### RobustScaler

Subtracts the median, divides by the IQR (25th–75th percentile). Best when features carry outliers or meaningful skew, because the median and IQR ignore the tails entirely, so extreme points don't warp the transform. Weakness: the output isn't bounded or unit-variance, so it's slightly less convenient for models that assume standardized inputs.

## Recommendation for this dataset

Use StandardScaler as the default for the Wine dataset. The diagnostics support this: no feature exceeds ~2.2% IQR outliers and the mean absolute skewness across all 13 features is only 0.43, so mean and std are stable, trustworthy summaries here — the exact condition StandardScaler assumes. It also matches the models this dataset is typically used with (PCA, SVM, kNN, logistic regression), which expect zero-mean/unit-variance inputs. Note that the real problem all three scalers solve first is scale disparity: unscaled, proline (278–1680) would swamp features like hue (0.48–1.71) in any distance or variance computation.

Switch to RobustScaler if you specifically want to shield the few skewed/outlier-bearing columns (magnesium, color_intensity, malic_acid), and to MinMaxScaler only if a downstream step needs a strict [0,1] bound.

One note for your task comment: your prompt mentioned Boston Housing — that dataset was removed from sklearn (v1.2+) over ethical concerns with one of its features, so I used Wine instead. If your assignment specifically requires a housing dataset, California Housing (fetch_california_housing) is the sanctioned replacement and would actually flip the recommendation to RobustScaler, since it has genuinely heavy outliers. Happy to regenerate against that if you have network access where you're running it.

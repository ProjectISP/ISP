from isp.cython_code.hampel import hampel
import numpy as np
data = np.random.randn(10*300).astype(np.float32)
filtered, outliers, medians, mads, thresholds = hampel(data, window_size=10, n_sigma=3)
print("Outliers:", filtered)
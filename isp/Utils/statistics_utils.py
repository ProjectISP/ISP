import numpy as np
from scipy.optimize import curve_fit


class GutenbergRichterLawFitter:
    def __init__(self, magnitudes):
        self.magnitudes = magnitudes
        self.params = self.fit_gutenberg_richter_law()

    @staticmethod
    def gutenberg_richter_law(magnitudes, a, b):
        return a - b * magnitudes

    def fit_gutenberg_richter_law(self, bins=20):
        # Fit Gutenberg-Richter law
        counts, bin_edges = np.histogram(self.magnitudes, bins=bins, density=False)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # Find the index of the zero
        if np.any(counts == 0):
            zero_index = np.where(counts == 0)
            # Assign 1 at the position of the zero
            counts[zero_index] = 1

        params, _ = curve_fit(self.gutenberg_richter_law, bin_centers, np.log10(counts))
        return params, bin_centers, counts
import numpy as np

def calc(peak_locs, num_pts):
    fraction_peak_window = 0.6
    full_peak_window = max(peak_locs) - min(peak_locs)
    pad_width = ((1 - fraction_peak_window) / (2 * fraction_peak_window)) * full_peak_window
    MHz_full_window = (0.5 * full_peak_window + pad_width) + (0.5 * full_peak_window + pad_width)
    points_per_MHzs = num_pts / MHz_full_window
    MHz_per_points = 1 / points_per_MHzs
    return MHz_per_points

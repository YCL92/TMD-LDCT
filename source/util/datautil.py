import numpy as np


# calculate angle between two vectors
def calcAngle(x1, x2, y1, y2):
    angle = np.arcsin((x1 * y2 - x2 * y1) / (np.sqrt(x1**2 + x2**2) * np.sqrt(y1**2 + y2**2)) + 1e-8)

    return angle


# calculate back-projection weight
def calcWeight(x, q=0.6):
    x_abs = np.abs(x)
    out = np.cos((np.pi / 2) * (x_abs - q) / (1 - q)) ** 2
    out[x_abs < q] = 1
    out[x_abs >= 1] = 0

    return out


# find the nearest point of the given value
def findNearest(beta_lookup, beta_query):
    query_idx = np.zeros_like(beta_query)
    query_offset = np.zeros_like(beta_query)

    for idx in range(len(beta_lookup) - 1):
        set_flags = beta_query > beta_lookup[idx]
        query_idx[set_flags] = idx
        query_offset[set_flags] = (beta_query[set_flags] - beta_lookup[idx]) / (beta_lookup[idx + 1] - beta_lookup[idx])
    query_idx = query_idx + query_offset
    query_idx = np.clip(query_idx, 0, len(beta_lookup) - 1)

    return query_idx

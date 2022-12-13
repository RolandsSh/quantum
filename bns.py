"""Functions relates to Bars & Stripes (BNS) problem analysis."""

from itertools import product

import numpy as np

def generate(num_dimensions=10):
    """Generate all positive samples of size num_dimensions."""
    samples = []
    for pattern in product([0,1], repeat=num_dimensions):
        pattern = np.array(pattern, dtype=np.int0)
        samples.append(np.tile(pattern, num_dimensions))
        samples.append(np.repeat(pattern, num_dimensions))
    samples.pop(0) # remove one of two all-zero sample
    samples.pop(-1) # remove one of two all-one sample
    samples = np.array(samples)
    np.random.shuffle(samples)
    return samples

def get_foms(generated_set, positive_samples):
    """Calculate all figures of merit."""
    f = frequencies(generated_set, positive_samples)
    fsum = f.sum()
    p = fsum / len(generated_set)
    r = (f > 0).sum() / len(positive_samples)
    dist = f / fsum if fsum > 0 else f
    l2 = np.sqrt(((dist - 1 / positive_samples.shape[0]) ** 2).sum())
    mad = mean_abs_distance(generated_set)
    return p, r, l2, mad

def precision(generated_set, positive_samples):
    """Calculate precision of generated set."""
    s = 0
    for row in generated_set:
        if (~np.logical_xor(row, positive_samples)).all(axis=1).any():
            s += 1
    return s / len(generated_set)

def recall(generated_set, positive_samples):
    """Calculate recall of generated set."""
    s = 0
    for row in positive_samples:
        if (~np.logical_xor(row, generated_set)).all(axis=1).any():
            s += 1
    return s / len(positive_samples)

def frequencies(generated_set, positive_samples):
    """Calculate the frequencies of the positive samples in the generated set."""
    n = int(np.log2(positive_samples.shape[0] + 2) - 1)
    counts = np.zeros(2 * 2 ** n, dtype=int)

    def _stripes(box):
        return ((box.sum(axis=0) == 0) | (box.sum(axis=0) == n)).all()

    def _index(box):
        ones = (box.sum(axis=0) == n)
        s = 0
        for i, one in enumerate(ones):
            s += 2 ** i if one else 0
        return s

    for row in generated_set:
        box = row.reshape(n, n)
        if _stripes(box):
            counts[_index(box)] += 1
        elif _stripes(box.T):
            counts[2 ** n + _index(box.T)] += 1
    return counts

def distribution(generated_set, positive_samples):
    """Calculate the distribution of the positive samples in the generated set."""
    counts = frequencies(generated_set, positive_samples)
    if counts.sum() > 0:
        return counts / counts.sum()
    else:
        return counts

def distribution_distance(generated_set, positive_samples, order=2):
    """
    Calculate the distance between the distribution of the generated set and the
    distribution of the positive samples.
    """
    dist1 = distribution(generated_set, positive_samples)
    dist2 = distribution(positive_samples, positive_samples)
    return np.linalg.norm(dist1 - dist2, ord=order)

def distance(image):
    """
    Calculate the distance from the given image to the nearest positive image.
    """
    n = int(np.sqrt(image.shape[0]))
    image = image.reshape(n, n)
    bars = image.sum(axis=0)
    stripes = image.sum(axis=1)
    dist_bars = np.min((bars, n - bars), axis=0).sum()
    dist_stripes = np.min((stripes, n - stripes), axis=0).sum()
    return min(dist_bars, dist_stripes)

def mean_abs_distance(generated_set):
    """
    Calculate the mean absolute distance of the generated set to the manifold.
    """
    distances = np.apply_along_axis(distance, 1, generated_set)
    return distances.mean()

def mean_squared_distance(generated_set):
    """
    Calculate the mean squared distance of the generated set to the manifold.
    """
    distances = np.apply_along_axis(distance, 1, generated_set)
    return (distances ** 2).mean()

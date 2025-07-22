"""
Helper for multiprocessing – lives in a real .py file so it can be
imported by spawned subprocesses (fixes the “Can't pickle local object”
error that occurs when the worker function is defined inside a notebook).
"""

from persistent_homology import (
    BettiZero,
    compute_intervals,
    compute_largest_bar,
)

def process_direction(direction, vertices, edges):
    """
    Run β₀ persistence for ONE direction (executed in a worker process).
    Returns a dict with the same structure you were using before.
    """
    bz = BettiZero(direction, vertices, edges)
    comps, mergers, verts, births = bz.compute_persistence()
    intervals = compute_intervals(births, mergers)
    length, bar = compute_largest_bar(intervals)
    return {
        "direction":        direction,
        "intervals":        intervals,
        "largest_bar":      bar,
        "largest_length":   length,
        "components":       list(comps),
    }
# --- Optimised β₀ persistence workflow ---------------------------------
# Rewrites the original triple‑nested loop to:
#   • Re‑use vertices/edges per segmentation (loaded once)
#   • Distribute direction work in parallel across CPU cores
#   • Collect JSON in‑memory and write once at the end

import json, pathlib, pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed

from persistent_homology import (
    BettiZero,
    compute_intervals,
    compute_largest_bar,
    generate_sphere_points,
)

root = pathlib.Path("./lung_segmentations")

def load_vertices_edges(seg_folder: pathlib.Path):
    """Read vertices/edges only once per segmentation."""
    verts = pd.read_csv(seg_folder / "vertices.csv").values.tolist()
    edges = pd.read_csv(seg_folder / "edges.csv").values.tolist()
    return verts, edges

def process_direction(args):
    """Run β₀ persistence for one direction (runs in worker)."""
    direction, vertices, edges = args
    bz = BettiZero(direction, vertices, edges)
    comps, mergers, verts, births = bz.compute_persistence()
    intervals = compute_intervals(births, mergers)
    length, bar = compute_largest_bar(intervals)
    return {
        "direction": direction,
        "intervals": intervals,
        "largest_bar": bar,
        "largest_length": length,
        "components": list(comps),
    }

json_data = []

for seg_folder in root.iterdir():
    if not seg_folder.is_dir():
        continue
    vertices, edges = load_vertices_edges(seg_folder)
    directions = generate_sphere_points(20, 5, 1e-7)

    # Parallel processing over directions
    with ProcessPoolExecutor() as ex:
        future_map = {
            ex.submit(process_direction, (d, vertices, edges)): d
            for d in directions
        }
        seg_results = [f.result() for f in as_completed(future_map)]

    json_data.append({"segmentation": seg_folder.name, "results": seg_results})
    print(f"✓ Processed {seg_folder.name}")

# Single JSON write at the end
with open("BettiZeroSegmentations.json", "w") as fp:
    json.dump(json_data, fp, indent=2)

print("✅ All segmentations done → BettiZeroSegmentations.json")
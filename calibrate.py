"""
Suggest new anchor values that put the current smokiest frames at ~target_score.
"""
import pandas as pd
import numpy as np

CSV = "smoke_scores.csv"
TARGET = 0.65   # where current top frames should land on the [0,1] feature scale
TOP_N = 50      # use the top N frames as the "smokiest in current data" reference

# current anchors (from analyze_smoke defaults)
SAT_CLEAR, SAT_SMOKY = 120.0, 20.0
EDGE_CLEAR, EDGE_SMOKY = 0.10, 0.01
DARK_CLEAR, DARK_SMOKY = 20.0, 150.0

df = pd.read_csv(CSV)
df = df[df["status"] == "ok"]
top = df.nlargest(TOP_N, "smoke_score")

# pick the "extreme but typical" raw value for each feature among smokiest frames
sat_extreme  = top["saturation_mean"].quantile(0.10)   # lower = smokier
edge_extreme = top["edge_density"].quantile(0.10)      # lower = smokier
dark_extreme = top["dark_channel_mean"].quantile(0.90) # higher = smokier

# solve _normalize(extreme, clear, smoky_new) = TARGET for smoky_new
def solve_smoky(extreme, clear, target):
    return clear + (extreme - clear) / target

new_sat_smoky  = solve_smoky(sat_extreme,  SAT_CLEAR,  TARGET)
new_edge_smoky = solve_smoky(edge_extreme, EDGE_CLEAR, TARGET)
new_dark_smoky = solve_smoky(dark_extreme, DARK_CLEAR, TARGET)

print(f"Top {TOP_N} smokiest frames — extreme raw values:")
print(f"  saturation_mean   ~ {sat_extreme:.2f}  (current anchor: {SAT_SMOKY})")
print(f"  edge_density      ~ {edge_extreme:.4f} (current anchor: {EDGE_SMOKY})")
print(f"  dark_channel_mean ~ {dark_extreme:.2f}  (current anchor: {DARK_SMOKY})")
print()
print(f"Suggested new anchors so these land at ~{int(TARGET*100)}% on the feature scale:")
print(f"  saturation_smoky = {new_sat_smoky:.1f}")
print(f"  edge_smoky       = {new_edge_smoky:.4f}")
print(f"  dark_smoky       = {new_dark_smoky:.1f}")
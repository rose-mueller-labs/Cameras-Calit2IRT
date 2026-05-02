import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import cv2
from scipy.stats import ttest_ind, linregress
from statsmodels.nonparametric.smoothers_lowess import lowess

def parse_coord(s):
    try:
        s = str(s).strip().strip("()")
        x, y = s.split("!")
        return float(x), float(y)
    except Exception:
        return np.nan, np.nan

def pop_color(name):
    if name.startswith("ACO"):
        return "deeppink"
    if name.startswith("CO"):
        return "steelblue"

test_names = ['ACO1.mov', 'ACO2.mov', 'ACO3.mov', 'ACO4.mov', 'ACO5.mov',
              # 'CACO4.mov', 'CACO5.mov', 
              'CO1.mov', 'CO2.mov', 'CO3.mov', 'CO4.mov', 'CO5.mov',
              # 'CAO4.mov', 'CAO5.mov'
              ]

aco_totals_cm = {} # dict for per-enclosure totals
# ca_totals_cm = {}
co_totals_cm = {}

avg_speed_per_frame = dict()
speed_info_per_vid = dict()

for vid_name in test_names:
    csv_name = f"./AldenAlg/Tracked_{vid_name}_pwsBacklit.csv"
    df = pd.read_csv(csv_name)
    BASE_PATH = "/Volumes/Crucial X9/Downloads/UROP Data Colletion 4-26-2026"

    df = df.iloc[4:].reset_index(drop=True)
    cap = cv2.VideoCapture(f"{BASE_PATH}/{vid_name}")
    if vid_name[0] == 'A':
        fps = 120
    else:
        fps = 60
    stopaatfiv = fps * 300
    start_frame = df["frame"].iloc[0]
    df = df[df["frame"] < start_frame + stopaatfiv].reset_index(drop=True)

    id_cols = [c for c in df.columns if c.startswith("ID_")]

    coords = {}
    for col in id_cols:
        xy = df[col].apply(parse_coord).tolist()
        coords[col] = pd.DataFrame(xy, columns=["x", "y"], index=df["frame"].values)

    for col in id_cols:
        coords[col] = coords[col].interpolate(method="linear", limit_direction="both")

    results = {}
    all_frame_dists = []

    for col in id_cols:
        xy = coords[col]
        dx = xy["x"].diff()
        dy = xy["y"].diff()
        dist_per_frame = np.sqrt(dx**2 + dy**2).fillna(0)
        all_frame_dists.append(dist_per_frame)

        total_dist_cm = dist_per_frame.sum() * (10 / 1730)
        results[col] = {"total_distance_px": round(dist_per_frame.sum(), 1), "total_distance_cm": round(total_dist_cm, 2)}

        # adding per-enclosure instead of appending per-fly by 20
        if vid_name.startswith("ACO"):
            aco_totals_cm[vid_name] = aco_totals_cm.get(vid_name, 0) + total_dist_cm
        elif vid_name.startswith("CO"):
            co_totals_cm[vid_name] = co_totals_cm.get(vid_name, 0) + total_dist_cm

    dist_matrix = pd.concat(all_frame_dists, axis=1).replace(0, np.nan)
    frame_avg_speed = (dist_matrix.sum(axis=1) / 20) * (10 / 1730) * fps  # cm/s

    for frame_idx, speed in zip(df["frame"].values, frame_avg_speed):
        avg_speed_per_frame[frame_idx] = speed

    seconds = [(f - df["frame"].iloc[0]) / fps for f in df["frame"].values]
    speed_info_per_vid[vid_name] = (seconds, list(frame_avg_speed))

    # trajectories
    colors = plt.cm.tab10.colors
    for i, col in enumerate(id_cols):
        xy = coords[col]
        plt.plot(xy["x"], xy["y"], color=colors[i % 10], lw=1, label=col)
    plt.gca().invert_yaxis()
    plt.title(f"{vid_name.split('.mov')[0]} Trajectories")
    plt.xlabel("X (px)")
    plt.ylabel("Y (px)")
    plt.savefig(f'./WatershedAlgorithm/UROPPlots/{vid_name}_trajectory.png', dpi=150)
    plt.close()
plt.close()
# raw speed
vid_names = list(speed_info_per_vid.keys())
n = len(vid_names)
ncols = 3
nrows = (n + ncols - 1) // ncols

# binned speeds (easier ngl)
BIN_SIZE = 5

fig2, axes2 = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 3), sharey=True)
axes2 = np.array(axes2).flatten()
fig2.suptitle("Avg Speed Over Time Per Enclosure (5 sec. bins)")

for ax, name in zip(axes2, vid_names):
    seconds, speeds = speed_info_per_vid[name]
    s_arr  = np.array(seconds, dtype=float)
    sp_arr = np.array(speeds,  dtype=float)

    bin_edges   = np.arange(0, s_arr.max() + BIN_SIZE, BIN_SIZE)
    bin_centers = bin_edges[:-1] + BIN_SIZE / 2
    bin_means, bin_sems = [], []

    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        vals = sp_arr[(s_arr >= lo) & (s_arr < hi)]
        vals = vals[~np.isnan(vals)]
        if len(vals):
            bin_means.append(np.mean(vals))
            bin_sems.append(np.std(vals) / np.sqrt(len(vals)))
        else:
            bin_means.append(np.nan)
            bin_sems.append(np.nan)

    bin_means = np.array(bin_means)
    bin_sems  = np.array(bin_sems)

    ax.bar(bin_centers, bin_means, width=BIN_SIZE * 0.85, color=pop_color(name), alpha=0.65)
    ax.errorbar(bin_centers, bin_means, yerr=bin_sems, fmt="none", color="black", capsize=2, linewidth=0.8)
    ax.set_title(name.replace(".mov", ""))
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Speed (cm/s)")

for ax in axes2[n:]:
    ax.set_visible(False)

fig2.tight_layout()
fig2.savefig("./WatershedAlgorithm/UROPPlots/speed_over_time_BINNED10.png", dpi=150, bbox_inches="tight")
# plt.show()
plt.close()


# t test 
populations = {"ACO": list(aco_totals_cm.values()), "CO":  list(co_totals_cm.values())}

t_aco_co, p_aco_co = ttest_ind(populations["ACO"], populations["CO"], equal_var=False)

print(f"p_aco_co {p_aco_co}")


# final overall 
colors = {"ACO": "deeppink", "CO": "steelblue"}

fig_box, ax_box = plt.subplots(figsize=(3.5, 4)) # skinny fig
pop_names = list(populations.keys())
data = [populations[p] for p in pop_names]
bp = ax_box.boxplot(data, patch_artist=True, widths=0.45)

for patch, pop in zip(bp['boxes'], pop_names):
    patch.set_facecolor(colors[pop])
    patch.set_alpha(0.7)

for element in ['whiskers', 'caps', 'medians', 'fliers']:
    for item in bp[element]:
        item.set_color('black')

ax_box.set_xticks(range(1, len(pop_names) + 1))
ax_box.set_xticklabels(pop_names)

y_max = max(max(v) for v in populations.values())
for step, (i, j, p) in enumerate([(0, 1, p_aco_co)]):
    xi, xj = i + 1, j + 1
    y = y_max * (1.08 + 0.07 * step)
    ax_box.plot([xi, xi, xj, xj], [y, y * 1.01, y * 1.01, y], color='black', lw=1)
    ax_box.text((xi + xj) / 2, y * 1.015, '*' if p < 0.001 else '', ha='center', va='bottom', fontsize=10)

ax_box.set_ylabel("Total Distance (cm)")
ax_box.set_title("Total Distance Distribution")
ax_box.text(0.03, 0.5, f"p = {p_aco_co:.3g}", transform=ax_box.transAxes, ha='left', va='top') # yo fix this

fig_box.tight_layout()
fig_box.savefig("./WatershedAlgorithm/UROPPlots/pop_move_OVERALL.png", dpi=150)
plt.close()

# binned speed avg with smoothed n lin fit
bin_edges   = np.arange(0, 300 + BIN_SIZE, BIN_SIZE)
bin_centers = bin_edges[:-1] + BIN_SIZE / 2

pop_groups = {"ACO": [], "CO": []}
for vid_name, (seconds, speeds) in speed_info_per_vid.items():
    key = "ACO" if vid_name.startswith("ACO") else "CO"
    s_arr  = np.array(seconds, dtype=float)
    sp_arr = np.array(speeds,  dtype=float)

    vid_bin_means = []
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        vals = sp_arr[(s_arr >= lo) & (s_arr < hi)]
        vals = vals[~np.isnan(vals)]
        vid_bin_means.append(np.nanmean(vals) if len(vals) else np.nan)
    pop_groups[key].append(vid_bin_means)

fig4, (ax_aco, ax_co) = plt.subplots(2, 1, figsize=(8, 12), sharey=True)
fig4.suptitle("Average Activity Over Time", fontsize=24)  # keep default size

axes_map = {"ACO": ax_aco, "CO": ax_co}

pop_styles = {"ACO": {"color": "deeppink",  "label": "ACO"}, "CO":  {"color": "steelblue", "label": "CO"}}

for pop, style in pop_styles.items():
    ax = axes_map[pop]
    mat = np.array(pop_groups[pop])
    means = np.nanmean(mat, axis=0)
    sems  = np.nanstd(mat, axis=0) / np.sqrt((~np.isnan(mat)).sum(axis=0))
    c = style["color"]

    ax.bar(bin_centers, means, width=BIN_SIZE * 0.4, color=c, alpha=0.5, label=f"{style['label']} (binned mean)")

    ax.errorbar(bin_centers, means, yerr=sems, fmt="none", color=c, capsize=2, linewidth=0.8, alpha=0.5)

    valid = ~np.isnan(means)
    x_v, y_v = bin_centers[valid], means[valid]

    smoothed = lowess(y_v, x_v, frac=0.3, return_sorted=True)
    ax.plot(smoothed[:, 0], smoothed[:, 1], color=c, lw=2, label=f"{style['label']} smoothed")

    slope, intercept, r, p_val, *_ = linregress(x_v, y_v)
    print(pop, p_val)

    y_fit = slope * x_v + intercept
    ax.plot(x_v, y_fit, color=c, lw=1, ls=":", label=f"{style['label']} fit (r={round(r, 2)}, p={round(p_val, 4)})")

    ax.legend(fontsize=20)  
    ax.set_xlabel("Time (s)", fontsize=24)
    ax.set_ylabel("Avg Speed (cm/s)", fontsize=24)
    ax.set_title(style["label"], fontsize=24)
    ax.tick_params(axis='both', labelsize=14)

fig4.tight_layout()
fig4.savefig("./WatershedAlgorithm/UROPPlots/pop_speed_averaged_fit.png", dpi=150, bbox_inches="tight")
plt.close()
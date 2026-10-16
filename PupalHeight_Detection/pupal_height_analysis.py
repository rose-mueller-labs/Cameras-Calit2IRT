import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind, linregress, t

EVAL_SUMMARY = "/Volumes/Crucial X9/Cameras-Calit2IRT/PupalHeight_Detection/TestingEval/eval_summary.csv"
ALGORITHM = None # set to a value from eval_summary's `algorithm` column to filter; None = use all rows
MIN_ID_LENGTH_SEC = 0 # skip videos whose IDs switch in under this many seconds, used to be 40 but I made it 0 now to see what happenes

# CSV path per video
TRACK_CSV = "/Volumes/Crucial X9/Cameras-Calit2IRT/PupalHeight_Detection/TestingOutput/{vid}_orb_watershed_velocity_hungarian_pathing.csv"
OUT_DIR = "/Volumes/Crucial X9/Cameras-Calit2IRT/PupalHeight_Detection/AnalysisOutput"

FPS = 119
N_FLIES = 20
CM_PER_PX = 10/1730 # idk this is based on calit2
DURATION_SEC = 300
BIN_SIZE = 5

PAIRS = [("f", "m"), ("fb", "mb"), ("fm", "mm"), ("ft", "mt")]
COLOR_A, COLOR_B = "deeppink", "steelblue"

def parse_coord(s):
    try:
        x, y = str(s).strip().strip("()").split("!")
        return float(x), float(y)
    except Exception:
        return np.nan, np.nan


def in_group(vid_name, prefix):
    """True if the video name starts with the prefix, e.g. 'fb3.mov' is in 'f' and in 'fb'."""
    return vid_name.lower().startswith(prefix)


def reg_ci(x, y, xgrid, alpha=0.05):
    x, y = np.asarray(x, float), np.asarray(y, float)
    n = len(x)
    slope, intercept, r, p, se = linregress(x, y)
    s_err = np.sqrt(np.sum((y - (slope * x + intercept)) ** 2) / (n - 2))
    sxx = np.sum((x - x.mean()) ** 2)
    ygrid = slope * xgrid + intercept
    ci = t.ppf(1 - alpha / 2, df=n - 2) * s_err * np.sqrt(1 / n + (xgrid - x.mean()) ** 2 / sxx)
    return ygrid, ygrid - ci, ygrid + ci, f"{p:.3g}", f"{slope:.3g}", f"{intercept:.3g}"


def stars(p):
    if p < 0.001:
        return "****"  
    elif p < 0.01:
        return "***"
    elif p < 0.05:
        return "**"
    elif p < 0.10:
       return "*" 
    else:
        return "ns"


# pick videos from eval_summary
# columns: video,algorithm,frames_processed,unique_ids,avg_id_length_frames,avg_id_length_sec,avg_flies_per_frame
eval_df = pd.read_csv(EVAL_SUMMARY)
if ALGORITHM is not None:
    eval_df = eval_df[eval_df["algorithm"] == ALGORITHM]

# skip any video where IDs are switched in under 40 seconds
bad = eval_df[eval_df["avg_id_length_sec"] < MIN_ID_LENGTH_SEC]["video"].unique()
print(f"Skipping {len(bad)} video(s) with avg ID length < {MIN_ID_LENGTH_SEC}s: {list(bad)}")

wanted_groups = {g for pair in PAIRS for g in pair}
os.makedirs(OUT_DIR, exist_ok=True)
test_names = []
for v in eval_df["video"].unique():
    if v in bad:
        continue
    name = v if v.endswith(".mov") else f"{v}.mov"
    if any(in_group(name, g) for g in wanted_groups):
        test_names.append(name)

# per-video processing 
totals_cm = {} # vid -> total distance of all flies (cm)
speed_info_per_vid = {} # vid -> (seconds, avg speed per frame in cm/s)

for vid_name in test_names:
    stem = vid_name.replace(".mov", "")
    vid_key = vid_name if os.path.exists(TRACK_CSV.format(vid=vid_name)) else stem
    df = pd.read_csv(TRACK_CSV.format(vid=vid_key))
    df = df.iloc[4:].reset_index(drop=True)

    start_frame = df["frame"].iloc[0]
    df = df[df["frame"] < start_frame + FPS * DURATION_SEC].reset_index(drop=True)

    id_cols = [c for c in df.columns if c.startswith("ID_")]
    coords = {}
    for col in id_cols:
        xy = df[col].apply(parse_coord).tolist()
        coords[col] = pd.DataFrame(xy, columns=["x", "y"], index=df["frame"].values)
        coords[col] = coords[col].interpolate(method="linear", limit_direction="both")

    frame_dists = []
    total_cm = 0
    for col in id_cols:
        xy = coords[col]
        d = np.sqrt(xy["x"].diff() ** 2 + xy["y"].diff() ** 2).fillna(0)
        frame_dists.append(d)
        total_cm += d.sum() * CM_PER_PX
    totals_cm[vid_name] = total_cm

    dist_matrix = pd.concat(frame_dists, axis=1).replace(0, np.nan)
    frame_avg_speed = (dist_matrix.sum(axis=1) / N_FLIES) * CM_PER_PX * FPS  # cm/s
    seconds = [(f - df["frame"].iloc[0]) / FPS for f in df["frame"].values]
    speed_info_per_vid[vid_name] = (np.array(seconds, float), frame_avg_speed.to_numpy(float))

    # trajectories
    colors = plt.cm.tab10.colors
    for i, col in enumerate(id_cols):
        plt.plot(coords[col]["x"], coords[col]["y"], color=colors[i % 10], lw=1, label=col)
    plt.gca().invert_yaxis()
    plt.title(f"{stem} Trajectories")
    plt.xlabel("X (px)")
    plt.ylabel("Y (px)")
    plt.savefig(f"{OUT_DIR}/{stem}_trajectory.png", dpi=150)
    plt.close()

# binned speed per video
bin_edges = np.arange(0, DURATION_SEC + BIN_SIZE, BIN_SIZE)
bin_centers = bin_edges[:-1] + BIN_SIZE / 2

vid_bin_means = {}
for name, (s_arr, sp_arr) in speed_info_per_vid.items():
    means = []
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        vals = sp_arr[(s_arr >= lo) & (s_arr < hi)]
        vals = vals[~np.isnan(vals)]
        means.append(vals.mean() if len(vals) else np.nan)
    vid_bin_means[name] = np.array(means)

# per-pair comparisons
for ga, gb in PAIRS:
    vids = {ga: [v for v in test_names if in_group(v, ga)],
            gb: [v for v in test_names if in_group(v, gb)]}
    if min(len(vids[ga]), len(vids[gb])) < 2:
        print(f"{ga} vs {gb}: need >=2 videos per group (have {len(vids[ga])}, {len(vids[gb])}), skipping")
        continue

    colors = {ga: COLOR_A, gb: COLOR_B}
    totals = {g: [totals_cm[v] for v in vids[g]] for g in (ga, gb)}
    _, p = ttest_ind(totals[ga], totals[gb], equal_var=False)
    print(f"{ga} vs {gb}: p = {p:.4g}")

    # boxplot of total distance
    fig, ax = plt.subplots(figsize=(3.5, 4))
    bp = ax.boxplot([totals[ga], totals[gb]], patch_artist=True, widths=0.45)
    for patch, g in zip(bp["boxes"], (ga, gb)):
        patch.set_facecolor(colors[g])
        patch.set_alpha(0.7)
    for element in ["whiskers", "caps", "medians", "fliers"]:
        for item in bp[element]:
            item.set_color("black")
    ax.set_xticks([1, 2])
    ax.set_xticklabels([ga, gb])
    y = max(max(totals[ga]), max(totals[gb])) * 1.08
    ax.plot([1, 1, 2, 2], [y, y * 1.01, y * 1.01, y], color="black", lw=1)
    ax.text(1.5, y * 1.015, stars(p), ha="center", va="bottom", fontsize=10)
    ax.set_ylabel("Total Distance (cm)")
    ax.set_title(f"{ga} vs {gb}: Total Distance")
    ax.set_xlabel(f"p = {p:.3g}")
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/{ga}_vs_{gb}_boxplot.png", dpi=150)
    plt.close(fig)

    # avg speed over time with linear fits (0-100, 100-200, 200-300 s) + CI
    fig, ax = plt.subplots(figsize=(11, 8))
    fig.suptitle(f"Average Activity Over Time ({ga} vs {gb})", fontsize=20)
    for g in (ga, gb):
        c = colors[g]
        mat = np.array([vid_bin_means[v] for v in vids[g]])
        means = np.nanmean(mat, axis=0)
        valid = ~np.isnan(means)
        x_v, y_v = bin_centers[valid], means[valid]
        ax.scatter(x_v, y_v, color=c, alpha=0.5)

        for k, (lo, hi) in enumerate([(0, 100), (100, 200), (200, DURATION_SEC)]):
            m = (x_v >= lo) & (x_v <= hi)
            if m.sum() < 3:
                continue
            xs, ys = x_v[m], y_v[m]
            xgrid = np.linspace(xs.min(), xs.max(), 200)
            yfit, ci_lo, ci_hi, p_fit, slope, intercept = reg_ci(xs, ys, xgrid)
            sign = "+" if float(intercept) >= 0 else "\u2212"
            eq = f"y = {slope}x {sign} {abs(float(intercept)):.3g}"
            ax.plot(xgrid, yfit, color=c, lw=2.2, alpha=0.5,
                    label=f"{g}, {lo}-{hi}s: {eq}, p = {p_fit}")
            ax.fill_between(xgrid, ci_lo, ci_hi, color="gray", alpha=0.22)

    ax.legend(fontsize=10, loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0)
    ax.set_xlabel("Time (s)", fontsize=18)
    ax.set_ylabel("Avg Speed (cm/s)", fontsize=18)
    ax.tick_params(axis="both", labelsize=14)
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/{ga}_vs_{gb}_speed_fit.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
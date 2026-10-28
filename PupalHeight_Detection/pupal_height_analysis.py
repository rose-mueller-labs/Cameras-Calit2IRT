import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind, linregress, t
from itertools import combinations

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

# speed_fit graphs for t, m, and b are put next to each other on a common scale,
# with a line at the 60s mark on the x axis for baseline
COMPARE_PAIRS = [("ft", "mt"), ("fm", "mm"), ("fb", "mb")]  # t, m, b

COLOR_A, COLOR_B = "deeppink", "steelblue"

# pooled treatment groups: every t video (ft + mt) vs every m video (fm + mm) vs every b video (fb + mb)
TREATMENT_GROUPS = {"t": ("ft", "mt"), "m": ("fm", "mm"), "b": ("fb", "mb")}
TREATMENT_COLORS = {"t": "darkgoldenrod", "m": "maroon", "b": "seagreen"}

SEX_PLOTS = {"female": ("ft", "fm", "fb"), "male": ("mt", "mm", "mb"),}

# color each prefix by its treatment (t, m, b = second letter of the prefix)
PREFIX_COLORS = {pre: TREATMENT_COLORS[pre[1]] for pres in SEX_PLOTS.values() for pre in pres}



def parse_coord(s):
    try:
        x, y = str(s).strip().strip("()").split("!")
        return float(x), float(y)
    except Exception:
        return np.nan, np.nan


def in_group(vid_name, prefix):
    """True if the video name starts with the prefix"""
    return vid_name.lower().startswith(prefix)


def reg_ci(x, y, xgrid, alpha=0.05):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
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


def plot_speed_fit(ax, groups, vids, colors):
    """Scatter of binned -> avg speed + piecewise linear fits (0-100, 100-200, 200-300 s) with CI."""
    for g in groups:
        c = colors[g]
        mat = np.array([vid_bin_means[v] for v in vids[g]])
        means = np.nanmean(mat, axis=0)
        valid = ~np.isnan(means)
        x_v = bin_centers[valid]
        y_v = means[valid]
        ax.scatter(x_v, y_v, color=c, alpha=0.5)

        for k, (lo, hi) in enumerate([(0, 100), (100, 200), (200, DURATION_SEC)]):
            m = (x_v >= lo) & (x_v <= hi)
            if m.sum() < 3:
                continue
            xs = x_v[m]
            ys = y_v[m]
            xgrid = np.linspace(xs.min(), xs.max(), 200)
            yfit, ci_lo, ci_hi, p_fit, slope, intercept = reg_ci(xs, ys, xgrid)

            if float(intercept) >= 0:
                sign = "+"
            else:
                sign = "\u2212"

            eq = f"y = {slope}x {sign} {abs(float(intercept)):.3g}"
            ax.plot(xgrid, yfit, color=c, lw=2.2, alpha=0.5,
                    label=f"{g}, {lo}-{hi}s: {eq}, p = {p_fit}")
            ax.fill_between(xgrid, ci_lo, ci_hi, color="gray", alpha=0.22)


def bin_speed(s_arr, sp_arr, bin_size=1, duration=DURATION_SEC):
    """Mean speed per bin for one video. Returns (bin_centers, means)."""
    edges = np.arange(0, duration + bin_size, bin_size)
    centers = edges[:-1] + bin_size / 2
    means = np.full(len(centers), np.nan)
    idx = np.digitize(s_arr, edges) - 1  # bin index for each frame
    for i in range(len(centers)):
        vals = sp_arr[idx == i]
        vals = vals[~np.isnan(vals)]
        if len(vals) > 0:
            means[i] = vals.mean()
    return centers, means


def plot_speed_individual(ax, groups, vids, colors, bin_size=1, fit=False, show_group_mean=False):
    """One trace per video (mean speed in each `bin_size`-second bin).
    fit=True adds a piecewise linear fit (0-100, 100-200, 200-300 s) for each video.
    show_group_mean=True overlays a thick line of the per-group mean."""
    for g in groups:
        c = colors[g]
        all_means = []
        for k, v in enumerate(vids[g]):
            s_arr, sp_arr = speed_info_per_vid[v]
            x, y = bin_speed(s_arr, sp_arr, bin_size)
            all_means.append(y)
            ax.scatter(x, y, color=c, label=f"{g} videos (n={len(vids[g])})" if k == 0 else None)

            if fit:
                valid = ~np.isnan(y)
                x_v, y_v = x[valid], y[valid]
                for lo, hi in [(0, 100), (100, 200), (200, DURATION_SEC)]:
                    m = (x_v >= lo) & (x_v <= hi)
                    if m.sum() < 3:
                        continue
                    slope, intercept, *_ = linregress(x_v[m], y_v[m])
                    xs = np.array([x_v[m].min(), x_v[m].max()])
                    ax.plot(xs, slope * xs + intercept, color=c, lw=1.2, alpha=0.7)

        if show_group_mean and all_means:
            ax.plot(x, np.nanmean(np.array(all_means), axis=0), color=c, lw=2.5, label=f"{g} mean")

def plot_speed_by_sex(ax, g, vids, bin_size=1, show_mean=False):
    """Per-video traces for treatment group g, colored by sex, 
    Female = first prefix in TREATMENT_GROUPS[g], male = second."""
    f_pre, m_pre = TREATMENT_GROUPS[g]
    sex_info = [("female", f_pre, COLOR_A), ("male", m_pre, COLOR_B)]

    for label, pre, c in sex_info:
        sex_vids = [v for v in vids[g] if in_group(v, pre)]
        all_means = []
        x = None
        for k, v in enumerate(sex_vids):
            s_arr, sp_arr = speed_info_per_vid[v]
            x, y = bin_speed(s_arr, sp_arr, bin_size)
            all_means.append(y)
            ax.scatter(x, y, color=c, label=f"{label} videos (n={len(sex_vids)})" if k == 0 else None)

        if show_mean and all_means:
            ax.plot(x, np.nanmean(np.array(all_means), axis=0), color=c, lw=2.5, label=f"{label} mean")

# pick videos from eval_summary
# columns: video,algorithm,frames_processed,unique_ids,avg_id_length_frames,avg_id_length_sec,avg_flies_per_frame
eval_df = pd.read_csv(EVAL_SUMMARY)
if ALGORITHM is not None:
    eval_df = eval_df[eval_df["algorithm"] == ALGORITHM]

# skip any video where IDs are switched in under 40 seconds
too_short = eval_df["avg_id_length_sec"] < MIN_ID_LENGTH_SEC
bad = eval_df[too_short]["video"].unique()
print(f"Skipping {len(bad)} video(s) with avg ID length < {MIN_ID_LENGTH_SEC}s: {list(bad)}")

wanted_groups = set()
for pair in PAIRS:
    for g in pair:
        wanted_groups.add(g)

os.makedirs(OUT_DIR, exist_ok=True)

test_names = []
for v in eval_df["video"].unique():
    if v in bad:
        continue

    if v.endswith(".mov"):
        name = v
    else:
        name = f"{v}.mov"

    in_any_wanted_group = False
    for g in wanted_groups:
        if in_group(name, g):
            in_any_wanted_group = True
            break

    if in_any_wanted_group:
        test_names.append(name)

# per-video processing
totals_cm = {} # vid -> total distance of all flies (cm)
speed_info_per_vid = {} # vid -> (seconds, avg speed per frame in cm/s)

for vid_name in test_names:
    stem = vid_name.replace(".mov", "")

    if os.path.exists(TRACK_CSV.format(vid=vid_name)):
        vid_key = vid_name
    else:
        vid_key = stem

    df = pd.read_csv(TRACK_CSV.format(vid=vid_key))
    df = df.iloc[4:].reset_index(drop=True)

    start_frame = df["frame"].iloc[0]
    df = df[df["frame"] < start_frame + FPS * DURATION_SEC].reset_index(drop=True)

    id_cols = []
    for c in df.columns:
        if c.startswith("ID_"):
            id_cols.append(c)

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

    first_frame = df["frame"].iloc[0]
    seconds = []
    for f in df["frame"].values:
        seconds.append((f - first_frame) / FPS)
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
        if len(vals) > 0:
            means.append(vals.mean())
        else:
            means.append(np.nan)
    vid_bin_means[name] = np.array(means)

# per-pair comparisons
pair_speed_data = {}  # (ga, gb) -> (vids, colors), kept for the side-by-side figure
for ga, gb in PAIRS:
    vids_a = []
    vids_b = []
    for v in test_names:
        if in_group(v, ga):
            vids_a.append(v)
        if in_group(v, gb):
            vids_b.append(v)
    vids = {ga: vids_a, gb: vids_b}

    if min(len(vids[ga]), len(vids[gb])) < 2:
        print(f"{ga} vs {gb}: need >=2 videos per group (have {len(vids[ga])}, {len(vids[gb])}), skipping")
        continue

    colors = {ga: COLOR_A, gb: COLOR_B}
    pair_speed_data[(ga, gb)] = (vids, colors)

    totals = {}
    for g in (ga, gb):
        totals[g] = []
        for v in vids[g]:
            totals[g].append(totals_cm[v])

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
    ax.set_ylim(1700, 4500)
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/{ga}_vs_{gb}_boxplot.png", dpi=150)
    plt.close(fig)

    # avg speed over time with linear fits (0-100, 100-200, 200-300 s) + CI
    fig, ax = plt.subplots(figsize=(11, 8))
    fig.suptitle(f"Average Activity Over Time ({ga} vs {gb})", fontsize=20)
    ax.axvline(60, c='r', label='Baseline')
    plot_speed_fit(ax, (ga, gb), vids, colors)

    ax.legend(fontsize=10, loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0)
    ax.set_xlabel("Time (s)", fontsize=18)
    ax.set_ylabel("Avg Speed (cm/s)", fontsize=18)
    ax.tick_params(axis="both", labelsize=14)
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/{ga}_vs_{gb}_speed_fit.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # each video's own avg speed per second
    fig, ax = plt.subplots(figsize=(11, 8))
    fig.suptitle(f"Per-Video Activity Over Time ({ga} vs {gb})", fontsize=20)
    ax.axvline(60, c='r', label='Baseline')
    plot_speed_individual(ax, (ga, gb), vids, colors, bin_size=1, fit=False)
    ax.legend(fontsize=10, loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0)
    ax.set_xlabel("Time (s)", fontsize=18)
    ax.set_ylabel("Avg Speed (cm/s)", fontsize=18)
    ax.tick_params(axis="both", labelsize=14)
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/{ga}_vs_{gb}_speed_individual.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

# side-by-side speed_fit for t, m, b on a common x/y scale, with a line at 60 s
available = []
missing = []
for pr in COMPARE_PAIRS:
    if pr in pair_speed_data:
        available.append(pr)
    else:
        missing.append(pr)

if len(missing) > 0:
    print(f"Side-by-side speed_fit: missing pairs {missing}")

if len(available) > 0:
    fig, axes = plt.subplots(1, len(available), figsize=(8 * len(available), 8),
                             sharex=True, sharey=True, squeeze=False)
    axes = axes[0]
    fig.suptitle("Average Activity Over Time (t, m, b)", fontsize=20)
    for ax, (ga, gb) in zip(axes, available):
        vids, colors = pair_speed_data[(ga, gb)]
        plot_speed_fit(ax, (ga, gb), vids, colors)
        ax.axvline(60, c='r', ls='--', lw=1.5)
        ax.set_title(f"{ga} vs {gb}", fontsize=16)
        ax.set_xlabel("Time (s)", fontsize=16)
        ax.tick_params(axis="both", labelsize=12)
        ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    axes[0].set_ylabel("Avg Speed (cm/s)", fontsize=16)
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/t_m_b_speed_fit_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# pooled comparison: all t vs all m vs all b (males and females combined)
tm_vids = {}
for g, prefixes in TREATMENT_GROUPS.items():
    tm_vids[g] = []
    for v in test_names:
        matches_group = False
        for pre in prefixes:
            if in_group(v, pre):
                matches_group = True
                break
        if matches_group:
            tm_vids[g].append(v)

smallest_group = None
for g, vs in tm_vids.items():
    print(f"pooled {g}: {len(vs)} videos")
    if smallest_group is None or len(vs) < smallest_group:
        smallest_group = len(vs)

if smallest_group < 2:
    print("pooled t/m/b comparison: need >=2 videos per group, skipping")
else:
    names = list(TREATMENT_GROUPS)

    tm_totals = {}
    for g in names:
        tm_totals[g] = []
        for v in tm_vids[g]:
            tm_totals[g].append(totals_cm[v])

    # boxplot of total distance with pairwise Welch t-tests
    box_data = []
    for g in names:
        box_data.append(tm_totals[g])

    fig, ax = plt.subplots(figsize=(5, 5))
    bp = ax.boxplot(box_data, patch_artist=True, widths=0.5)
    for patch, g in zip(bp["boxes"], names):
        patch.set_facecolor(TREATMENT_COLORS[g])
        patch.set_alpha(0.7)
    for element in ["whiskers", "caps", "medians", "fliers"]:
        for item in bp[element]:
            item.set_color("black")
    ax.set_xticks(range(1, len(names) + 1))
    ax.set_xticklabels(names)

    top = 0
    for g in names:
        group_max = max(tm_totals[g])
        if group_max > top:
            top = group_max

    step = top * 0.06
    y = top * 1.04
    p_labels = []
    for a, b in combinations(range(len(names)), 2):
        _, p = ttest_ind(tm_totals[names[a]], tm_totals[names[b]], equal_var=False)
        print(f"pooled {names[a]} vs {names[b]}: p = {p:.4g}")
        p_labels.append(f"{names[a]} v {names[b]}: {p:.3g}")
        ax.plot([a + 1, a + 1, b + 1, b + 1], [y, y + step * 0.2, y + step * 0.2, y], color="black", lw=1)
        ax.text((a + b) / 2 + 1, y + step * 0.25, stars(p), ha="center", va="bottom", fontsize=10)
        y += step
    ax.set_ylim(top=y + step)
    ax.set_ylabel("Total Distance (cm)")
    ax.set_title("t vs m vs b: Total Distance")
    ax.set_xlabel("p: " + ", ".join(p_labels), fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/t_vs_m_vs_b_boxplot.png", dpi=150)
    plt.close(fig)

    # avg speed over time, one panel per pooled group, same x/y scale (60 s baseline marked)
    fig, axes = plt.subplots(1, len(names), figsize=(8 * len(names), 8), sharex=True, sharey=True, squeeze=False)
    axes = axes[0]
    fig.suptitle("Average Activity Over Time (t vs m vs b)", fontsize=20)
    for ax, g in zip(axes, names):
        ax.axvline(60, c='r', ls='--', lw=1.5)
        plot_speed_fit(ax, (g,), tm_vids, TREATMENT_COLORS)
        ax.set_title(f"{g} (n = {len(tm_vids[g])} videos)", fontsize=16)
        ax.set_xlabel("Time (s)", fontsize=16)
        ax.tick_params(axis="both", labelsize=12)
        ax.legend(fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    axes[0].set_ylabel("Avg Speed (cm/s)", fontsize=16)
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/t_vs_m_vs_b_speed_fit.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # treatment group individual stuff
    fig, axes = plt.subplots(1, len(names), figsize=(8 * len(names), 8), sharex=True, sharey=True, squeeze=False)
    axes = axes[0]
    fig.suptitle("Per-Video Activity Over Time (t vs m vs b)", fontsize=20)
    for ax, g in zip(axes, names):
        ax.axvline(60, c='r', ls='--', lw=1.5)
        plot_speed_by_sex(ax, g, tm_vids)
        ax.set_title(f"{g} (n = {len(tm_vids[g])} videos)", fontsize=16)
        ax.set_xlabel("Time (s)", fontsize=16)
        ax.set_ylim(top=0.8)
        ax.tick_params(axis="both", labelsize=12)
        ax.legend(fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    axes[0].set_ylabel("Avg Speed (cm/s)", fontsize=16)
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/t_vs_m_vs_b_speed_individual_by_sex.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

# co plots the sexes on one graph's individual activity
for sex, prefixes in SEX_PLOTS.items():
    sex_vids = {pre: [v for v in test_names if in_group(v, pre)] for pre in prefixes}
    present = tuple(pre for pre in prefixes if len(sex_vids[pre]) > 0)
    if not present:
        print(f"{sex} individual co-plot: no videos found, skipping")
        continue

    fig, ax = plt.subplots(figsize=(11, 8))
    fig.suptitle(f"Per-Video Activity Over Time ({', '.join(present)})", fontsize=20)
    ax.axvline(60, c='r', ls='--', lw=1.5, label='Baseline')
    plot_speed_individual(ax, present, sex_vids, PREFIX_COLORS, bin_size=1)
    ax.legend(fontsize=10, loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0)
    ax.set_xlabel("Time (s)", fontsize=18)
    ax.set_ylabel("Avg Speed (cm/s)", fontsize=18)
    ax.set_ylim(0.1, 0.8)
    ax.tick_params(axis="both", labelsize=14)
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/{'_'.join(present)}_speed_individual.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import cv2
from scipy.stats import ttest_ind, linregress, t
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

# test_names = ['ACO1.mov', 'ACO2.mov', 'ACO3.mov', 'ACO4.mov', 'ACO5.mov',
#               # 'CACO4.mov', 'CACO5.mov', 
#               'CO1.mov', 'CO2.mov', 'CO3.mov', 'CO4.mov', 'CO5.mov',
#               # 'CAO4.mov', 'CAO5.mov'
#               ]

test_names = ['CO1d14.mov', 'CO1d42.mov', 'CO2d14.mov', 'CO2d42.mov', 'CO3d14.mov', 'CO3d42.mov', 'CO4d14.mov', 'CO4d42.mov', 'CO5d14.mov', 'CO5d42.mov',
             'ACO1.mov', 'ACO2.mov', 'ACO3.mov', 'ACO4.mov', 'ACO5.mov']

aco_totals_cm = {} # dict for per-enclosure totals
# ca_totals_cm = {}
co_totals_cm = {}

avg_speed_per_frame = dict()
speed_info_per_vid = dict()

for vid_name in test_names:
    csv_name = f"./WatershedAlgorithm/Output/Velocity/CalitVids{vid_name[:2]}/Tracked_{vid_name}_pwsBacklitV3_debug.csv"
    # csv_name = f"./AldenAlg/Tracked_{vid_name}_pwsBacklit.csv"
    df = pd.read_csv(csv_name)
    # BASE_PATH = "/Volumes/Crucial X9/Downloads/UROP Data Colletion 4-26-2026"

    df = df.iloc[4:].reset_index(drop=True)
    # cap = cv2.VideoCapture(f"{BASE_PATH}/{vid_name}")
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
    plt.savefig(f'./WatershedAlgorithm/CalitPlots/nVel2/{vid_name}_trajectory.png', dpi=150)
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
    s_arr = np.array(seconds, dtype=float)
    sp_arr = np.array(speeds,  dtype=float)

    bin_edges = np.arange(0, s_arr.max() + BIN_SIZE, BIN_SIZE)
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
    bin_sems = np.array(bin_sems)

    ax.bar(bin_centers, bin_means, width=BIN_SIZE * 0.85, color=pop_color(name), alpha=0.65)
    ax.errorbar(bin_centers, bin_means, yerr=bin_sems, fmt="none", color="black", capsize=2, linewidth=0.8)
    ax.set_title(name.replace(".mov", ""))
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Speed (cm/s)")

for ax in axes2[n:]:
    ax.set_visible(False)

fig2.tight_layout()
fig2.savefig("./WatershedAlgorithm/CalitPlots/nVel2/speed_over_time_BINNED10.png", dpi=150, bbox_inches="tight")
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
fig_box.savefig("./WatershedAlgorithm/CalitPlots/nVel2/pop_move_OVERALL.png", dpi=150)
plt.close()

# binned speed avg with smoothed n lin fit
bin_edges = np.arange(0, 300 + BIN_SIZE, BIN_SIZE)
bin_centers = bin_edges[:-1] + BIN_SIZE / 2

def reg_ci(x, y, xgrid, alpha=0.05):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    n = len(x)
    slope, intercept, r, p, se = linregress(x, y)
    yhat = slope * x + intercept
    s_err = np.sqrt(np.sum((y - yhat) ** 2) / (n - 2))
    xbar = x.mean()
    sxx = np.sum((x - xbar) ** 2)
    ygrid = slope * xgrid + intercept
    tcrit = t.ppf(1 - alpha / 2, df=n - 2)
    ci = tcrit * s_err * np.sqrt(1 / n + (xgrid - xbar) ** 2 / sxx)
    slope = f"{slope:.3g}"
    intercept = f"{intercept:.3g}"
    p = f"{p:.3g}"
    return ygrid, ygrid - ci, ygrid + ci, p, slope, intercept

pop_groups = {"ACO": [], "CO": []}

for vid_name, (seconds, speeds) in speed_info_per_vid.items():
    key = "ACO" if vid_name.startswith("ACO") else "CO"
    s_arr = np.array(seconds, dtype=float)
    sp_arr = np.array(speeds, dtype=float)

    vid_bin_means = []
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        vals = sp_arr[(s_arr >= lo) & (s_arr < hi)]
        vals = vals[~np.isnan(vals)]
        vid_bin_means.append(np.nanmean(vals) if len(vals) else np.nan)

    pop_groups[key].append(vid_bin_means)


fig4, ax4 = plt.subplots(1, 1, figsize=(8, 8))
fig4.suptitle("Average Activity Over Time", fontsize=20) # keep default size


pop_styles = {"CO": {"color": "steelblue", "label": "CO"}, "ACO": {"color": "deeppink", "label": "ACO"}, }


for pop, style in pop_styles.items():
    ax = ax4
    mat = np.array(pop_groups[pop])
    means = np.nanmean(mat, axis=0)
    sems = np.nanstd(mat, axis=0) / np.sqrt((~np.isnan(mat)).sum(axis=0))
    c = style["color"]

    ax.scatter(bin_centers, means, color=c, alpha=0.5)
    # ax.errorbar(bin_centers, means, yerr=sems, fmt="none", color=c, capsize=2, linewidth=0.8, alpha=0.5)

    valid = ~np.isnan(means)
    x_v, y_v = bin_centers[valid], means[valid]

    # smoothed = lowess(y_v, x_v, frac=0.3, return_sorted=True)
    # ax.plot(smoothed[:, 0], smoothed[:, 1], color=c, lw=2, label=f"{style['label']} smoothed")

    # first 100s fit
    first = x_v <= 100
    if first.sum() >= 3:
        x1 = x_v[first]
        y1 = y_v[first]
        xgrid1 = np.linspace(x1.min(), x1.max(), 200)
        yfit1, lo1, hi1, p1, slope, intercept = reg_ci(x1, y1, xgrid1)
        ax4.plot(xgrid1, yfit1, color=c, lw=2.2, label=f"{style['label'][0]}", alpha=0.4)
        if pop=='ACO':
            ax4.text(15, 0.5, f"{slope}x+{intercept}\np={p1}", color=c)
        else:
            ax4.text(32.3, 2, f"{slope}x+{intercept}\np={p1}", color=c)
        ax4.fill_between(xgrid1, lo1, hi1, color="gray", alpha=0.22)

    # middle 100s fit
    # middle = x_v.all(x_v, where=(100 <= x_v <= 200))
    # if middle.sum() >= 3:
    x3 = x_v[np.where(np.logical_and(x_v>=100, x_v<=200))]
    y3 = y_v[np.where(np.logical_and(x_v>=100, x_v<=200))]
    xgrid3 = np.linspace(x3.min(), x3.max(), 200)
    yfit3, lo3, hi3, p3, slope, intercept = reg_ci(x3, y3, xgrid3)
    if pop=='ACO':
        ax4.text(121, 0.43, f"{slope}x+{intercept}\np={p3}", color=c)
    else:
        ax4.text(126, 1.72, f"{slope}x+{intercept}\np={p3}", color=c)
    ax4.plot(xgrid3, yfit3, color=c, lw=2.2, alpha=0.4)
    ax4.fill_between(xgrid3, lo3, hi3, color="gray", alpha=0.22)


    # last 100s fit
    last = x_v >= 200
    if last.sum() >= 3:
        x2 = x_v[last]
        y2 = y_v[last]
        xgrid2 = np.linspace(x2.min(), x2.max(), 200)
        yfit2, lo2, hi2, p2, slope, intercept = reg_ci(x2, y2, xgrid2)
        if pop=='ACO':
            ax4.text(208, 0.5, f"{slope}x+{intercept}\np={p2}", color=c)
        else:
            ax4.text(212, 1.67, f"{slope}x+{intercept}\np={p2}", color=c)
        ax4.plot(xgrid2, yfit2, color=c, lw=2.2, alpha=0.4)
        ax4.fill_between(xgrid2, lo2, hi2, color="gray", alpha=0.22)

    slope, intercept, r, p_val, *_ = linregress(x_v, y_v)
    print(pop, p_val)

    y_fit = slope * x_v + intercept
    # ax.plot(x_v, y_fit, color=c, lw=1, label=f"{style['label']} fit (r={round(r, 2)}, p={round(p_val, 4)})")


ax.legend(fontsize=10)
ax.set_xlabel("Time (s)", fontsize=18)
ax.set_ylabel("Avg Speed (cm/s)", fontsize=18)
# ax.set_title(style["label"], fontsize=24)
ax.tick_params(axis='both', labelsize=14)


fig4.tight_layout()
fig4.savefig("./WatershedAlgorithm/CalitPlots/nVel2/pop_speed_averaged_fit.png", dpi=150)
plt.close()

# Boxplot — Cd14 vs Cd42

co_d14_totals = {k: v for k, v in co_totals_cm.items() if "d14" in k}
co_d42_totals = {k: v for k, v in co_totals_cm.items() if "d42" in k}

t_d14_d42, p_d14_d42 = ttest_ind(
    list(co_d14_totals.values()),
    list(co_d42_totals.values()),
    equal_var=False
)
print(f"p_co_d14_vs_d42: {p_d14_d42}")

day_populations = {"Cd14": list(co_d14_totals.values()), "Cd42": list(co_d42_totals.values())}
day_colors = {"Cd14": "steelblue", "Cd42": "darkorange"}

fig_box2, ax_box2 = plt.subplots(figsize=(3.5, 4))
pop_names2 = list(day_populations.keys())
data2 = [day_populations[p] for p in pop_names2]
bp2 = ax_box2.boxplot(data2, patch_artist=True, widths=0.45)

for patch, pop in zip(bp2["boxes"], pop_names2):
    patch.set_facecolor(day_colors[pop])
    patch.set_alpha(0.7)

for element in ["whiskers", "caps", "medians", "fliers"]:
    for item in bp2[element]:
        item.set_color("black")

ax_box2.set_xticks(range(1, len(pop_names2) + 1))
ax_box2.set_xticklabels(pop_names2)

y_max2 = max(max(v) for v in day_populations.values())
for step, (i, j, p) in enumerate([(0, 1, p_d14_d42)]):
    xi, xj = i + 1, j + 1
    y = y_max2 * (1.08 + 0.07 * step)
    ax_box2.plot([xi, xi, xj, xj], [y, y * 1.01, y * 1.01, y], color="black", lw=1)
    ax_box2.text((xi + xj) / 2, y * 1.015, "*" if p < 0.001 else "", ha="center", va="bottom", fontsize=10)

ax_box2.set_ylabel("Total Distance (cm)")
ax_box2.set_title("CO: Day 14 vs Day 42")
ax_box2.text(0.03, 0.97, f"p = {p_d14_d42:.3g}", transform=ax_box2.transAxes, ha="left", va="top")

fig_box2.tight_layout()
fig_box2.savefig("./WatershedAlgorithm/CalitPlots/nVel2/co_d14_vs_d42_boxplot.png", dpi=150)
plt.close()


# pop_speed_averaged_fit for C d14 vs C d42 

# Separate speed_info_per_vid into the two day groups
day_groups = {"Cd14": [], "Cd42": []}

for vid_name, (seconds, speeds) in speed_info_per_vid.items():
    if not vid_name.startswith("CO"):
        continue
    key = "Cd14" if "d14" in vid_name else "Cd42"
    s_arr = np.array(seconds, dtype=float)
    sp_arr = np.array(speeds,  dtype=float)

    vid_bin_means = []
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        vals = sp_arr[(s_arr >= lo) & (s_arr < hi)]
        vals = vals[~np.isnan(vals)]
        vid_bin_means.append(np.nanmean(vals) if len(vals) else np.nan)
    day_groups[key].append(vid_bin_means)

fig5, ax5 = plt.subplots(1, 1, figsize=(8, 8))
fig5.suptitle("Average Activity Over Time (CO: d14 vs d42)", fontsize=20)

day_styles = {
    "Cd14": {"color": "steelblue",   "label": "Cd14"},
    "Cd42": {"color": "darkorange",  "label": "Cd42"},
}

for pop, style in day_styles.items():
    mat = np.array(day_groups[pop])
    means = np.nanmean(mat, axis=0)
    c = style["color"]

    ax5.scatter(bin_centers, means, color=c, alpha=0.5)

    valid = ~np.isnan(means)
    x_v, y_v = bin_centers[valid], means[valid]

    # first 100 s
    first = x_v <= 100
    if first.sum() >= 3:
        x1, y1 = x_v[first], y_v[first]
        xgrid1 = np.linspace(x1.min(), x1.max(), 200)
        yfit1, lo1, hi1, p1, slope1, intercept1 = reg_ci(x1, y1, xgrid1)
        ax5.plot(xgrid1, yfit1, color=c, lw=2.2, alpha=0.6, label=style["label"])
        ax5.fill_between(xgrid1, lo1, hi1, color="gray", alpha=0.18)
        offset = 0.05 if pop == "Cd14" else -0.12   # nudge labels apart
        ax5.text(15, y_v[first].mean() + offset, f"{slope1}x+{intercept1}\np={p1}", color=c, fontsize=8)

    # middle 100–200 s
    mid_mask = np.logical_and(x_v >= 100, x_v <= 200)
    x3, y3 = x_v[mid_mask], y_v[mid_mask]
    if len(x3) >= 3:
        xgrid3 = np.linspace(x3.min(), x3.max(), 200)
        yfit3, lo3, hi3, p3, slope3, intercept3 = reg_ci(x3, y3, xgrid3)
        ax5.plot(xgrid3, yfit3, color=c, lw=2.2, alpha=0.6)
        ax5.fill_between(xgrid3, lo3, hi3, color="gray", alpha=0.18)
        offset = 0.05 if pop == "Cd14" else -0.12
        ax5.text(130, y3.mean() + offset, f"{slope3}x+{intercept3}\np={p3}", color=c, fontsize=8)

    # last 200–300 s
    last = x_v >= 200
    if last.sum() >= 3:
        x2, y2 = x_v[last], y_v[last]
        xgrid2 = np.linspace(x2.min(), x2.max(), 200)
        yfit2, lo2, hi2, p2, slope2, intercept2 = reg_ci(x2, y2, xgrid2)
        ax5.plot(xgrid2, yfit2, color=c, lw=2.2, alpha=0.6)
        ax5.fill_between(xgrid2, lo2, hi2, color="gray", alpha=0.18)
        offset = 0.05 if pop == "Cd14" else -0.12
        ax5.text(210, y2.mean() + offset, f"{slope2}x+{intercept2}\np={p2}", color=c, fontsize=8)

    slope_ov, intercept_ov, r_ov, p_ov, *_ = linregress(x_v, y_v)
    print(f"{pop}: overall p={p_ov:.4g}, r={r_ov:.3f}")

ax5.legend(fontsize=10)
ax5.set_xlabel("Time (s)", fontsize=18)
ax5.set_ylabel("Avg Speed (cm/s)", fontsize=18)
ax5.tick_params(axis="both", labelsize=14)

fig5.tight_layout()
fig5.savefig("./WatershedAlgorithm/CalitPlots/nVel2/co_d14_vs_d42_speed_fit.png", dpi=150)
plt.close()

#-------------
# we have all of the different avg. points for each sample (5 dots at each sample for A and C) instead of overall A and C
# linear fits for the first hundred and last hundred seconds
# have the confidence interval around the above linear lines.

# # binned speed avg with smoothed n lin fit
# bin_edges= np.arange(0, 300 + BIN_SIZE, BIN_SIZE)
# bin_centers = bin_edges[:-1] + BIN_SIZE / 2

# pop_groups = {"ACO": [], "CO": []}
# for vid_name, (seconds, speeds) in speed_info_per_vid.items():
#     key = "ACO" if vid_name.startswith("ACO") else "CO"
#     s_arr = np.array(seconds, dtype=float)
#     sp_arr = np.array(speeds,  dtype=float)

#     vid_bin_means = []
#     for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
#         vals = sp_arr[(s_arr >= lo) & (s_arr < hi)]
#         vals = vals[~np.isnan(vals)]
#         vid_bin_means.append(np.nanmean(vals) if len(vals) else np.nan)
#     pop_groups[key].append(vid_bin_means)

# fig4, ax4 = plt.subplots(figsize=(8, 12))
# fig4.suptitle("Average Activity Over Time", fontsize=24)

# pop_styles = {
#     "ACO": {"color": "deeppink", "label": "ACO"},
#     "CO": {"color": "steelblue", "label": "CO"},
# }

# for pop, style in pop_styles.items():
#     mat = np.array(pop_groups[pop], dtype=float)   # shape: n_videos x n_bins
#     c = style["color"]

#     # plot each sample/video as its own set of dots
#     first_point = True
#     for row in mat:
#         valid = ~np.isnan(row)
#         ax4.scatter(bin_centers[valid], row[valid], color=c, alpha=0.25, s=20, label=f"{style['label']} samples" if first_point else None)
#         first_point = False

#     # population mean + SEM across videos
#     means = np.nanmean(mat, axis=0)
#     sems = np.nanstd(mat, axis=0) / np.sqrt((~np.isnan(mat)).sum(axis=0))
#     valid = ~np.isnan(means)
#     x_v, y_v = bin_centers[valid], means[valid]

#     # ax4.errorbar(x_v, y_v, yerr=sems[valid], fmt="o", color=c, capsize=2, alpha=0.8, label=style["label"])

# ax4.set_xlabel("Time (s)", fontsize=24)
# ax4.set_ylabel("Avg Speed (cm/s)", fontsize=24)
# # ax4.set_title("Average Activity Over Time", fontsize=24)
# ax4.tick_params(axis="both", labelsize=14)
# ax4.legend(fontsize=14)
# fig4.tight_layout()
# fig4.savefig("./WatershedAlgorithm/CalitPlotswUROPVids/pop_speed_averaged_fit_updated.png", dpi=150, bbox_inches="tight")
# plt.close(fig4)
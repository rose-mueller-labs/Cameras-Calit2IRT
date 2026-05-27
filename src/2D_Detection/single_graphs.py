import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def parse_coord(s):
    try:
        s = str(s).strip().strip("()")
        x, y = s.split("!")
        return float(x), float(y)
    except Exception:
        return np.nan, np.nan

# params
vid_name = "ACO2.mov" 
csv_path = f"./WatershedAlgorithm/Output/Velocity/CalitVidsAC/Tracked_{vid_name}_pwsBacklitV3_debug.csv"
video_dir = "/Volumes/Crucial X9/Downloads/Calit2 Data Collection 05-06-2026"
fps = 120
MAX_SEC = 300
PX_TO_CM = 10/1730
N_FLIES = 20

df = pd.read_csv(csv_path)
df = df.iloc[4:].reset_index(drop = True)

start_frame = df["frame"].iloc[0]
df = df[df["frame"] < start_frame + fps * MAX_SEC].reset_index(drop = True)

id_cols = [c for c in df.columns if c.startswith("ID_")]

# parse & interplate
coords = {}
for col in id_cols:
    xy = df[col].apply(parse_coord).tolist()
    coords[col] = pd.DataFrame(xy, columns = ["x", "y"], index = df["frame"].values)
    coords[col] = coords[col].interpolate(method = "linear", limit_direction = "both")

# plot 1: trajectories
fig1, ax1 = plt.subplots(figsize = (7, 6))
colors = plt.cm.tab20.colors

for i, col in enumerate(id_cols):
    xy = coords[col]
    ax1.plot(xy["x"], xy["y"], color = colors[i % 20], lw = 0.8, label = col)

ax1.invert_yaxis()
ax1.set_title(f"{vid_name.replace('.mov', '')} — Trajectories")
ax1.set_xlabel("X (px)")
ax1.set_ylabel("Y (px)")
ax1.legend(id_cols, fontsize = 3, loc = "upper right")
fig1.tight_layout()
fig1.savefig(f"./WatershedAlgorithm/CalitPlots/nVel2/{vid_name}_trajectory.png", dpi = 150)
plt.close(fig1)

# per frame avg speed
all_frame_dists = []
for col in id_cols:
    xy = coords[col]
    dx = xy["x"].diff()
    dy = xy["y"].diff()
    dist  = np.sqrt(dx**2 + dy**2).fillna(0)
    all_frame_dists.append(dist)

dist_matrix  = pd.concat(all_frame_dists, axis = 1).replace(0, np.nan)
frame_avg_speed = (dist_matrix.sum(axis = 1) / N_FLIES) * PX_TO_CM * fps   # cm/s

seconds = [(f-start_frame)/fps for f in df["frame"].values]

# plot 2: Speed over time (line plot instead of binned) 
# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.rolling.html

fig2, ax2 = plt.subplots(figsize = (10, 4))

ax2.plot(seconds, frame_avg_speed, lw = 0.7, alpha = 0.5, color = "grey", label = "_nolegend_")

smooth = pd.Series(frame_avg_speed.values, index = seconds).rolling(fps, center = True).mean()
ax2.plot(seconds, smooth, lw = 1.5, color = "steelblue", label = "1 s rolling mean")

ax2.set_title(f"{vid_name.replace('.mov', '')} — Avg Speed Over Time")
ax2.set_xlabel("Time (s)")
ax2.set_ylabel("Speed (cm/s)")
ax2.legend()
fig2.tight_layout()
fig2.savefig(f"./WatershedAlgorithm/CalitPlots/nVel2/{vid_name}_speed_line.png", dpi = 150)
plt.close(fig2)
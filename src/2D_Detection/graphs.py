import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df = pd.read_csv("./WatershedAlgorithm/Output/Backlit/UROPVids/Tracked_ACO1.MOV_pwsBacklit.csv")

print(df.head())

# delete first 4 frames
df = df.iloc[4:].reset_index(drop=True)

id_cols = [c for c in df.columns if c.startswith("ID_")]

def parse_coord(s):
    '''cuz it's ! i want it to be ,'''
    try:
        s = str(s).strip().strip("()")
        x, y = s.split("!")
        return float(x), float(y)
    except Exception:
        return np.nan, np.nan

coords = {}
for col in id_cols:
    xy = df[col].apply(parse_coord).tolist()
    coords[col] = pd.DataFrame(xy, columns=["x", "y"], index=df["frame"].values)

for col in id_cols:
    coords[col] = coords[col].interpolate(method="linear", limit_direction="both")

STILL_THRESHOLD_PX = 10 # if movement is greater than 10 pixels
STILL_MIN_FRAMES = 3 # it needs to be still for like 3ish frames?

results = {}
for col in id_cols:
    xy = coords[col]
    dx = xy["x"].diff()
    dy = xy["y"].diff()
    dist_per_frame = np.sqrt(dx**2 + dy**2).fillna(0)

    total_dist = dist_per_frame.sum()

    is_moving = dist_per_frame > STILL_THRESHOLD_PX
    move_things = 0
    still_things = 0
    run_len = 1
    for i in range(1, len(is_moving)):
        if is_moving.iloc[i] == is_moving.iloc[i - 1]:
            run_len += 1
        else:
            if is_moving.iloc[i - 1]:
                move_things += 1
            else:
                if run_len >= STILL_MIN_FRAMES:
                    still_things += 1
            run_len = 1
    if is_moving.iloc[-1]:
        move_things += 1
    elif run_len >= STILL_MIN_FRAMES:
        still_things += 1

    avg_speed = dist_per_frame[dist_per_frame > 0].mean()

    results[col] = {"total_distance_px": round(total_dist, 1), "move_things": move_things, "still_things": still_things, "avg_speed_px_per_frame": round(avg_speed, 2)}

metrics_df = pd.DataFrame(results).T
print("fly metrics df printed")
print(metrics_df.to_string())

# plots
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
colors = plt.cm.tab10.colors
fly_labels = list(metrics_df.index)

# trajectories
ax = axes[0, 0]
for i, col in enumerate(id_cols):
    xy = coords[col]
    ax.plot(xy["x"], xy["y"], color=colors[i % 10], lw=1, label=col)
ax.invert_yaxis()
ax.set_title("Trajectories"); ax.set_xlabel("X (px)"); ax.set_ylabel("Y (px)")
# ax.legend(fontsize=7)

# speed over time
ax = axes[0, 1]
for i, col in enumerate(id_cols):
    xy = coords[col]
    dist = np.sqrt(xy["x"].diff()**2 + xy["y"].diff()**2).fillna(0)
    ax.plot(df["frame"].values, dist.values, color=colors[i % 10], lw=1, label=col)
ax.axhline(STILL_THRESHOLD_PX, color="red", linestyle="--", lw=1)
ax.set_title("Speed Over Time"); ax.set_xlabel("Frame"); ax.set_ylabel("px / frame")
# ax.legend(fontsize=7)

# total distance
ax = axes[1, 0]
ax.bar(fly_labels, metrics_df["total_distance_px"].values, color=colors[:len(fly_labels)])
ax.set_title("Total Distance (px)"); ax.set_ylabel("Pixels")

# move vs still things
ax = axes[1, 1]
x = np.arange(len(fly_labels))
ax.bar(x - 0.2, metrics_df["move_things"].values,  width=0.4, label="Moving", color="steelblue")
ax.bar(x + 0.2, metrics_df["still_things"].values, width=0.4, label="Still",  color="salmon")
ax.set_xticks(x); ax.set_xticklabels(fly_labels)
ax.set_title("Move vs Still Things"); ax.set_ylabel("Count")
# ax.legend()

plt.tight_layout()
plt.savefig("./WatershedAlgorithm/Plots/Tracked_ACO1.MOV_pwsBacklit.png", dpi=150, bbox_inches="tight")
plt.show()
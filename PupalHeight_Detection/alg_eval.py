'''
evaluation of the output CSVs from the 3 algs
--------------------------------------------------
per video + alg it calculates:
  1. avg length of an id (frames + seconds)  -> CSV
  2. total flies detected per frame          -> line plot (1 PNG per video, 1 line per alg)
  3. avg flies detected per frame            -> CSV
  + unique ID count (a lot more ids than flies = lots of id swaps / re-spawns)
--------------------------------------------------
CSV format expected (same for all 3 algs):
  frame, ID_0, ID_1, ...    cells are "(cx!cy)" or blank when that id isn't detected that frame
'''

# TODO: Make plot of Each ID and draw a line for which frames they are present in.

import csv
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RESULTS_LOC = "/Volumes/Crucial X9/Cameras-Calit2IRT/PupalHeight_Detection/TestingOutput"
EVAL_LOC = "/Volumes/Crucial X9/Cameras-Calit2IRT/PupalHeight_Detection/TestingEval"

os.makedirs(EVAL_LOC, exist_ok=True)

# Params
FPS = 119 # only used to turn id length in frames into seconds
MIN_ID_LEN = 1 # ignore ids seen in fewer than this many frames (applied to ALL algs so the compare is fair)

VIDEOS = ['fb1.mov', 'fm2.mov', 'fb2.mov', 'fb3.mov', 'fb4.mov', 'fm1.mov', 'fm3.mov', 
          'fm4.mov', 'ft1.mov', 'ft2.mov', 'ft3.mov', 'ft4.mov', 'mb1.mov', 'mb2.mov', 
          'mb3.mov', 'mb4.mov', 'mm1.mov', 'mm2.mov', 'mm3.mov', 'mm4.mov', 'mt1.mov', 
          'mt2.mov', 'mt3.mov', 'mt4.mov']

# alg name -> csv suffix (the part after {video}_ in the filename)
ALGS = {
    'orb': 'orb_watershed_velocity_hungarian_pathing.csv',
}

ALG_COLORS = {'orb': 'tab:pink'}


# helpers

def load_csv(csv_path):
    """
    Returns (frames, id_frames)
      frames    : list of frame numbers in order
      id_frames : id -> set of frames it was detected in
    """
    frames = []
    id_frames = {}
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        id_cols = [c for c in reader.fieldnames if c.startswith('ID_')]
        for row in reader:
            fr = int(row['frame'])
            frames.append(fr)
            for col in id_cols:
                if row[col].strip() != '':
                    id_frames.setdefault(col, set()).add(fr)
    return frames, id_frames


def flies_per_frame(frames, id_frames):
    """count of ids present in each frame, aligned w frames"""
    counts = {fr: 0 for fr in frames}
    for fr_set in id_frames.values():
        for fr in fr_set:
            counts[fr] += 1
    return [counts[fr] for fr in frames]


# MAIN LOOP

summary_rows = []

for vid_name in VIDEOS:
    plt.figure(figsize=(10, 5))
    plotted_any = False

    for alg, suffix in ALGS.items():
        csv_path = f"{RESULTS_LOC}/{vid_name}_{suffix}"
        if not os.path.exists(csv_path):
            print(f"[MISSING] {csv_path}")
            continue

        frames, id_frames = load_csv(csv_path)

        # drop ids that are too short
        id_frames = {k: v for k, v in id_frames.items() if len(v) >= MIN_ID_LEN}

        if not frames:
            print(f"[EMPTY] {csv_path}")
            continue

        counts = flies_per_frame(frames, id_frames)

        n_ids = len(id_frames)
        avg_len_frames = sum(len(v) for v in id_frames.values()) / n_ids if n_ids else 0.0
        avg_flies = sum(counts) / len(counts)

        summary_rows.append({
            'video': vid_name,
            'algorithm': alg,
            'frames_processed': len(frames),
            'unique_ids': n_ids,
            'avg_id_length_frames': round(avg_len_frames, 2),
            'avg_id_length_sec': round(avg_len_frames / FPS, 3),
            'avg_flies_per_frame': round(avg_flies, 2),
        })

        plt.plot(frames, counts, label=alg, color=ALG_COLORS.get(alg), linewidth=1.5)
        plotted_any = True
        print(f"{vid_name} | {alg}: ids={n_ids}, avg_id_len={round(avg_len_frames, 1)} frames, avg_flies={round(avg_flies, 2)}")

    if plotted_any:
        plt.title(f'Flies detected per frame - {vid_name}')
        plt.xlabel('Frame')
        plt.ylabel('Flies detected')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{EVAL_LOC}/{vid_name}_flies_per_frame.png", dpi=150)
    plt.close()

# per-alg averages across all videos
for alg in ALGS:
    rows = [r for r in summary_rows if r['algorithm'] == alg]
    if rows:
        summary_rows.append({
            'video': 'ALL_AVG',
            'algorithm': alg,
            'frames_processed': round(sum(r['frames_processed'] for r in rows) / len(rows), 1),
            'unique_ids': round(sum(r['unique_ids'] for r in rows) / len(rows), 2),
            'avg_id_length_frames': round(sum(r['avg_id_length_frames'] for r in rows) / len(rows), 2),
            'avg_id_length_sec': round(sum(r['avg_id_length_sec'] for r in rows) / len(rows), 3),
            'avg_flies_per_frame': round(sum(r['avg_flies_per_frame'] for r in rows) / len(rows), 2),
        })

# write the comparison CSV
eval_csv = f"{EVAL_LOC}/eval_summary.csv"
if summary_rows:
    fieldnames = ['video', 'algorithm', 'frames_processed', 'unique_ids', 'avg_id_length_frames', 'avg_id_length_sec', 'avg_flies_per_frame']
    with open(eval_csv, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"Saved {eval_csv}")
else:
    print("No CSVs found, nothing saved")
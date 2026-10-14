'''
ORB -> watershed -> velocity prediction -> Hungarian matching fly tracker
--------------------------------------------------
pipeline:
  0. Darkness gate: flies are a bit darker than their *local* background, so a black-hat transform
     (closing - gray) lights up small dark things and ignores big/flat areas. ORB only looks near those.
  1. ORB keypoints are detected on every frame (restricted to the arena AND the dark gate, so walls/box edges don't count).
     ORB = FAST corners + descriptors, so on a fly the keypoints land on the body outline / legs / wings.
  2. Keypoints are stamped as filled circles and morphologically closed so the clusters on each fly
     merge into one blob per fly -> this is the fg_mask (the "sure foreground" for watershed).
  3. Watershed splits blobs that touch, then contours get filtered by area / circularity / arena.
  4. Tracking is the same as watershed_velocity_hungarian_pathing.py: Hungarian matching, EMA velocity,
     ghost positions for lost flies, ID-swap guard, and new-fly gate.
--------------------------------------------------
things to tune first (in order):
- ORB_FAST_THRESHOLD / ORB_N_FEATURES: too few keypoints = flies missing, too many = bg noise
- KP_RADIUS / KP_CLOSE_KERNEL: controls how far apart keypoints can be and still merge into one fly
- MIN/MAX_CONTOUR_AREA: look at the debug png's to pick these
- DARK_THRESH: gray levels darker than the local bg that count as "fly". Raise if walls/shadows leak in, lower if faint flies vanish
- DARK_KERNEL: must be bigger than a fly (px) and smaller than anything you want ignored as 'background'
--------------------------------------------------
https://docs.opencv.org/4.x/d1/d89/tutorial_py_orb.html
https://www.geeksforgeeks.org/dsa/hungarian-algorithm-assignment-problem-set-1-introduction/
https://cp-algorithms.com/graph/hungarian-algorithm.html
'''

import cv2
import numpy as np
import random
import csv
from collections import deque
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt
import os

DEBUG = True
SHOW_PREVIEW = True # plt window of the 1st frame (ORB kps / kp blobs / fg mask) per video; close it to continue

# Params
MAX_LOST_FRAMES = 35 # longer flights at 119 fps need more buffer
MIN_LIFETIME = 5 # frames before an ID is drawn/exported
MAX_PATH_LENGTH = 50

MIN_CONTOUR_AREA = 350
MAX_CONTOUR_AREA = 2600

# circ ranges
CIRC_MIN = 0.15
CIRC_MAX = 0.80

VELOCITY_ALPHA = 0.8 # EMA weight (higher = more reactive)

# distance thresholds calibrated for 119 fps + possible flight
DISTANCE_THRESHOLD = 60 # walking match radius (px)
FLIGHT_DISTANCE_THRESHOLD = 350 # covers full flight arc across lost frames

RECOVERY_THRESHOLD = DISTANCE_THRESHOLD * 2 # same as walking for on-ground recovery

FLYING_SPEED_THRESHOLD = 15 # px/frame speed they gotta go to be flying

STOP_SEC = 60 # seconds of video to process per clip

# ORB params
ORB_N_FEATURES = 3000 # max keypoints per frame
ORB_SCALE_FACTOR = 1.2
ORB_N_LEVELS = 4 # flies are small so we dont need a deep pyramid
ORB_EDGE_THRESHOLD = 15 # border where kps are ignored, keep <= patch size
ORB_PATCH_SIZE = 15 # small patch so tiny flies still get kps
ORB_FAST_THRESHOLD = 10 # lower = more kps (default is 20)

# kp -> blob params
KP_RADIUS = 7 # each keypoint is drawn as a filled circle this big (px)
KP_CLOSE_KERNEL = 15 # closing kernel that merges the kp circles on one fly into one blob
KP_CLOSE_ITERS = 2
KP_MIN_NEIGHBORS = 1 # drop kps with fewer than this many other kps within KP_CLOSE_KERNEL px (lone noise kps)

USE_DARK_GATE = True # flies are a lil darker than the bg: only let ORB look near dark spots, and AND the blobs w the dark mask
DARK_KERNEL = 61 # black-hat window (px); bigger than a fly, so a fly counts as 'small dark thing' but a wall/shadow band doesnt
DARK_THRESH = 15 # gray levels darker than the local bg to count as fly
DARK_GATE_DILATE = 15 # how far around each dark spot ORB is allowed to look (px)

SPLIT_TOUCHING = True # distance transform seeds so two flies brushing by dont become one big fly
SEED_WINDOW = 31 # roughly one fly wide (px)
SEED_FRAC = 0.55 # a px is a seed if its dist >= this * the biggest dist in the window
SPLIT_MIN_AREA = 1200 # only blobs at least this big get split. A single fly is smaller, so it goes to watershed whole (elongated flies were getting cut in half)

WS_BG_DILATE = 15 # sure_bg = fg blobs grown by this much, so watershed only floods a thin band around each fly (NOT the whole arena)

ARENA_THRESH = None # None = auto (median brightness - ARENA_MARGIN); or a fixed number like 175 for the og behavior
ARENA_MARGIN = 14 # auto mode: arena = px within this many levels of the median brightness. Walls (~25 darker than the floor) fall outside
ARENA_ERODE = 25 # how far to pull in from the box walls (was 53; the dark gate now handles walls so flies hugging the edge can be kept)

# vel helpers

def update_velocity(obj_id, new_center):
    """Exponential moving average of (vx, vy) per fly."""
    if obj_id in object_velocities:
        prev_cx, prev_cy = last_centers[obj_id]
        new_vx = new_center[0] - prev_cx
        new_vy = new_center[1] - prev_cy
        old_vx, old_vy = object_velocities[obj_id]
        object_velocities[obj_id] = (
            VELOCITY_ALPHA * new_vx + (1 - VELOCITY_ALPHA) * old_vx,
            VELOCITY_ALPHA * new_vy + (1 - VELOCITY_ALPHA) * old_vy,
        )
    else:
        object_velocities[obj_id] = (0.0, 0.0)
    last_centers[obj_id] = new_center


def get_velocity(obj_id):
    return object_velocities.get(obj_id, (0.0, 0.0))


def predict_position(bbox, obj_id):
    """Return the predicted center for obj_id one frame ahead."""
    cx, cy = get_center(bbox)
    vx, vy = get_velocity(obj_id)
    return (cx + vx, cy + vy)


def predicted_distance(pred_center, curr_bbox):
    """Euclidean distance from a predicted (float) center to a detected bbox center."""
    cx2, cy2 = get_center(curr_bbox)
    return np.sqrt((pred_center[0] - cx2) ** 2 + (pred_center[1] - cy2) ** 2)


def is_flying(obj_id):
    vx, vy = get_velocity(obj_id)
    return np.sqrt(vx ** 2 + vy ** 2) > FLYING_SPEED_THRESHOLD


def size_penalty(bbox1, bbox2):
    """
    Distance-comparable penalty for area mismatch between two bboxes.
    Scaled to be additive with px distances (0 to 50 px equivalent).
    """
    a1 = bbox1[2] * bbox1[3]
    a2 = bbox2[2] * bbox2[3]
    return abs(a1 - a2) / max(a1, a2, 1) * 50


# ghost position fixes

def advance_ghost(obj_id):
    """
    Each frame a fly stays lost, advance its stored ghost position by its last known velocity (with decay), so recovery searches the right area.
    """
    if obj_id not in lost_objects:
        return
    data = lost_objects[obj_id]
    vx, vy = get_velocity(obj_id)
    # decay velocity while lost so ghost doesn't run forever
    object_velocities[obj_id] = (vx * 0.90, vy * 0.90)
    vx, vy = get_velocity(obj_id)

    # update ghost bbox center
    x, y, w, h = data['bbox']
    cx = x + w // 2 + vx
    cy = y + h // 2 + vy
    new_x = int(cx - w // 2)
    new_y = int(cy - h // 2)
    data['bbox'] = (new_x, new_y, w, h)
    data['ghost_center'] = (cx, cy)


def ghost_center(obj_id):
    """Return the ghost (predicted) center for a lost fly."""
    if obj_id in lost_objects:
        return lost_objects[obj_id].get('ghost_center', get_center(lost_objects[obj_id]['bbox']))
    return None


# og helpers

def get_center(bbox):
    x, y, w, h = bbox
    return (x + w // 2, y + h // 2)


def calculate_distance(bbox1, bbox2):
    x1, y1 = get_center(bbox1)
    x2, y2 = get_center(bbox2)
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def get_unique_color(obj_id):
    if obj_id not in colors:
        rng = random.Random(obj_id * 6364136223846793005 + 1442695040888963407)
        colors[obj_id] = (rng.randint(30, 225), rng.randint(30, 225), rng.randint(30, 225))
    return colors[obj_id]


def draw_paths(frame, paths, obj_id):
    if obj_id in paths and len(paths[obj_id]) > 1:
        color = get_unique_color(obj_id)
        points = list(paths[obj_id])
        for i in range(len(points) - 1):
            cv2.line(frame, points[i], points[i + 1], color, 2)
        cv2.circle(frame, points[-1], 3, color, -1)


# ORB

orb = cv2.ORB_create(
    nfeatures=ORB_N_FEATURES,
    scaleFactor=ORB_SCALE_FACTOR,
    nlevels=ORB_N_LEVELS,
    edgeThreshold=ORB_EDGE_THRESHOLD,
    patchSize=ORB_PATCH_SIZE,
    fastThreshold=ORB_FAST_THRESHOLD,
)


def get_arena_mask(gray):
    """Biggest bright region = the arena. Eroded so the box walls dont make keypoints."""
    thresh = ARENA_THRESH if ARENA_THRESH is not None else float(np.median(gray)) - ARENA_MARGIN
    _, white_region = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(white_region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    arena_mask = np.zeros_like(gray)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        cv2.drawContours(arena_mask, [largest], -1, 255, thickness=cv2.FILLED)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ARENA_ERODE, ARENA_ERODE))
        arena_mask = cv2.erode(arena_mask, kernel, iterations=1)
    if cv2.countNonZero(arena_mask) == 0:
        arena_mask[:] = 255 # no arena found, dont throw out the whole frame
    return arena_mask


def get_dark_mask(gray, arena_mask):
    """
    Black-hat = closing(gray) - gray. The closing fills in anything dark that is smaller than DARK_KERNEL,
    so (closing - gray) is big on flies and ~0 on flat bg. Thin wall lines are cut off by the arena mask.
    """
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (DARK_KERNEL, DARK_KERNEL)) # rect is way faster than ellipse here
    blackhat = cv2.morphologyEx(blur, cv2.MORPH_BLACKHAT, kernel)
    _, dark_mask = cv2.threshold(blackhat, DARK_THRESH, 255, cv2.THRESH_BINARY)
    dark_mask = cv2.bitwise_and(dark_mask, arena_mask)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_OPEN, k, iterations=1)
    return dark_mask


def get_orb_keypoints(gray, search_mask):
    """ORB keypoints (and descriptors) inside search_mask only."""
    kp = orb.detect(gray, search_mask)
    kp, des = orb.compute(gray, kp)
    return kp, des


def filter_lone_keypoints(kp):
    """Real flies have a bunch of kps close together; a single kp out in the open is just noise."""
    if KP_MIN_NEIGHBORS <= 0 or len(kp) == 0:
        return kp
    pts = np.array([k.pt for k in kp], dtype=np.float32)
    keep = []
    for i, k in enumerate(kp):
        d = np.sqrt(((pts - pts[i]) ** 2).sum(axis=1))
        n_neighbors = int((d < KP_CLOSE_KERNEL).sum()) - 1 # minus itself
        if n_neighbors >= KP_MIN_NEIGHBORS:
            keep.append(k)
    return keep


def keypoints_to_blobs(kp, shape):
    """Stamp every keypoint as a filled circle then close, so a fly's kps become one solid blob."""
    kp_mask = np.zeros(shape, dtype=np.uint8)
    for k in kp:
        cv2.circle(kp_mask, (int(k.pt[0]), int(k.pt[1])), KP_RADIUS, 255, -1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (KP_CLOSE_KERNEL, KP_CLOSE_KERNEL))
    kp_mask = cv2.morphologyEx(kp_mask, cv2.MORPH_CLOSE, kernel, iterations=KP_CLOSE_ITERS)
    return kp_mask


# fg mask

def get_fg_mask(frame, name):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    arena_mask = get_arena_mask(gray)

    dark_mask = get_dark_mask(gray, arena_mask)
    search_mask = arena_mask
    if USE_DARK_GATE:
        gate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (DARK_GATE_DILATE, DARK_GATE_DILATE))
        search_mask = cv2.bitwise_and(arena_mask, cv2.dilate(dark_mask, gate, iterations=1))

    kp, des = get_orb_keypoints(gray, search_mask)
    kp = filter_lone_keypoints(kp)
    kp_blobs = keypoints_to_blobs(kp, gray.shape)

    fly_mask = kp_blobs
    if USE_DARK_GATE:
        # keep only the part of each blob that is actually darker than bg (grown a bit so the fly body isnt shaved)
        grow = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fly_mask = cv2.bitwise_and(fly_mask, cv2.dilate(dark_mask, grow, iterations=1))

    fly_mask = cv2.bitwise_and(fly_mask, arena_mask)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    fly_mask = cv2.morphologyEx(fly_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    fly_mask = cv2.morphologyEx(fly_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    return fly_mask, arena_mask, kp, kp_blobs, dark_mask


# watershed

def get_seeds(fg_mask):
    """
    Shrink each blob down to its fat middle(s) with a distance transform.
    Two flies stuck together have a thin neck, so they end up with two seeds instead of one.
    """
    if not SPLIT_TOUCHING:
        return fg_mask
    dist = cv2.distanceTransform(fg_mask, cv2.DIST_L2, 5)
    local_max = cv2.dilate(dist, np.ones((SEED_WINDOW, SEED_WINDOW), np.uint8))
    split = ((dist >= SEED_FRAC * local_max) & (dist >= 2))

    # small blobs stay whole, only big (probably merged) blobs get the split
    n, labels, stats, _ = cv2.connectedComponentsWithStats(fg_mask)
    seeds = fg_mask.copy()
    for lab in range(1, n):
        if stats[lab, cv2.CC_STAT_AREA] >= SPLIT_MIN_AREA:
            comp = labels == lab
            seeds[comp] = split[comp].astype(np.uint8) * 255
    return seeds


def apply_watershed_segmentation(sure_fg, sure_bg, original_frame):
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    ret, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    if len(original_frame.shape) == 2:
        original_frame_3ch = cv2.cvtColor(original_frame, cv2.COLOR_GRAY2BGR)
    else:
        original_frame_3ch = original_frame.copy()

    markers = cv2.watershed(original_frame_3ch, markers)
    contours_list = []
    for label in np.unique(markers):
        if label <= 1: # -1 = watershed lines, 1 = background
            continue
        target = np.where(markers == label, 255, 0).astype(np.uint8)
        cnts, _ = cv2.findContours(target, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            contours_list.append(cnts[0])
    return contours_list


# CONTOURS

def get_good_cnts(contours, frame, arena_mask, verbose=False):
    """
    Filter watershed contours to acc flies.
    verbose=True prints why each contour was kept/rejected (blue box = area, orange = outside arena, red = circularity)
    """
    large_contours = []
    disp_frm = frame.copy()

    for cnt in contours:
        cnt_ar = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        cx, cy = x + w // 2, y + h // 2
        circularity = (4 * 3.14159 * cnt_ar) / (perimeter ** 2)

        if cnt_ar < MIN_CONTOUR_AREA or cnt_ar > MAX_CONTOUR_AREA:
            cv2.rectangle(disp_frm, (x, y), (x + w, y + h), (255, 120, 0), 3)
            if verbose:
                print(f"  [REJECT area] ({cx},{cy}) area={round(cnt_ar)} (allowed {MIN_CONTOUR_AREA}-{MAX_CONTOUR_AREA}) circ={round(circularity, 2)}")
            continue

        if arena_mask[cy, cx] == 0:
            cv2.rectangle(disp_frm, (x, y), (x + w, y + h), (0, 165, 255), 3)
            if verbose:
                print(f"  [REJECT arena] ({cx},{cy}) area={round(cnt_ar)} circ={round(circularity, 2)}")
            continue

        if circularity > CIRC_MAX or circularity < CIRC_MIN:
            cv2.rectangle(disp_frm, (x, y), (x + w, y + h), (0, 0, 200), 3)
            if verbose:
                print(f"  [REJECT circ] ({cx},{cy}) area={round(cnt_ar)} circ={round(circularity, 2)} (allowed {CIRC_MIN}-{CIRC_MAX})")
            continue

        if verbose:
            print(f"  [KEEP] ({cx},{cy}) area={round(cnt_ar)} circ={round(circularity, 2)}")
        cv2.rectangle(disp_frm, (x, y), (x + w, y + h), (0, 200, 0), 3)
        cv2.putText(disp_frm, f'{round(circularity, 2)}', (x + 4, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        large_contours.append(cnt)

    return large_contours, disp_frm


def segment_frame(frame, name, verbose=False):
    """Full ORB -> blobs -> watershed -> filtered contours for one frame. Returns everything the debug views need."""
    fg_mask, bg_mask, kp, kp_blobs, dark_mask = get_fg_mask(frame, name)
    seeds = get_seeds(fg_mask)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (WS_BG_DILATE, WS_BG_DILATE))
    sure_bg = cv2.bitwise_and(cv2.dilate(fg_mask, kernel, iterations=1), bg_mask)
    watershed_cnts = apply_watershed_segmentation(seeds, sure_bg, frame)
    large_contours, disp_frm = get_good_cnts(watershed_cnts, frame, bg_mask, verbose)
    return large_contours, disp_frm, fg_mask, bg_mask, kp, kp_blobs, dark_mask


def show_preview(frame, kp, kp_blobs, fg_mask, disp_frm, dark_mask, name):
    """darkness gate -> What ORB sees -> kp blobs -> fg mask -> final fly boxes"""
    kp_img = cv2.drawKeypoints(frame, kp, None, color=(0, 255, 0), flags=0)
    fig, axes = plt.subplots(1, 5, figsize=(25, 5))
    for ax, im, title in zip(
        axes,
        [dark_mask, kp_img, kp_blobs, fg_mask, disp_frm],
        ['darker-than-bg mask', f'ORB keypoints ({len(kp)})', 'keypoints merged into blobs', 'fg mask', 'fly contours (green=kept)'],
    ):
        if im.ndim == 2:
            ax.imshow(im, cmap='gray')
        else:
            ax.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        ax.set_title(title)
        ax.axis('off')
    fig.suptitle(name)
    plt.tight_layout()
    plt.show()


# MAIN LOOP

BASE_PATH = "/Volumes/Crucial X9/Downloads/PupalHeight" # "/Volumes/Crucial X9/Cameras-Calit2IRT/BlueVideo_Detection/TestingSet"
RESULTS_LOC = "/Volumes/Crucial X9/Cameras-Calit2IRT/PupalHeight_Detection/TestingOutput" # "/Volumes/Crucial X9/Cameras-Calit2IRT/BlueVideo_Detection/TestingOutput"
DEBUG_LOC = "/Volumes/Crucial X9/Cameras-Calit2IRT/PupalHeight_Detection/TestingDebug"

print(os.listdir(BASE_PATH))

for vid_name in os.listdir(BASE_PATH):
    skip_list = {'fb1.mov', 'fb2.mov'}

    if vid_name[0] == '.' or vid_name in skip_list:
        continue

    vid_path = f"{BASE_PATH}/{vid_name}"
    cap = cv2.VideoCapture(vid_path)
    name = vid_path.split('/')[-1]

    csv_name = f"{RESULTS_LOC}/{name}_orb_watershed_velocity_hungarian_pathing.csv"

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    output_path = f"{RESULTS_LOC}/{name}_orb_watershed_velocity_hungarian_pathing.mp4"

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    # TRACKING DS
    frame_count = 0
    next_object_id = 0
    tracked_objects = {}# id -> bbox
    lost_objects = {} # id -> {bbox, frames_lost, ghost_center}
    object_paths = {} # id -> deque of (cx,cy)
    object_lifetimes = {} # id -> int
    colors = {}

    object_velocities = {} # id -> (vx, vy)  EMA
    last_centers = {} # id -> (cx, cy)

    tracking_data = [] # for csv writing, not other use
    CURRENT_TOTAL_FLIES = 0

    if not cap.isOpened():
        print(f"Error opening {vid_path}")
        continue

    # FRAME
    ret, frame = cap.read()
    if not ret:
        cap.release(); out.release(); continue

    print(f"{name}: 1st frame contour report")
    large_contours, disp_frm, fg_mask, bg_mask, kp, kp_blobs, dark_mask = segment_frame(frame, name, verbose=True)

    if SHOW_PREVIEW:
        show_preview(frame, kp, kp_blobs, fg_mask, disp_frm, dark_mask, name)

    for cnt in large_contours:
        bbox = cv2.boundingRect(cnt)
        tracked_objects[next_object_id] = bbox
        object_lifetimes[next_object_id] = 1
        cx, cy = get_center(bbox)
        object_paths[next_object_id] = deque([(cx, cy)], maxlen=MAX_PATH_LENGTH)
        object_velocities[next_object_id] = (0.0, 0.0)
        last_centers[next_object_id] = (cx, cy)
        next_object_id += 1

    frame_data = {'frame': 0}
    for obj_id, bbox in tracked_objects.items():
        cx, cy = get_center(bbox)
        frame_data[f'ID_{obj_id}'] = f'({cx}!{cy})'
    tracking_data.append(frame_data)

    print(f"Starting tracking with {len(tracked_objects)} initial flies detected")
    frame_count = 1

    # MAIN FRAME2
    while cap.isOpened():
        ret, frame2 = cap.read()
        if not ret or frame_count >= fps * STOP_SEC:
            break
        frame_count += 1

        large_contours, disp_frm, fg_mask, bg_mask, kp, kp_blobs, dark_mask = segment_frame(frame2, name)

        if DEBUG:
            # kps drawn on the frame so you can see exactly what ORB is picking up
            kp_img = cv2.drawKeypoints(frame2, kp, None, color=(0, 255, 0), flags=0)
            cv2.imwrite(f"{DEBUG_LOC}/{name}_orb_keypoints.png", kp_img)
            cv2.imwrite(f"{DEBUG_LOC}/{name}_orb_kp_blobs.png", kp_blobs)
            cv2.imwrite(f"{DEBUG_LOC}/{name}_dark_mask.png", dark_mask)
            cv2.imwrite(f"{DEBUG_LOC}/{name}_fg_mask_orb_watershed_velocity_hungarian_pathing.png", fg_mask)
            cv2.imwrite(f"{DEBUG_LOC}/{name}_bg_mask_orb_watershed_velocity_hungarian_pathing.png", bg_mask)
            cv2.imwrite(f"{DEBUG_LOC}/{name}_disp_frm_orb_watershed_velocity_hungarian_pathing.png", disp_frm)
        current_bboxes = [cv2.boundingRect(cnt) for cnt in large_contours]

        new_tracked_objects = {}
        used_current = set()
        matched_tracked_ids = set()

        # 0) move ghost positions for all lost flies so that if they moved around swhere else it ok
        for obj_id in list(lost_objects.keys()):
            advance_ghost(obj_id)

        # 1) Hungarian matching
        tracked_ids = list(tracked_objects.keys())

        if tracked_ids and current_bboxes:
            n_tracked = len(tracked_ids)
            n_det = len(current_bboxes)
            cost = np.full((n_tracked, n_det), 1e6, dtype=float)

            for r, obj_id in enumerate(tracked_ids):
                prev_bbox = tracked_objects[obj_id]
                pred_center = predict_position(prev_bbox, obj_id)
                for c, curr_bbox in enumerate(current_bboxes):
                    dist_pred = predicted_distance(pred_center, curr_bbox)
                    dist_direct = calculate_distance(prev_bbox, curr_bbox)
                    if dist_pred < FLIGHT_DISTANCE_THRESHOLD or dist_direct < DISTANCE_THRESHOLD:
                        cost[r, c] = dist_pred + size_penalty(prev_bbox, curr_bbox)

            row_ind, col_ind = linear_sum_assignment(cost)

            for r, c in zip(row_ind, col_ind):
                if cost[r, c] >= 1e6:
                    continue

                obj_id = tracked_ids[r]
                matched_bbox = current_bboxes[c]

# ID-swap fix where we reject the match if another *currently tracked* fly's last position is closer to the detection than obj_id's prediction.
                matched_cx, matched_cy = get_center(matched_bbox)
                pred_cx, pred_cy = predict_position(tracked_objects[obj_id], obj_id)
                dist_to_pred = np.sqrt((matched_cx - pred_cx)**2 + (matched_cy - pred_cy)**2)

                stolen = False
                for other_id, other_bbox in tracked_objects.items():
                    if other_id == obj_id:
                        continue
                    other_cx, other_cy = get_center(other_bbox)
                    dist_other = np.sqrt((matched_cx - other_cx)**2 + (matched_cy - other_cy)**2)
# If another fly's *current* position is closer and obj_id is predicting a large jump, suspect a swap
                    if dist_other < dist_to_pred * 0.5 and dist_to_pred > DISTANCE_THRESHOLD:
                        stolen = True
                        break

                if stolen:
                    # NEW FLY b/c not assigned
                    continue

                new_tracked_objects[obj_id] = matched_bbox
                used_current.add(c)
                matched_tracked_ids.add(obj_id)

                object_lifetimes[obj_id] += 1
                cx, cy = get_center(matched_bbox)
                object_paths[obj_id].append((cx, cy))
                update_velocity(obj_id, (cx, cy))

        # unmatched tracked flies are lost
        for obj_id in tracked_ids:
            if obj_id not in matched_tracked_ids:
                if obj_id not in lost_objects:
                    cx, cy = get_center(tracked_objects[obj_id])
                    lost_objects[obj_id] = {
                        'bbox': tracked_objects[obj_id],
                        'frames_lost': 1,
                        'ghost_center': (float(cx), float(cy)),
                    }
                else:
                    lost_objects[obj_id]['frames_lost'] += 1

        # No tracked objects means that everything is lost
        if not tracked_ids:
            for obj_id in list(tracked_objects.keys()):
                if obj_id not in lost_objects:
                    cx, cy = get_center(tracked_objects[obj_id])
                    lost_objects[obj_id] = {
                        'bbox': tracked_objects[obj_id],
                        'frames_lost': 1,
                        'ghost_center': (float(cx), float(cy)),
                    }
                else:
                    lost_objects[obj_id]['frames_lost'] += 1

        # 2) recovery w the ghost pos now
        lost_to_remove = []
        for i, curr_bbox in enumerate(current_bboxes):
            if i in used_current:
                continue

            best_lost_id = -1
            best_lost_dist = float('inf')

            for obj_id, lost_data in lost_objects.items():
                if lost_data['frames_lost'] > MAX_LOST_FRAMES:
                    continue
                if obj_id in new_tracked_objects:
                    continue
                if obj_id in lost_to_remove: # already claimed by another detection this frame
                    continue

                # use ghost position for recovery search here now instead of just static things
                gc = ghost_center(obj_id)
                if gc is not None:
                    det_cx, det_cy = get_center(curr_bbox)
                    dist = np.sqrt((gc[0] - det_cx)**2 + (gc[1] - det_cy)**2)
                else:
                    dist = calculate_distance(lost_data['bbox'], curr_bbox)

                if dist < best_lost_dist:
                    best_lost_dist = dist
                    best_lost_id = obj_id

            # recovery radius is FLIGHT_DISTANCE_THRESHOLD for previously fast flies, else RECOVERY_THRESHOLD w velocity pred
            if best_lost_id != -1:
                fly_was_fast = (np.sqrt(sum(v**2 for v in get_velocity(best_lost_id))) > FLYING_SPEED_THRESHOLD)
                radius = FLIGHT_DISTANCE_THRESHOLD if fly_was_fast else RECOVERY_THRESHOLD

                if best_lost_dist < radius:
                    new_tracked_objects[best_lost_id] = curr_bbox
                    used_current.add(i)
                    object_lifetimes[best_lost_id] += 1
                    cx, cy = get_center(curr_bbox)
                    object_paths[best_lost_id].append((cx, cy))
                    update_velocity(best_lost_id, (cx, cy))
                    lost_to_remove.append(best_lost_id)
                    print(f"  [RECOVERY] ID {best_lost_id} at dist {round(best_lost_dist, 1)}px after {lost_objects[best_lost_id]['frames_lost']} lost frames{'  [FLIGHT]' if fly_was_fast else ''}")
                    continue

            # truly a new fly if the ghost doesn't work and it wasn't recovery with static lost either
            new_tracked_objects[next_object_id] = curr_bbox
            object_lifetimes[next_object_id] = 1
            cx, cy = get_center(curr_bbox)
            object_paths[next_object_id] = deque([(cx, cy)], maxlen=MAX_PATH_LENGTH)
            object_velocities[next_object_id] = (0.0, 0.0)
            last_centers[next_object_id] = (cx, cy)
            print(f"  [NEW FLY] ID {next_object_id} at ({cx}, {cy})")
            next_object_id += 1

        # 3) Clean up lost buffer to avoid swaps from old things
        for obj_id in lost_to_remove:
            del lost_objects[obj_id]

        expired = [oid for oid, d in lost_objects.items() if d['frames_lost'] > MAX_LOST_FRAMES]
        for obj_id in expired:
            del lost_objects[obj_id]

        tracked_objects = new_tracked_objects

        # data saving
        frame_data = {'frame': frame_count}
        for obj_id, bbox in tracked_objects.items():
            if object_lifetimes.get(obj_id, 0) >= MIN_LIFETIME:
                cx, cy = get_center(bbox)
                frame_data[f'ID_{obj_id}'] = f'({cx}!{cy})'
        tracking_data.append(frame_data)

        CURRENT_TOTAL_FLIES = len(tracked_objects)

        # drawing
        for obj_id, bbox in tracked_objects.items():
            if object_lifetimes.get(obj_id, 0) >= MIN_LIFETIME:
                draw_paths(frame2, object_paths, obj_id)
                x, y, w, h = bbox
                color = get_unique_color(obj_id)
                cv2.rectangle(frame2, (x, y), (x + w, y + h), color, 3)
                label = f'ID:{obj_id} [FLY]' if is_flying(obj_id) else f'ID:{obj_id}'
                cv2.putText(frame2, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 2)

        # draw ghost positions for lost flies (both static lost and with velocity pred)
        if DEBUG:
            for obj_id, data in lost_objects.items():
                gc = ghost_center(obj_id)
                if gc is not None:
                    color = get_unique_color(obj_id)
                    cv2.circle(frame2, (int(gc[0]), int(gc[1])), 8, color, 2)
                    cv2.putText(frame2, f'L:{obj_id}', (int(gc[0]) + 10, int(gc[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 1)

        valid_flies = sum(1 for oid in tracked_objects if object_lifetimes.get(oid, 0) >= MIN_LIFETIME)
        cv2.putText(frame2, f'TOTAL FLIES: {valid_flies}', (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 2)

        out.write(frame2)

        if frame_count % fps == 0:
            print(f"{name} @ frame {frame_count}| {frame_count//fps} sec: valid={valid_flies},tracked={len(tracked_objects)},lost={len(lost_objects)}")

    # write to the CSV
    if tracking_data:
        all_fly_ids = set()
        for fd in tracking_data:
            all_fly_ids.update(k for k in fd if k.startswith('ID_'))
        sorted_fly_ids = sorted(all_fly_ids, key=lambda x: int(x.split('_')[1]))

        with open(csv_name, 'w', newline='') as csvfile:
            fieldnames = ['frame'] + sorted_fly_ids
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for fd in tracking_data:
                row = {'frame': fd['frame']}
                for fid in sorted_fly_ids:
                    row[fid] = fd.get(fid, '')
                writer.writerow(row)

        print(f"Total unique fly IDs tracked: {len(sorted_fly_ids)}")

    cap.release()
    out.release()
    cv2.destroyAllWindows()
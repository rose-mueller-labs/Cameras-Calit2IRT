# Cameras Neuroassay

Tools for detecting and uniquely tracking individual fruit flies in top-down videos of a box.

## Test Set (Blue)

We have a test set consisting of videos of flies in a box surrounded by bright, blue borders. There are five videos (`CO1d78.mov` to `CO5d78.mov`). Each was processed for 7,140 frames, which is 60 seconds at 119 fps.

## Algorithms

All of the algorithms follow the same overall idea but have different ways of doing it. We are trying to make a scritp that can detect the flies in each frame, then track them so that every fly keeps one unique ID over time.

| Algorithm | Pros | Cons |
|---|---|---|
| **Kalman filter** (`kalman_filter.py`) | Estimates each fly's position and velocity, then corrects the prediction with every new detection.<br>Detected more flies than thresholding on CO3d78, CO4d78 and CO5d78. | Very badly fragmented: about 1,900 IDs per video, each lasting about a quarter of a second.<br>Never averaged more than 9.71 flies per frame out of 20, so many flies are missed. |
| **High pass filter** (`high_pass_filter.py`, labeled `fft`) | Gave by far the most stable tracking: fewest IDs (51.8 on average) and the longest-lived ones (about 24.7 s).<br>Finds 15 to 20 flies per frame in all five videos.<br>Removes the slowly varying background, so no hand-picked brightness threshold is needed. | Still produces 2 to 4 times more IDs than the 20 real flies, so some tracks break.<br>IDs last only about 14 s in CO3d78 and CO5d78.<br>Averages 18.2 of 20 flies per frame, so some are still missed. |
| **Thresholding** (`watershed_velocity_hungarian_pathing.py`) | Simple method: thresholding plus watershed, which also splits flies that touch.<br>Velocity prediction and Hungarian matching handle fast movement and flight.<br>Ghost tracks let a fly regain its ID when it reappears.<br>Worked reasonably on CO1d78 and CO2d78. | Needs a hand-picked brightness threshold.<br>Detects almost no flies in CO3d78, CO4d78 (0.01 per frame) and CO5d78.<br>Its low ID counts in those videos come from missed detections, not good tracking.<br>Only 4.59 flies per frame on average. |
| **ORB** (`orb_watershed_velocity_hungarian_pathing.py`) | Built for smaller, lighter flies that the other methods struggle with.<br>The darkness gate and black-hat transform limit ORB's search to small dark spots.<br>Merging keypoints into blobs gives one detection per fly instead of dotty points.<br>The wall zone finds flies along the walls by subtracting a static background, which cancels smudges, shadows and scratches. | Not evaluated on the Blue test set, so there are no results to compare with the others.<br>Designed for non-Blue, small-fly videos, so it isn't a fit for the Blue set.<br>Has the most components to tune: the darkness gate, ORB keypoints, blob closing and the wall zone. |

### `watershed_velocity_hungarian_pathing.py`

With these videos, I made an algorithm that uses simple thresholding and the watershed algorithm to uncover the fruit flies. Tracking is done with velocity prediction and Hungarian matching:

- Thresholding + watershed separates the flies from the background and splits flies that touch.
- Velocity (an exponential moving average of each fly's movement) predicts where each fly will be in the next frame, including during fast movement or flight.
- Hungarian matching assigns detections to existing IDs by minimizing the total distance between predicted positions and detections.

Flies that disappear are kept as "ghosts" that keep moving at their last velocity, so they can be recovered with the same ID when they reappear.

### `kalman_filter.py`

I then made a Kalman filter algorithm. Instead of a simple moving-average velocity, each fly's state (position and velocity) is estimated by a Kalman filter, which predicts where the fly will be and corrects that prediction with each new detection.

### `high_pass_filter.py`

I then made a high pass filter in place of thresholding. Flies are small, sharp features, so filtering out the slowly varying background leaves the flies standing out without needing a hand-picked brightness threshold. This algorithm is labeled `fft` after the Fast Fourier Transform in the results below.

### `orb_watershed_velocity_hungarian_pathing.py`

The most recent algorithm replaces thresholding with ORB keypoints. The idea behind this is to tackle detection on flies that are smaller and lighter than the C-types. The algorithm uses:

- A "darkness gate" where flies that are a little darker than their local background are counted automatically. So a black-hat transform finds small dark spots and ORB only searches near them instead.
- I get the ORB keypoints. In there, FAST corners and descriptors are detected on each frame. On a fly they land on the body outline, legs, and wings.
- Blobs are keypoints drawn as filled circles and morphologically closed, so the keypoints on one fly merge into a single blob instead of fuzzy, dotty points.
- I added a wall zone, which is an idea from previous iterations of the algorithm before the Blue edges existed. Flies walking along the walls are found by subtracting a static background (a bright percentile of frames sampled across the clip). The walls never move but the flies do, so smudges, shadows, and scratches along the walls cancel out.

It is not features in the results section because it is specifically for non-Blue, small fly videos.

## Results

Here are the results in a tabular format from the five Blue videos (Each video has 20 flies):

| video | algorithm | frames_processed | unique_ids | avg_id_length_frames | avg_id_length_sec | avg_flies_per_frame |
|---|---|---:|---:|---:|---:|---:|
| CO1d78.mov | watershed | 7140 | 159 | 470.69 | 3.955 | 10.48 |
| CO1d78.mov | fft | 7140 | 40 | 3172.9 | 26.663 | 17.78 |
| CO1d78.mov | kalman | 7140 | 3119 | 22.22 | 0.187 | 9.71 |
| CO2d78.mov | watershed | 7140 | 384 | 167.9 | 1.411 | 9.03 |
| CO2d78.mov | fft | 7140 | 40 | 3460.43 | 29.079 | 19.39 |
| CO2d78.mov | kalman | 7140 | 1481 | 32.96 | 0.277 | 6.84 |
| CO3d78.mov | watershed | 7140 | 42 | 239.29 | 2.011 | 1.41 |
| CO3d78.mov | fft | 7140 | 65 | 1661.62 | 13.963 | 15.13 |
| CO3d78.mov | kalman | 7140 | 2348 | 25.75 | 0.216 | 8.47 |
| CO4d78.mov | watershed | 7140 | 7 | 10.0 | 0.084 | 0.01 |
| CO4d78.mov | fft | 7140 | 29 | 4805.86 | 40.385 | 19.52 |
| CO4d78.mov | kalman | 7140 | 1215 | 30.03 | 0.252 | 5.11 |
| CO5d78.mov | watershed | 7140 | 136 | 104.79 | 0.881 | 2.0 |
| CO5d78.mov | fft | 7140 | 85 | 1615.21 | 13.573 | 19.23 |
| CO5d78.mov | kalman | 7140 | 1376 | 45.05 | 0.379 | 8.68 |
| **ALL_AVG** | **watershed** | 7140.0 | 145.6 | 198.53 | 1.668 | 4.59 |
| **ALL_AVG** | **fft** | 7140.0 | 51.8 | 2943.2 | 24.733 | 18.21 |
| **ALL_AVG** | **kalman** | 7140.0 | 1907.8 | 31.2 | 0.262 | 7.76 |

- `unique_ids`: how many different IDs were assigned over the clip. With a roughly fixed number of flies in the box, fewer is better, since extra IDs mean a fly's track was broken and restarted.
- `avg_id_length_frames` / `avg_id_length_sec`: how long an average ID survived (at 119 fps). Longer means more continuous tracking.
- `avg_flies_per_frame`: the average number of flies detected in each frame.

### Discussion

- **High pass filter (`fft`)** gave by far the most stable tracking! It had the fewest IDs on average (51.8) and the longest-lived ones (about 24.7 s), and it found a consistent 15 to 20 flies per frame across all five videos.
- **Kalman** was very badly fragmented. It detected 1,900 IDs per video on average, each lasting only about a quarter of a second.
- **Watershed with thresholding** worked reasonably on `CO1d78` and `CO2d78` but detected almost no flies in `CO3d78`, `CO4d78` (0.01 per frame), and `CO5d78`. Its low ID counts in those videos reflect missed detections rather than good tracking.
- These numbers measure tracking *continuity*, not accuracy.

The five videos' results can be viewed [here](https://drive.proton.me/urls/Z5SHP3RHJC#4f84hwtt9fM6).
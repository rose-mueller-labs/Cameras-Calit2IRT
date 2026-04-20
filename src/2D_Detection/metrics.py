'''
Metrics for how well the algorithm is performing based on how many flies are actually in the video 
and how many are detected in each frame. 

Metrics used: MSE, RMSE, and R^2 value of subsequent line produced. An R^2 value close to 0 is best since
the number of flies in the video remains constant and does not change.
'''

from sklearn.metrics import r2_score, mean_squared_error, root_mean_squared_error
import pandas as pd
from csv import DictWriter


def get_total_flies_per_frame(df: pd.DataFrame) -> pd.Series:
    """
    Count the number of detected flies per frame by counting non-empty ID columns.
    Each non-empty cell in an ID column represents a detected fly at that (x, y) coordinate.
    """
    id_cols = [col for col in df.columns if col.startswith("ID_")]
    return df[id_cols].notna().sum(axis=1)

def get_total_unique_flies(df: pd.DataFrame) -> int:
    shape = df.shape
    ids = shape[1]-1
    return ids

if __name__ == '__main__':
    video_test_csv_path = input("What is the video test csv output's path (e.g. ./WatershedAlgorithm/Output/Pathing/Tracked_2k 120fps backlit.MXF_pws.csv): ")
    name = video_test_csv_path.split("/")[-1].split('.csv')[0].split('.')[0].split('_')[-1]
    print(name)
    video_test_name = f"{video_test_csv_path.split("/")[-1].split('.csv')[0].split('.')[0].split('_')[-1]}.MOV"
    alg_used = video_test_csv_path.split("/")[-1].split('.csv')[0].split('.')[-1].split("_")[-1]

    UPDATE_CSV_PATH=f"./2D_Detection/performance_{alg_used}.csv"
    
    fly_cnt_df = pd.read_csv("/Volumes/Crucial X9/Cameras-Calit2IRT/src/SampleVideos/fly_cnt.csv")
    df = pd.read_csv(video_test_csv_path)
    detected_per_frame = get_total_flies_per_frame(df)
    TRUE_FLY_COUNT = fly_cnt_df.loc[fly_cnt_df['VideoPath'] == video_test_name, 'FlyCount'].values[0]
    total_unique = get_total_unique_flies(df)
 
    y_true = [TRUE_FLY_COUNT] * len(detected_per_frame)
    y_pred = detected_per_frame.tolist()
 
    mse = mean_squared_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
 
    update_fields_dict = {'VideoName': video_test_name, 'FramesAnalyzed': len(detected_per_frame), 'TrueFlyCount': TRUE_FLY_COUNT, 'MeanDetected': detected_per_frame.mean(),
                          'MinDetectedFrame': detected_per_frame.min(), 'MaxDetectedFrame': detected_per_frame.max(), 'TotalUniqueFlies': total_unique,
                          'MSE': mse, 'RMSE': rmse, 'R2': r2}
    
    with open(UPDATE_CSV_PATH, 'a', newline='') as f:
        writer = DictWriter(f, fieldnames=list(update_fields_dict.keys()))
        writer.writerow(update_fields_dict)

 
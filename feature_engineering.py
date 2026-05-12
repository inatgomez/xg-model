import pandas as pd
import numpy as np

PARQUET_PATH = "data/la_liga_1516_events.parquet"
OUTPUT_PATH = "data/shots_features.parquet"

GOAL_CENTER = [120, 40]
GOAL_POSTS = [[120, 36], [120, 44]]

df = pd.read_parquet(PARQUET_PATH)
shots = df[df["type"] == "Shot"].copy()

shots["is_goal"] = (shots["shot_outcome"] == "Goal").astype(int)

def extract_location(loc):
    if isinstance(loc, np.ndarray)and len(loc) == 2:
        return loc[0], loc[1]
    return np.nan, np.nan

shots[["x", "y"]] = pd.DataFrame(
    shots["location"].apply(extract_location).tolist(),
    index=shots.index
)

shots["distance"] = np.sqrt(
    (shots["x"] - GOAL_CENTER[0]) ** 2 +
    (shots["y"] - GOAL_CENTER[1]) ** 2
)

def shot_angle(x, y):
    post_left = GOAL_POSTS[0]
    post_right = GOAL_POSTS[1]

    a = np.array([x, y])
    b = np.array(post_left)
    c = np.array(post_right)

    ab = b - a
    ac = c - a

    cos_angle = np.dot(ab, ac) / (np.linalg.norm(ab) * np.linalg.norm(ac) + 1e-9)
    angle_rad = np.arccos(np.clip(cos_angle, -1, 1))
    return np.degrees(angle_rad)

shots["angle"] = shots.apply(
    lambda row: shot_angle(row["x"], row["y"]), axis=1
)

shots["is_header"] = (shots["shot_body_part"] == "Head").astype(int)
shots["is_open_play"] = (shots["shot_type"] == "Open Play").astype(int)

FEATURE_COLS = [
    "id", "match_id", "player", "team",
    "x", "y", "distance", "angle",
    "is_header", "is_open_play",
    "is_goal", "shot_statsbomb_xg"
]

available = [c for c in FEATURE_COLS if c in shots.columns]
shots_out = shots[available].copy()

null_counts = shots_out.isnull().sum()

assert shots_out["is_goal"].isin([0, 1]).all(), "Outcome column has unexpected values"
assert shots_out["distance"].between(0, 120).all(), "Distance out of range"

shots_out.to_parquet(OUTPUT_PATH, index=False)
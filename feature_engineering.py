import pandas as pd
import numpy as np

PARQUET_PATH = "data/la_liga_1516_events.parquet"
OUTPUT_PATH = "data/shots_features.parquet"
OUTPUT_PATH_V2 = "data/shots_features_v2.parquet"

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

# The following are the features for the second version of the model.

shots["is_penalty"] = (shots["shot_type"] == "Penalty").astype(int)

def goalkeeper_coverage(freeze_frame, shooter_x, shooter_y):
    if freeze_frame is None or len(freeze_frame) == 0:
        return np.nan

    gk = next(
        (p for p in freeze_frame
         if isinstance(p.get("position"), dict)
         and p["position"].get("name") == "Goalkeeper"
         and p.get("teammate") is False),
        None
    )
    if gk is None:
        return np.nan

    loc = gk.get("location")
    if loc is None or len(loc) != 2:
        return np.nan

    gk_pos = np.array(loc, dtype=float)
    shooter = np.array([shooter_x, shooter_y], dtype=float)
    goal = np.array(GOAL_CENTER, dtype=float)

    line_vec = goal - shooter
    line_len = np.linalg.norm(line_vec)
    if line_len < 1e-9:
        return np.nan

    t = np.dot(gk_pos - shooter, line_vec) / (line_len ** 2)
    t_clamped = np.clip(t, 0, 1)
    closest_point = shooter + t_clamped * line_vec
    return float(np.linalg.norm(gk_pos - closest_point))

shots["goalkeeper_coverage"] = shots.apply(
    lambda row: goalkeeper_coverage(row["shot_freeze_frame"], row["x"], row["y"]),
    axis=1
)

def point_in_cone(px, py, shooter_x, shooter_y):
    """
    Returns True if point (px, py) is inside the triangle formed by
    shooter location and both goalposts.
    Uses cross-product sign to test same-side membership.
    """
    s = np.array([shooter_x, shooter_y], dtype=float)
    p_left = np.array(GOAL_POSTS[0], dtype=float)
    p_right = np.array(GOAL_POSTS[1], dtype=float)
    pt = np.array([px, py], dtype=float)

    def cross2d(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    d1 = cross2d(s, p_left, pt)
    d2 = cross2d(p_left, p_right, pt)
    d3 = cross2d(p_right, s, pt)

    has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
    has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
    return not (has_neg and has_pos)


def defenders_in_lane(freeze_frame, shooter_x, shooter_y):
    if freeze_frame is None or len(freeze_frame) == 0:
        return np.nan

    count = 0
    for p in freeze_frame:
        if p.get("teammate") is True:
            continue
        pos_name = ""
        if isinstance(p.get("position"), dict):
            pos_name = p["position"].get("name", "")
        if pos_name == "Goalkeeper":
            continue
        loc = p.get("location")
        if loc is None or len(loc) != 2:
            continue
        if point_in_cone(loc[0], loc[1], shooter_x, shooter_y):
            count += 1

    return float(count)

shots["defenders_in_lane"] = shots.apply(
    lambda row: defenders_in_lane(row["shot_freeze_frame"], row["x"], row["y"]),
    axis=1
)

FEATURE_COLS_V2 = FEATURE_COLS + ["is_penalty", "goalkeeper_coverage", "defenders_in_lane"]
available_v2 = [c for c in FEATURE_COLS_V2 if c in shots.columns]
shots_out_v2 = shots[available_v2].copy()

null_counts_v2 = shots_out_v2.isnull().sum()

assert shots_out_v2["is_penalty"].isin([0, 1]).all(), "is_penalty has unexpected values"
assert shots_out_v2["goalkeeper_coverage"].between(0, 120).all() or shots_out_v2["goalkeeper_coverage"].isnull().any(), \
    "goalkeeper_coverage out of range"
assert shots_out_v2["defenders_in_lane"].dropna().between(0, 11).all(), \
    "defenders_in_lane out of range"


shots_out_v2.to_parquet(OUTPUT_PATH_V2, index=False)
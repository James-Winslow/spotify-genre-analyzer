#!/usr/bin/env python3
import pandas as pd, pathlib as P, ast, sys

ROOT = P.Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"

def to_list(x):
    if isinstance(x, list): return x
    if pd.isna(x) or not isinstance(x, str): return []
    try:
        v = ast.literal_eval(x)
        return v if isinstance(v, list) else []
    except Exception:
        return []

def main():
    recent_p = PROC / "recently_played.csv"
    saved_p  = PROC / "saved_tracks.csv"
    genres_p = PROC / "artist_genres.csv"
    for p in (recent_p, saved_p, genres_p):
        if not p.exists():
            sys.exit(f"Missing {p}. Run fetch_spotify_data.py first.")

    recent = pd.read_csv(recent_p)
    saved  = pd.read_csv(saved_p)
    genres = pd.read_csv(genres_p)

    common = ["track_id","track_name","artist_id","artist_name","album_name"]
    for col in common:
        for name, df in (("recently_played", recent), ("saved_tracks", saved)):
            if col not in df.columns:
                sys.exit(f"{name} is missing column '{col}'")

    combo = pd.concat([recent[common], saved[common]], ignore_index=True).drop_duplicates()

    if genres.empty or "artist_id" not in genres.columns or "genres" not in genres.columns:
        sys.exit("artist_genres.csv is empty or missing required columns.")

    df = combo.merge(genres[["artist_id","artist_name","genres"]],
                     on=["artist_id","artist_name"], how="left")
    df["genres"] = df["genres"].apply(to_list)

    exploded = df.explode("genres").dropna(subset=["genres"])
    if exploded.empty:
        out = PROC / "genre_summary.csv"
        pd.DataFrame(columns=["genres","size"]).to_csv(out, index=False)
        print(f"✅ Wrote {out} (empty)"); return

    counts = (exploded.groupby("genres", as_index=False)
              .size().sort_values("size", ascending=False))

    out = PROC / "genre_summary.csv"
    counts.to_csv(out, index=False)
    print(f"✅ Wrote {out} with {len(counts)} rows")
    print(counts.head(20).to_string(index=False))

if __name__ == "__main__":
    main()

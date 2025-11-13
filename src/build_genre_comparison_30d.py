#!/usr/bin/env python3
import pandas as pd, pathlib as P, ast
from datetime import datetime, timedelta, timezone

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

def load_required():
    recent = pd.read_csv(PROC / "recently_played.csv", parse_dates=["played_at"])
    saved  = pd.read_csv(PROC / "saved_tracks.csv",    parse_dates=["added_at"])
    genres = pd.read_csv(PROC / "artist_genres.csv")
    genres["genres"] = genres["genres"].apply(to_list)
    return recent, saved, genres

def explode_with_genres(df_tracks, genres):
    cols = ["track_id","track_name","artist_id","artist_name","album_name"]
    base = df_tracks[cols].drop_duplicates()
    merged = base.merge(genres[["artist_id","artist_name","genres"]],
                        on=["artist_id","artist_name"], how="left")
    merged["genres"] = merged["genres"].apply(to_list)
    return merged.explode("genres").dropna(subset=["genres"])

def main():
    recent, saved, genres = load_required()

    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(days=30)
    recent_30 = recent[recent["played_at"] >= cutoff]

    exp_30  = explode_with_genres(recent_30, genres)
    exp_all = explode_with_genres(pd.concat([recent, saved], ignore_index=True), genres)

    if exp_all.empty:
        out = PROC / "genre_compare_30d.csv"
        pd.DataFrame(columns=["genre","size_30","pct_30","size_all","pct_all","lift"]).to_csv(out, index=False)
        print(f"✅ Wrote {out} (empty)"); return

    g30  = exp_30.groupby("genres", as_index=False).size().rename(columns={"genres":"genre","size":"size_30"})
    gall = exp_all.groupby("genres", as_index=False).size().rename(columns={"genres":"genre","size":"size_all"})

    comp = gall.merge(g30, on="genre", how="left")
    comp["size_30"] = comp["size_30"].fillna(0).astype(int)

    total_30  = comp["size_30"].sum()
    total_all = comp["size_all"].sum()
    comp["pct_30"]  = (comp["size_30"]  / total_30)  if total_30  > 0 else 0
    comp["pct_all"] = (comp["size_all"] / total_all) if total_all > 0 else 0
    comp["lift"]    = comp.apply(lambda r: (r["pct_30"]/r["pct_all"]) if r["pct_all"]>0 else None, axis=1)

    # round for readability
    comp["pct_30"]  = comp["pct_30"].round(4)
    comp["pct_all"] = comp["pct_all"].round(4)
    comp["lift"]    = comp["lift"].round(2)

    comp = comp.sort_values(["lift","size_30"], ascending=[False, False])
    comp = comp[["genre","size_all","size_30","pct_30","pct_all","lift"]]

    out = PROC / "genre_compare_30d.csv"
    comp.to_csv(out, index=False)
    print(f"✅ Wrote {out} with {len(comp)} rows")
    print(comp.head(15).to_string(index=False))
    print("\nColumns:", list(comp.columns))

if __name__ == "__main__":
    main()

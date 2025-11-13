# src/streamlit_app.py
import ast
from pathlib import Path
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Spotify Genre Analyzer", layout="wide")
PROC = Path("data/processed")

# ---------- helpers ----------
def load_csv(path: Path, parse_dates=None) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, parse_dates=parse_dates)

def friendly_cols(df: pd.DataFrame) -> pd.DataFrame:
    mapping = {
        "track_name": "Track Name",
        "artist_name": "Artist",
        "album_name": "Album",
        "played_at": "Played At",
        "added_at": "Added At",
        "time_range": "Time Range",
        "genre": "Genre",
        "size_all": "Artists in Library",
        "size_30": "Artists in Last 30 Days",
        "pct_30": "% of Last 30 Days",
        "pct_all": "% of Library",
        "lift": "Lift vs Library",
    }
    cols = {c: mapping[c] for c in df.columns if c in mapping}
    return df.rename(columns=cols)

def ensure_genre_list(df_gen: pd.DataFrame) -> pd.DataFrame:
    if df_gen.empty:
        return df_gen
    g = df_gen.copy()
    if "genres" in g.columns:
        def _to_list(x):
            if isinstance(x, list):
                return x
            if isinstance(x, str):
                try:
                    v = ast.literal_eval(x)
                    return v if isinstance(v, list) else []
                except Exception:
                    return [s.strip() for s in x.split(",")] if x else []
            return []
        g["genres"] = g["genres"].apply(_to_list)
    return g

def explode_genres(df_tracks: pd.DataFrame, df_genres: pd.DataFrame) -> pd.DataFrame:
    """Join by artist_id only; explode to genre rows."""
    if df_tracks.empty or df_genres.empty:
        return pd.DataFrame(columns=list(df_tracks.columns) + ["genre"])
    g = ensure_genre_list(df_genres)
    g = g.explode("genres").rename(columns={"genres": "genre"})
    # join on artist_id only; keep original artist_name from tracks
    return (
        df_tracks.merge(g[["artist_id", "genre"]], on="artist_id", how="left")
                 .dropna(subset=["genre"])
    )

# ---------- load cached ----------
recent   = load_csv(PROC / "recently_played.csv", parse_dates=["played_at"])
saved    = load_csv(PROC / "saved_tracks.csv",     parse_dates=["added_at"])
genres   = load_csv(PROC / "artist_genres.csv")  # artist_id, artist_name, genres
features = load_csv(PROC / "audio_features.csv") # optional / may be empty

# ---------- header ----------
st.title("Spotify Wrapped (WIP) – Genre Focus")
st.caption("Last 30 days vs. all-time, built from your cached CSVs in `data/processed/`.")

with st.expander("Debug: columns present", expanded=False):
    st.write("recently_played.csv:", list(recent.columns))
    st.write("saved_tracks.csv:", list(saved.columns))
    st.write("artist_genres.csv:", list(genres.columns))
    st.write("audio_features.csv:", list(features.columns))

# ---------- 30-day window ----------
if recent.empty or "played_at" not in recent.columns:
    st.warning("No recent plays found (or `played_at` column missing). Run the fetch step first.")
    recent_30 = recent.iloc[0:0]
else:
    cutoff = recent["played_at"].max() - pd.Timedelta(days=30)
    recent_30 = recent[recent["played_at"] >= cutoff]

# ---------- genre summary (artist-based) ----------
lib_art = saved[["artist_id", "artist_name"]].drop_duplicates() if not saved.empty else saved
lib_art_gen = explode_genres(lib_art, genres)
lib_counts = (
    lib_art_gen.groupby("genre")["artist_id"].nunique().rename("size_all")
    if not lib_art_gen.empty else pd.Series(dtype="int64", name="size_all")
)

r30_art = recent_30[["artist_id", "artist_name"]].drop_duplicates() if not recent_30.empty else recent_30
r30_art_gen = explode_genres(r30_art, genres)
r30_counts = (
    r30_art_gen.groupby("genre")["artist_id"].nunique().rename("size_30")
    if not r30_art_gen.empty else pd.Series(dtype="int64", name="size_30")
)

summary = pd.concat([lib_counts, r30_counts], axis=1).fillna(0)
if summary.empty:
    summary = pd.DataFrame(columns=["size_all", "size_30"])

summary = (
    summary.assign(
        pct_all=lambda d: (d["size_all"] / d["size_all"].sum()) if d["size_all"].sum() else 0,
        pct_30=lambda d: (d["size_30"] / d["size_30"].sum()) if d["size_30"].sum() else 0,
        lift=lambda d: (d["pct_30"] / d["pct_all"]).replace([pd.NA, pd.NaT, float("inf")], 0)
    )
    .reset_index()
    .rename(columns={"index": "genre"})
)

# ---------- controls ----------
k = st.sidebar.slider("How many genres to show (by Lift vs Library)", min_value=5, max_value=50, value=20, step=5)

# ---------- top genres table ----------
st.subheader("Top Genres by Lift (Last 30 Days vs Library)")
top_genres = summary.sort_values("lift", ascending=False).head(k)
st.dataframe(
    friendly_cols(top_genres)[
        ["Genre", "Artists in Last 30 Days", "% of Last 30 Days", "Artists in Library", "% of Library", "Lift vs Library"]
    ].style.format({
        "% of Last 30 Days": "{:.1%}",
        "% of Library": "{:.1%}",
        "Lift vs Library": "{:.2f}",
    }),
    use_container_width=True,
)

# ---------- genre drill-down ----------
if not top_genres.empty and not recent_30.empty:
    picked_genre = st.selectbox("Drill down to see the tracks behind a genre:", top_genres["genre"])
    recent_30_tracks = recent_30[["track_id", "track_name", "artist_id", "artist_name", "album_name", "played_at"]]
    r30_track_gen = explode_genres(recent_30_tracks, genres)
    r30_track_gen = r30_track_gen[r30_track_gen["genre"] == picked_genre]

    st.markdown(f"### Tracks contributing to **{picked_genre}** in the last 30 days")
    if r30_track_gen.empty:
        st.info("No recent plays found for this genre in the last 30 days.")
    else:
        t_recent = (
            r30_track_gen.drop_duplicates(subset=["track_id"])
                         .rename(columns={
                             "track_name": "Track Name",
                             "artist_name": "Artist",
                             "album_name": "Album",
                             "played_at": "Played At",
                         })[["Track Name", "Artist", "Album", "Played At"]]
                         .sort_values("Played At", ascending=False)
        )
        st.dataframe(t_recent, use_container_width=True)

    st.markdown("#### Also-Tagged Genres for These Artists")
    if r30_track_gen.empty:
        st.caption("—")
    else:
        artists_this = r30_track_gen[["artist_id", "artist_name"]].drop_duplicates()

        # Merge on artist_id; unify artist_name
        g2 = genres.merge(artists_this, on="artist_id", how="inner", suffixes=("_lib", "_r30"))
        g2 = ensure_genre_list(g2)
        # Use the name from recent plays if present, otherwise library name
        if "artist_name_r30" in g2.columns or "artist_name_lib" in g2.columns:
            g2["artist_name"] = g2.get("artist_name_r30", pd.Series(index=g2.index)).combine_first(
                g2.get("artist_name_lib", pd.Series(index=g2.index))
            )

        also = g2.explode("genres").rename(columns={"genres": "Genre"})
        also = also[also["Genre"] != picked_genre]
        also_counts = (
            also.groupby("Genre")["artist_id"].nunique()
                .rename("Artists")
                .reset_index()
                .sort_values("Artists", ascending=False)
        )
        if also_counts.empty:
            st.caption("(No additional tags for these artists.)")
        else:
            st.dataframe(also_counts, use_container_width=True)

# ---------- multi-tag artists (last 30d) ----------
st.subheader("Artists with Multiple Genre Tags (Last 30 Days)")
if recent_30.empty or genres.empty:
    st.caption("No recent plays window or genres to analyze.")
else:
    r30_artists = recent_30[["artist_id", "artist_name"]].drop_duplicates()
    g3 = genres.merge(r30_artists, on="artist_id", how="inner", suffixes=("_lib", "_r30"))
    g3 = ensure_genre_list(g3)

    # unify artist_name column
    if "artist_name_r30" in g3.columns or "artist_name_lib" in g3.columns:
        g3["artist_name"] = g3.get("artist_name_r30", pd.Series(index=g3.index)).combine_first(
            g3.get("artist_name_lib", pd.Series(index=g3.index))
        )

    g3["n_tags"] = g3["genres"].apply(lambda lst: len(lst or []))
    multi = g3[g3["n_tags"] >= 3].copy()
    if multi.empty:
        st.caption("(No artists with 3+ tags in the last 30 days.)")
    else:
        multi["All Tags"] = multi["genres"].apply(lambda lst: ", ".join(lst) if isinstance(lst, list) else "")
        multi_view = (
            multi[["artist_name", "All Tags", "n_tags"]]
            .drop_duplicates()
            .rename(columns={"artist_name": "Artist", "n_tags": "# of Tags"})
            .sort_values("# of Tags", ascending=False)
        )
        st.dataframe(multi_view, use_container_width=True)

# ---------- peeks ----------
with st.expander("Peek: Latest 50 Recent Plays (from cache)"):
    if recent.empty:
        st.caption("—")
    else:
        st.dataframe(
            friendly_cols(
                recent.sort_values("played_at", ascending=False)
                      .rename(columns={
                          "track_name": "Track Name",
                          "artist_name": "Artist",
                          "album_name": "Album",
                          "played_at": "Played At",
                      })
            )[["Played At", "Track Name", "Artist", "Album"]].head(50),
            use_container_width=True,
        )

with st.expander("Peek: Library Stats"):
    if saved.empty:
        st.caption("—")
    else:
        st.write(f"Saved tracks in library: **{len(saved):,}**")
        st.dataframe(
            friendly_cols(
                saved.sort_values("added_at", ascending=False)
                     .rename(columns={
                         "track_name": "Track Name",
                         "artist_name": "Artist",
                         "album_name": "Album",
                         "added_at": "Added At",
                     })
            )[["Added At", "Track Name", "Artist", "Album"]].head(50),
            use_container_width=True,
        )

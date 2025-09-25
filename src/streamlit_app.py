import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta
from dotenv import load_dotenv

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ---------- Setup ----------
load_dotenv()
DATA_DIR = "data/processed"

@st.cache_data
def load_csv(name):
    path = os.path.join(DATA_DIR, name)
    return pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()

top_artists = load_csv("top_artists.csv")
top_tracks  = load_csv("top_tracks.csv")
recent      = load_csv("recently_played.csv")
saved       = load_csv("saved_tracks.csv")
features    = load_csv("audio_features.csv")
artist_genres = load_csv("artist_genres.csv")

st.set_page_config(page_title="Your Music Taste — Deep Dive", layout="wide")

# ---------- Sidebar Controls ----------
st.sidebar.title("Controls")
time_range = st.sidebar.selectbox("Top items time range", ["short_term","medium_term","long_term"], index=1)
hide_artists = st.sidebar.checkbox("Hide artist names (for sharing)", value=True)
include_recent = st.sidebar.checkbox("Include Recently Played (last ~1y)", value=True)
include_saved  = st.sidebar.checkbox("Include Saved Library timeline", value=True)
top_n_genres = st.sidebar.slider("Top Genres to show", 5, 30, 15)
min_plays_per_genre = st.sidebar.slider("Min count to include in charts", 1, 20, 3)

st.sidebar.write("---")
st.sidebar.caption("Tip: refresh data via `python src/fetch_spotify_data.py`")

# ---------- Helpers ----------
def masked_artist(name, idx):
    return f"Hidden Artist #{idx+1}" if hide_artists else name

def extract_genre_counts(df_tracks_like, artist_col="artist_id"):
    if df_tracks_like.empty or artist_genres.empty:
        return pd.DataFrame(columns=["genre","count"])
    g = df_tracks_like[[artist_col]].dropna().merge(artist_genres.explode("genres"), on=artist_col, how="left")
    g = g.rename(columns={"genres":"genre"})
    g = g.dropna(subset=["genre"])
    return g.groupby("genre").size().reset_index(name="count").sort_values("count", ascending=False)

def map_features(tracks_df, id_col="track_id"):
    if tracks_df.empty or features.empty: return pd.DataFrame()
    f = tracks_df[[id_col]].dropna().merge(features, left_on=id_col, right_on="id", how="left")
    return f

def polar_radar(avg_dict, title):
    cats = ["danceability","energy","valence","acousticness","instrumentalness","liveness","speechiness"]
    vals = [avg_dict.get(k,0) for k in cats]
    fig = go.Figure(data=go.Scatterpolar(r=vals, theta=cats, fill='toself'))
    fig.update_layout(title=title, margin=dict(l=20,r=20,t=50,b=20))
    return fig

def key_mode_heatmap(fdf, title):
    if fdf.empty: return go.Figure()
    pivot = fdf.pivot_table(index="key_name", columns="mode_name", values="id", aggfunc="count", fill_value=0)
    pivot = pivot.reindex(index=["C","C♯/D♭","D","D♯/E♭","E","F","F♯/G♭","G","G♯/A♭","A","A♯/B♭","B"])
    fig = px.imshow(pivot, text_auto=True, aspect="auto", title=title)
    return fig

def energy_valence_quadrant(fdf, title):
    if fdf.empty: return go.Figure()
    fig = px.scatter(fdf, x="energy", y="valence", hover_name="name", opacity=0.6)
    fig.add_vline(x=0.5); fig.add_hline(y=0.5)
    fig.update_layout(title=title, xaxis_title="Energy", yaxis_title="Valence (positivity)")
    return fig

def bpm_hist(fdf, title):
    if fdf.empty: return go.Figure()
    fig = px.histogram(fdf, x="tempo", nbins=40, title=title)
    fig.update_xaxes(title="Tempo (BPM)")
    return fig

def genre_over_time(df, time_col, title):
    if df.empty: return go.Figure()
    # join genres
    g = df.merge(artist_genres.explode("genres"), on="artist_id", how="left").rename(columns={"genres":"genre"})
    g = g.dropna(subset=["genre", time_col])
    g[time_col] = pd.to_datetime(g[time_col])
    g["ym"] = g[time_col].dt.to_period("M").dt.to_timestamp()
    top = g.groupby("genre").size().sort_values(ascending=False).head(15).index
    g2 = g[g["genre"].isin(top)]
    counts = g2.groupby(["ym","genre"]).size().reset_index(name="count")
    fig = px.area(counts, x="ym", y="count", color="genre", title=title)
    fig.update_xaxes(title="Month")
    fig.update_yaxes(title="Plays / Adds")
    return fig

# ---------- Data prep ----------
# Top entities filtered by range
ta = top_artists[top_artists["time_range"]==time_range].copy() if not top_artists.empty else pd.DataFrame()
tt = top_tracks[top_tracks["time_range"]==time_range].copy()  if not top_tracks.empty else pd.DataFrame()

# Mask artist names if requested
if not ta.empty and "name" in ta.columns:
    ta = ta.reset_index(drop=True)
    ta["display_name"] = [masked_artist(n,i) for i,n in enumerate(ta["name"])]
if not tt.empty:
    # harmonious column names
    tt["primary_artist_name"] = tt["artists"].apply(lambda arr: json.loads(arr.replace("'",'"'))[0]["name"] if isinstance(arr,str) and arr.startswith("[") else None) if "artists" in tt else tt.get("name")
    tt = tt.reset_index(drop=True)

# Audio features join for top tracks
tt_feats = map_features(tt.rename(columns={"id":"track_id"}), "track_id")
if not tt_feats.empty:
    tt_feats["name"] = tt.merge(tt_feats[["id"]], left_on="id", right_on="id", how="left")["name"]

# ---------- Layout ----------
st.title("🎧 Your Music Taste — Deep Dive")
st.caption("A private, story-oriented dashboard built from your Spotify data.")

# Row A: Hero metrics
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Top Artists (selected range)", value=0 if ta.empty else len(ta))
with col2:
    st.metric("Top Tracks (selected range)", value=0 if tt.empty else len(tt))
with col3:
    st.metric("Audio-featured tracks", value=0 if tt_feats.empty else tt_feats["id"].nunique())
with col4:
    st.metric("Genres cataloged", value=0 if artist_genres.empty else artist_genres.explode("genres")["genres"].nunique())

st.write("---")

# Row B: Genre snapshot + radar
left, right = st.columns([1.1, 1.0])

with left:
    # Genre counts (from top tracks + recent/saved if toggled)
    pools = []
    if not tt.empty:
        # derive artist_id via primary artist; best effort
        if "artists" in tt.columns:
            # robust parse
            def artist_id_first(row):
                try:
                    arr = json.loads(row.replace("'",'"'))
                    return arr[0]["id"]
                except Exception:
                    return None
            tt["artist_id"] = tt["artists"].apply(artist_id_first)
        pools.append(tt[["artist_id"]])
    if include_recent and not recent.empty:
        pools.append(recent[["artist_id"]])
    if include_saved and not saved.empty:
        pools.append(saved[["artist_id"]])

    cat_df = pd.concat(pools, ignore_index=True).dropna() if pools else pd.DataFrame(columns=["artist_id"])
    genre_counts = extract_genre_counts(cat_df) if not cat_df.empty else pd.DataFrame(columns=["genre","count"])
    if not genre_counts.empty:
        genre_counts = genre_counts[genre_counts["count"]>=min_plays_per_genre]
        fig = px.treemap(genre_counts.head(200), path=["genre"], values="count", title="Your Genre Landscape")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No genres available yet—play or save some tracks and refresh data.")

with right:
    # Average audio profile (radar)
    prof = {}
    if not tt_feats.empty:
        for k in ["danceability","energy","valence","acousticness","instrumentalness","liveness","speechiness"]:
            prof[k] = float(np.nanmean(tt_feats[k])) if k in tt_feats else 0.0
    st.plotly_chart(polar_radar(prof, "Your Average Audio Profile"), use_container_width=True)

st.write("---")

# Row C: Tempo + Key/Mode + Mood quad
c1,c2,c3 = st.columns([1.1,1.0,1.1])
with c1:
    st.plotly_chart(bpm_hist(tt_feats, "Most-Listened BPM (Top Tracks)"), use_container_width=True)
with c2:
    st.plotly_chart(key_mode_heatmap(tt_feats, "Musical Key × Mode (Top Tracks)"), use_container_width=True)
with c3:
    st.plotly_chart(energy_valence_quadrant(tt_feats, "Energy vs Valence — Mood Map"), use_container_width=True)

st.write("---")

# Row D: Evolution over time
st.subheader("📈 Evolution Over Time")
cols = st.columns(2)
with cols[0]:
    if include_recent and not recent.empty:
        st.plotly_chart(genre_over_time(recent, "played_at", "Genres Over Time (Recently Played)"), use_container_width=True)
    else:
        st.info("Recently Played timeline not included.")
with cols[1]:
    if include_saved and not saved.empty:
        st.plotly_chart(genre_over_time(saved, "added_at", "Genres Over Time (Saved Library)"), use_container_width=True)
    else:
        st.info("Saved Library timeline not included.")

st.write("---")

# Row E: Top artists/tracks tables (with hide toggle)
st.subheader("🎭 Spotlight — without giving it all away")
a, b = st.columns(2)
with a:
    st.markdown("**Top Artists (masked if selected)**")
    if not ta.empty:
        view = ta[["display_name","popularity","followers.total","genres"]].rename(columns={
            "display_name":"artist",
            "followers.total":"followers"
        }) if hide_artists else ta[["name","popularity","followers.total","genres"]].rename(columns={
            "name":"artist",
            "followers.total":"followers"
        })
        st.dataframe(view.head(20), use_container_width=True, hide_index=True)
    else:
        st.info("No top artists in this range.")

with b:
    st.markdown("**Top Tracks (primary artist masked if selected)**")
    if not tt.empty:
        # best-effort masking
        tt_show = tt.copy()
        if hide_artists:
            tt_show["primary_artist_name"] = [f"Hidden Artist #{i+1}" for i in range(len(tt_show))]
        table = tt_show[["name","primary_artist_name","popularity"]].rename(columns={"name":"track","primary_artist_name":"artist"})
        st.dataframe(table.head(20), use_container_width=True, hide_index=True)
    else:
        st.info("No top tracks in this range.")

st.write("---")

# Row F: Smart suggestions (no top-artist leakage)
st.subheader("🧠 Smart Suggestions You’ll Probably Like (Artist-Safe)")
st.caption("Based on your audio profile & genres, not just your top artists.")
if not genre_counts.empty:
    top_seed_genres = genre_counts.head(5)["genre"].tolist()
    st.markdown(f"**Seed genres:** {', '.join(top_seed_genres)}")
    st.markdown("- Use these as prompts for coworkers in your music league.")
else:
    st.info("No seed genres yet. Refresh after some listening.")

st.write("— End —")

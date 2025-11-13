import os
import time
from datetime import datetime, timezone
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import spotipy
from spotipy.oauth2 import SpotifyOAuth

DATA_DIR = "data/processed"
RAW_DIR = "data/raw"

KEY_NAMES = ["C","C♯/D♭","D","D♯/E♭","E","F","F♯/G♭","G","G♯/A♭","A","A♯/B♭","B"]

def sp_client(scopes: str):
    load_dotenv()
    return spotipy.Spotify(
        auth_manager=SpotifyOAuth(
            client_id=os.getenv("SPOTIPY_CLIENT_ID"),
            client_secret=os.getenv("SPOTIPY_CLIENT_SECRET"),
            redirect_uri=os.getenv("SPOTIPY_REDIRECT_URI"),
            scope=scopes,
            cache_path=".cache",
            open_browser=True,
        )
    )

def chunked(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i+n]

def fetch_top(sp, entity="artists", time_range="medium_term", limit=50):
    if entity == "artists":
        res = sp.current_user_top_artists(limit=limit, time_range=time_range)
    else:
        res = sp.current_user_top_tracks(limit=limit, time_range=time_range)
    return res.get("items", [])

def fetch_recently_played(sp, limit=50, after_days=365):
    # Page backwards from "now" until we hit cutoff or run out
    items = []
    before = int(datetime.now(timezone.utc).timestamp() * 1000)
    cutoff = int((datetime.now(timezone.utc) - pd.Timedelta(days=after_days)).timestamp() * 1000)

    while True:
        res = sp.current_user_recently_played(limit=min(limit, 50), before=before)
        batch = res.get("items", [])
        if not batch:
            break

        items.extend(batch)

        # compute next 'before' from the oldest item in this batch
        oldest_ms = min(int(pd.to_datetime(i["played_at"]).timestamp() * 1000) for i in batch)
        before = oldest_ms - 1

        if before < cutoff:
            break
        if len(batch) < 2:
            break

        time.sleep(0.1)

    return items

def fetch_saved_tracks(sp, max_total=5000):
    items = []
    limit = 50
    offset = 0
    while True:
        res = sp.current_user_saved_tracks(limit=limit, offset=offset)
        batch = res.get("items", [])
        if not batch:
            break
        items.extend(batch)
        if len(batch) < limit or len(items) >= max_total:
            break
        offset += limit
        time.sleep(0.1)
    return items

def get_audio_features(sp, track_ids):
    """
    Spotify has tightened access to /audio-features for new/dev apps.
    We keep the pipeline resilient by returning [] on 403.
    """
    try:
        feats = []
        for group in chunked(track_ids, 100):
            # This may raise a 403 in your app state; we'll handle that.
            part = sp.audio_features(group)
            if part:
                feats.extend([f for f in part if f])
            time.sleep(0.05)
        return feats
    except spotipy.exceptions.SpotifyException as e:
        # Gracefully degrade
        print("audio_features blocked or failed (continuing without features):", repr(e))
        return []

def normalize_recent(items):
    rows = []
    for it in items:
        played_at = pd.to_datetime(it["played_at"])
        track = it["track"]
        rows.append({
            "played_at": played_at,
            "track_id": track.get("id"),
            "track_name": track.get("name"),
            "artist_id": track["artists"][0]["id"] if track.get("artists") else None,
            "artist_name": track["artists"][0]["name"] if track.get("artists") else None,
            "album_name": (track.get("album") or {}).get("name"),
        })
    return pd.DataFrame(rows)

def normalize_saved(items):
    rows = []
    for it in items:
        added_at = pd.to_datetime(it["added_at"])
        track = it["track"]
        rows.append({
            "added_at": added_at,
            "track_id": track.get("id"),
            "track_name": track.get("name"),
            "artist_id": track["artists"][0]["id"] if track.get("artists") else None,
            "artist_name": track["artists"][0]["name"] if track.get("artists") else None,
            "album_name": (track.get("album") or {}).get("name"),
        })
    return pd.DataFrame(rows)

def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(RAW_DIR, exist_ok=True)

    # Use spaces (OAuth expects space-delimited scopes)
    scopes = os.getenv(
        "SPOTIFY_SCOPES",
        "user-top-read user-read-recently-played user-library-read playlist-modify-private"
    )
    sp = sp_client(scopes)

    # 1) Top Artists/Tracks across ranges
    frames_artists, frames_tracks = [], []
    for rng in ["short_term", "medium_term", "long_term"]:
        arts = fetch_top(sp, "artists", rng, 50)
        trs  = fetch_top(sp, "tracks",  rng, 50)

        df_a = pd.json_normalize(arts)
        if not df_a.empty:
            df_a["time_range"] = rng
            frames_artists.append(df_a)

        df_t = pd.json_normalize(trs)
        if not df_t.empty:
            df_t["time_range"] = rng
            frames_tracks.append(df_t)

    top_artists = pd.concat(frames_artists, ignore_index=True) if frames_artists else pd.DataFrame()
    top_tracks  = pd.concat(frames_tracks,  ignore_index=True) if frames_tracks  else pd.DataFrame()

    # 2) Recently Played (~365 days back, paged)
    # Seed call not strictly needed; we normalize below
    rec_all = fetch_recently_played(sp, limit=50, after_days=365)
    recent_df = normalize_recent(rec_all) if rec_all else pd.DataFrame()

    # 3) Saved Tracks (library)
    saved_items = fetch_saved_tracks(sp, max_total=5000)
    saved_df = normalize_saved(saved_items) if saved_items else pd.DataFrame()

 # --- 4) Audio Features for tracks we’ve seen ---
    track_ids = set()
    for df in [top_tracks, recent_df, saved_df]:
        if not df.empty and "track_id" in df.columns:
            track_ids.update(df["track_id"].dropna().tolist())
        elif not df.empty and "id" in df.columns:
            track_ids.update(df["id"].dropna().tolist())
    track_ids = [t for t in track_ids if isinstance(t, str)]
    feats_df = pd.DataFrame()
    try:
        features = get_audio_features(sp, track_ids) if track_ids else []
        if features:
            feats_df = pd.json_normalize([f for f in features if f])
    except Exception as e:
        print("audio_features blocked or failed (continuing without features):", repr(e))

    # Always ensure a file is written so downstream code never breaks
    if feats_df.empty:
        feats_df = pd.DataFrame(columns=[
            "id","key","mode","tempo","danceability","energy",
            "speechiness","acousticness","instrumentalness","liveness","valence"
        ])


    # 5) Artist genres from recent & saved
    artist_ids = set()
    for df in [recent_df, saved_df]:
        if not df.empty and "artist_id" in df.columns:
            artist_ids.update(df["artist_id"].dropna().tolist())

    genres_rows = []
    for group in chunked(list(artist_ids), 50):
        arts = sp.artists(group)["artists"] if group else []
        for a in arts:
            genres_rows.append({
                "artist_id": a.get("id"),
                "artist_name": a.get("name"),
                "genres": a.get("genres", [])
            })
        time.sleep(0.05)
    genres_df = pd.DataFrame(genres_rows)

    # 6) Save everything that exists
    if not top_artists.empty: top_artists.to_csv(f"{DATA_DIR}/top_artists.csv", index=False)
    if not top_tracks.empty:  top_tracks.to_csv(f"{DATA_DIR}/top_tracks.csv", index=False)
    if not recent_df.empty:   recent_df.to_csv(f"{DATA_DIR}/recently_played.csv", index=False)
    if not saved_df.empty:    saved_df.to_csv(f"{DATA_DIR}/saved_tracks.csv", index=False)
    if not feats_df.empty:    feats_df.to_csv(f"{DATA_DIR}/audio_features.csv", index=False)
    if not genres_df.empty:   genres_df.to_csv(f"{DATA_DIR}/artist_genres.csv", index=False)

    print("✅ Data refresh complete. Wrote:")
    if not top_artists.empty: print("- data/processed/top_artists.csv")
    if not top_tracks.empty:  print("- data/processed/top_tracks.csv")
    if not recent_df.empty:   print("- data/processed/recently_played.csv")
    if not saved_df.empty:    print("- data/processed/saved_tracks.csv")
    if not feats_df.empty:    print("- data/processed/audio_features.csv")
    if not genres_df.empty:   print("- data/processed/artist_genres.csv")

if __name__ == "__main__":
    main()

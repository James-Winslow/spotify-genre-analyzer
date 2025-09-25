import os, time, math
from datetime import datetime, timezone
from collections import defaultdict, Counter

import pandas as pd
import numpy as np
from dotenv import load_dotenv
import spotipy
from spotipy.oauth2 import SpotifyOAuth

DATA_DIR = "data/processed"
RAW_DIR = "data/raw"

KEY_NAMES = ["C","C♯/D♭","D","D♯/E♭","E","F","F♯/G♭","G","G♯/A♭","A","A♯/B♭","B"]

def sp_client(scopes):
    load_dotenv()
    return spotipy.Spotify(auth_manager=SpotifyOAuth(
        client_id=os.getenv("SPOTIPY_CLIENT_ID"),
        client_secret=os.getenv("SPOTIPY_CLIENT_SECRET"),
        redirect_uri=os.getenv("SPOTIPY_REDIRECT_URI"),
        scope=scopes,
        cache_path=".cache",
        open_browser=True,
    ))

def chunked(iterable, n):
    for i in range(0, len(iterable), n):
        yield iterable[i:i+n]

def fetch_top(sp, entity="artists", time_range="medium_term", limit=50):
    if entity == "artists":
        res = sp.current_user_top_artists(limit=limit, time_range=time_range)
    else:
        res = sp.current_user_top_tracks(limit=limit, time_range=time_range)
    return res.get("items", [])

def fetch_recently_played(sp, limit=50, after_days=365):
    # Spotify limits to ~50 latest per call; we’ll page until exhausted or hit time window
    items = []
    before = int(datetime.now(timezone.utc).timestamp()*1000)
    cutoff = int((datetime.now(timezone.utc) - pd.Timedelta(days=after_days)).timestamp() * 1000)
    while True:
        res = sp.current_user_recently_played(limit=min(limit,50), before=before)
        batch = res.get("items", [])
        if not batch:
            break
        items.extend(batch)
        before = min(int(i["played_at_dt"].timestamp()*1000) if "played_at_dt" in i else int(pd.to_datetime(i["played_at"]).timestamp()*1000) for i in batch) - 1
        # Stop if we’re before cutoff
        if before < cutoff:
            break
        if len(batch) < 2:
            break
        time.sleep(0.1)
    return items

def fetch_saved_tracks(sp, max_total=5000):
    # user-library-read; page through saved tracks w/ added_at timestamps
    items = []
    limit = 50
    offset = 0
    while True:
        res = sp.current_user_saved_tracks(limit=limit, offset=offset)
        batch = res.get("items", [])
        items.extend(batch)
        if len(batch) < limit or len(items) >= max_total:
            break
        offset += limit
        time.sleep(0.1)
    return items

def get_audio_features(sp, track_ids):
    feats = []
    for group in chunked(track_ids, 100):
        feats.extend(sp.audio_features(group))
        time.sleep(0.05)
    return feats

def normalize_recent(items):
    rows = []
    for it in items:
        played_at = pd.to_datetime(it["played_at"])
        track = it["track"]
        rows.append({
            "played_at": played_at,
            "track_id": track["id"],
            "track_name": track["name"],
            "artist_id": track["artists"][0]["id"] if track["artists"] else None,
            "artist_name": track["artists"][0]["name"] if track["artists"] else None,
            "album_name": track["album"]["name"] if track.get("album") else None,
        })
    return pd.DataFrame(rows)

def normalize_saved(items):
    rows = []
    for it in items:
        added_at = pd.to_datetime(it["added_at"])
        track = it["track"]
        rows.append({
            "added_at": added_at,
            "track_id": track["id"],
            "track_name": track["name"],
            "artist_id": track["artists"][0]["id"] if track["artists"] else None,
            "artist_name": track["artists"][0]["name"] if track["artists"] else None,
            "album_name": track["album"]["name"] if track.get("album") else None,
        })
    return pd.DataFrame(rows)

def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(RAW_DIR, exist_ok=True)

    scopes = os.getenv("SPOTIFY_SCOPES", "user-top-read,user-read-recently-played,user-library-read,playlist-modify-private")
    sp = sp_client(scopes)

    # --- 1) Top Artists/Tracks across ranges (short/medium/long) ---
    frames_artists = []
    frames_tracks = []
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
    top_tracks  = pd.concat(frames_tracks, ignore_index=True) if frames_tracks else pd.DataFrame()

    # --- 2) Recently Played (last ~365 days)
    rec_items = sp.current_user_recently_played(limit=50)  # seed call to attach played_at_dt
    # spotipy returns ISO strings; normalize below via our helper
    rec_all = fetch_recently_played(sp, limit=50, after_days=365)
    recent_df = normalize_recent(rec_all) if rec_all else pd.DataFrame()

    # --- 3) Saved Tracks (library) with added_at timestamps
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
    features = get_audio_features(sp, track_ids) if track_ids else []
    feats_df = pd.json_normalize([f for f in features if f])

    # map key/mode to labels
    if not feats_df.empty and "key" in feats_df.columns:
        feats_df["key_name"] = feats_df["key"].apply(lambda k: KEY_NAMES[k] if isinstance(k, (int, np.integer)) and 0 <= k < 12 else None)
        feats_df["mode_name"] = feats_df["mode"].map({1:"Major",0:"Minor"})

    # --- 5) Artist genres for recent & saved (for evolution)
    artist_ids = set()
    for df in [recent_df, saved_df]:
        if not df.empty and "artist_id" in df.columns:
            artist_ids.update(df["artist_id"].dropna().tolist())

    genres_rows = []
    for group in chunked(list(artist_ids), 50):
        arts = sp.artists(group)["artists"]
        for a in arts:
            genres_rows.append({"artist_id": a["id"], "artist_name": a["name"], "genres": a.get("genres", [])})
        time.sleep(0.05)
    genres_df = pd.DataFrame(genres_rows)

    # --- 6) Save everything ---
    if not top_artists.empty: top_artists.to_csv(f"{DATA_DIR}/top_artists.csv", index=False)
    if not top_tracks.empty:  top_tracks.to_csv(f"{DATA_DIR}/top_tracks.csv", index=False)
    if not recent_df.empty:   recent_df.to_csv(f"{DATA_DIR}/recently_played.csv", index=False)
    if not saved_df.empty:    saved_df.to_csv(f"{DATA_DIR}/saved_tracks.csv", index=False)
    if not feats_df.empty:    feats_df.to_csv(f"{DATA_DIR}/audio_features.csv", index=False)
    if not genres_df.empty:   genres_df.to_csv(f"{DATA_DIR}/artist_genres.csv", index=False)

    print("✅ Data refresh complete. Files saved in data/processed:")
    print("- top_artists.csv, top_tracks.csv")
    print("- recently_played.csv, saved_tracks.csv")
    print("- audio_features.csv")
    print("- artist_genres.csv")

if __name__ == "__main__":
    main()

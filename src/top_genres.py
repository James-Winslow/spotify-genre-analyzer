import os
from collections import Counter

from dotenv import load_dotenv
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import matplotlib.pyplot as plt

def main():
    load_dotenv()

    client_id = os.getenv("SPOTIPY_CLIENT_ID")
    client_secret = os.getenv("SPOTIPY_CLIENT_SECRET")
    redirect_uri = os.getenv("SPOTIPY_REDIRECT_URI", "http://localhost:8888/callback")

    if not client_id or not client_secret:
        raise RuntimeError("Missing SPOTIPY_CLIENT_ID or SPOTIPY_CLIENT_SECRET in .env")

    time_range = os.getenv("SPOTIFY_TIME_RANGE", "medium_term")
    limit = int(os.getenv("SPOTIFY_TOP_LIMIT", "50"))
    top_n = int(os.getenv("TOP_N_GENRES", "10"))

    sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
        client_id=client_id,
        client_secret=client_secret,
        redirect_uri=redirect_uri,
        scope="user-top-read",
        open_browser=True,
        cache_path=".cache"
    ))

    top_artists = sp.current_user_top_artists(limit=limit, time_range=time_range)

    genres = []
    for artist in top_artists.get("items", []):
        genres.extend(artist.get("genres", []))

    if not genres:
        print("No genres found. Try changing time_range or verify your listening history.")
        return

    counts = Counter(genres).most_common(top_n)

    print(f"\nTop {top_n} Genres ({time_range}):")
    for g, c in counts:
        print(f"{g}: {c}")

    labels, values = zip(*counts)
    plt.figure()
    plt.barh(labels, values)
    plt.gca().invert_yaxis()
    plt.title(f"Top {top_n} Spotify Genres ({time_range})")
    plt.xlabel("Count")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()

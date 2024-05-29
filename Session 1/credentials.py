import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
 
CLIENT_ID = "849c76e710024c2c8d2cb0d2e7ca3fd0"
CLIENT_SECRET = "475f4ecd34aa4a0388f02fdd697c13bc"
auth_manager = SpotifyClientCredentials(
    client_id=CLIENT_ID,
    client_secret=CLIENT_SECRET)
sp = spotipy.Spotify(auth_manager=auth_manager)
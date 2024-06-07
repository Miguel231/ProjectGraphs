import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

CLIENT_ID = '6b66690d17274d08848489ba3a4e5ba4'
CLIENT_SECRET = '7dd5fa4c5f32476b80dc669dbeff2979'

#CLIENT_ID = "849c76e710024c2c8d2cb0d2e7ca3fd0"
#CLIENT_SECRET = "475f4ecd34aa4a0388f02fdd697c13bc"
auth_manager = SpotifyClientCredentials(
    client_id=CLIENT_ID,
    client_secret=CLIENT_SECRET)
sp = spotipy.Spotify(auth_manager=auth_manager)
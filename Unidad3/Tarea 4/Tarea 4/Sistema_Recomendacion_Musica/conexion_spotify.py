import spotipy
from spotipy.oauth2 import SpotifyOAuth

# Configuración de autenticación
CLIENT_ID = 'b175d731588c4409adac84685ab58940'
CLIENT_SECRET = 'a36efe4b77fd435b99287809ee5d7c02'
REDIRECT_URI = 'http://127.0.0.1:3000'
SCOPE = "user-top-read"  # Permite acceder a las canciones más escuchadas

def conectar_spotify():
    sp = spotipy.Spotify(auth_manager=SpotifyOAuth(client_id=CLIENT_ID,
                                                   client_secret=CLIENT_SECRET,
                                                   redirect_uri=REDIRECT_URI,
                                                   scope=SCOPE))
    return sp

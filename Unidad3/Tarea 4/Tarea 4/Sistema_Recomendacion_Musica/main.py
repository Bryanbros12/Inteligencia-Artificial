from conexion_spotify import conectar_spotify
from preferencias_usuario import obtener_canciones_mas_escuchadas, obtener_preferencias_usuario

def recomendar_canciones(sp, estado_animo, actividad, genero):
    consulta = f"{estado_animo} {actividad} {genero}"
    resultados = sp.search(q=consulta, type='track', limit=5)

    print(f"\n🎧 Recomendaciones para: {consulta.upper()}")
    for track in resultados['tracks']['items']:
        nombre = track['name']
        artista = track['artists'][0]['name']
        url = track['external_urls']['spotify']
        print(f"{nombre} - {artista} → {url}")

    # Obtener canciones más escuchadas del usuario y mostrar
    print("\n🎧 Canciones más escuchadas en tu sesión:")
    canciones_mas_escuchadas = obtener_canciones_mas_escuchadas(sp)
    for cancion in canciones_mas_escuchadas:
        print(cancion)

# Programa principal
if __name__ == '__main__':
    sp = conectar_spotify()
    estado, actividad, genero = obtener_preferencias_usuario()
    recomendar_canciones(sp, estado, actividad, genero)

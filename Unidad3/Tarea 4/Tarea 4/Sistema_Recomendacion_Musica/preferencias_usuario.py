def obtener_preferencias_usuario():
    print("=== Recomendación musical personalizada ===")

    estado_animo = input("¿Cuál es tu estado de ánimo? (feliz, triste, relajado, etc.): ").strip().lower()
    actividad = input("¿Qué estás haciendo? (estudiar, hacer ejercicio, dormir, etc.): ").strip().lower()
    genero = input("¿Qué género musical prefieres? (pop, rock, reggaetón, etc.): ").strip().lower()

    return estado_animo, actividad, genero


def obtener_canciones_mas_escuchadas(sp):
    # Obtén las canciones más escuchadas del usuario
    top_tracks = sp.current_user_top_tracks(limit=5)  # Cambia el número si lo necesitas

    canciones = []
    for track in top_tracks['items']:
        canciones.append(f"{track['name']} - {track['artists'][0]['name']} ({track['external_urls']['spotify']})")

    return canciones

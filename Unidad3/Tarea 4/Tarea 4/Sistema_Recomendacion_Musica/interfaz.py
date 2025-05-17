import tkinter as tk
from tkinter import ttk
from conexion_spotify import conectar_spotify
from preferencias_usuario import obtener_canciones_mas_escuchadas
from spotipy import Spotify


def recomendar_canciones_gui(sp: Spotify, estado_animo, actividad, genero, texto_resultado):
    consulta = f"{estado_animo} {actividad} {genero}"
    resultados = sp.search(q=consulta, type='track', limit=5)

    texto_resultado.configure(state='normal')  # Habilita edición
    texto_resultado.delete('1.0', tk.END)  # Limpia texto anterior
    texto_resultado.insert(tk.END, f"🎧 Recomendaciones para: {consulta.upper()}\n\n")

    for track in resultados['tracks']['items']:
        nombre = track['name']
        artista = track['artists'][0]['name']
        url = track['external_urls']['spotify']
        texto_resultado.insert(tk.END, f"{nombre} - {artista}\n{url}\n\n")

    # Mostrar canciones más escuchadas en la interfaz
    texto_resultado.insert(tk.END, "\n🎧 Canciones más escuchadas en tu sesión:\n")
    canciones_mas_escuchadas = obtener_canciones_mas_escuchadas(sp)
    for cancion in canciones_mas_escuchadas:
        texto_resultado.insert(tk.END, f"{cancion}\n")

    texto_resultado.configure(state='disabled')  # Desactiva edición


def main():
    sp = conectar_spotify()

    ventana = tk.Tk()
    ventana.title("Sistema Recomendador de Música 🎵")
    ventana.geometry("600x500")
    ventana.resizable(False, False)

    # Campos de entrada
    tk.Label(ventana, text="Estado de ánimo:").pack()
    entrada_estado = ttk.Entry(ventana, width=50)
    entrada_estado.pack()

    tk.Label(ventana, text="Actividad:").pack()
    entrada_actividad = ttk.Entry(ventana, width=50)
    entrada_actividad.pack()

    tk.Label(ventana, text="Género musical:").pack()
    entrada_genero = ttk.Entry(ventana, width=50)
    entrada_genero.pack()

    # Área de resultados (no editable)
    texto_resultado = tk.Text(ventana, height=15, width=70, state='disabled', wrap='word')
    texto_resultado.pack(pady=10)

    # Botón para generar recomendaciones
    def ejecutar_recomendacion():
        estado = entrada_estado.get()
        actividad = entrada_actividad.get()
        genero = entrada_genero.get()
        if estado and actividad and genero:
            recomendar_canciones_gui(sp, estado, actividad, genero, texto_resultado)

    boton = ttk.Button(ventana, text="Recomendar Canciones", command=ejecutar_recomendacion)
    boton.pack(pady=10)

    ventana.mainloop()


if __name__ == '__main__':
    main()

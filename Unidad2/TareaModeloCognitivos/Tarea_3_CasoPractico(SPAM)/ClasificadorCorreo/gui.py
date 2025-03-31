import tkinter as tk
from clasificar import clasificar_correo

def analizar_correo():
    texto_correo = entrada_correo.get("1.0", tk.END)
    resultado = clasificar_correo(texto_correo)
    etiqueta_resultado.config(text=f"Resultado: {resultado}")

# Crear ventana
ventana = tk.Tk()
ventana.title("Clasificador de Spam")

# Área de texto para ingresar el correo
entrada_correo = tk.Text(ventana, height=10, width=50)
entrada_correo.pack()

# Botón para analizar
boton_analizar = tk.Button(ventana, text="Analizar Correo", command=analizar_correo)
boton_analizar.pack()

# Etiqueta para mostrar resultado
etiqueta_resultado = tk.Label(ventana, text="")
etiqueta_resultado.pack()

# Iniciar interfaz
ventana.mainloop()

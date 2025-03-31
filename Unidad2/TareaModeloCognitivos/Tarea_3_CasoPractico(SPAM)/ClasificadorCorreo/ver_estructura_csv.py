import pandas as pd

# Cargar el archivo CSV
datos = pd.read_csv('spam_assassin.csv')

# Ver las primeras 5 filas del archivo CSV
print(datos.head())

# Ver los nombres de las columnas
print(datos.columns)

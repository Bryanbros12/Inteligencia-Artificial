import pandas as pd
import re
import string
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

# Descargar stopwords en español
nltk.download("stopwords")
palabras_vacias = set(stopwords.words("english"))

def limpiar_texto(texto):
    """Limpia el texto eliminando caracteres especiales, stopwords y lo convierte a minúsculas."""
    texto = texto.lower()
    texto = re.sub(r'\d+', '', texto)  # Eliminar números
    texto = texto.translate(str.maketrans("", "", string.punctuation))  # Eliminar puntuación
    texto = texto.strip()
    palabras = texto.split()
    palabras = [palabra for palabra in palabras if palabra not in palabras_vacias]
    return " ".join(palabras)

def cargar_y_preprocesar(ruta_archivo="spam_assassin.csv"):
    """Carga el dataset, limpia los textos y los vectoriza con TF-IDF."""
    datos = pd.read_csv(ruta_archivo)  # 📌 Cargar el archivo CSV

    datos.drop_duplicates(inplace=True)  # Eliminar duplicados
    datos["texto_limpio"] = datos["text"].astype(str).apply(limpiar_texto)  # Usar 'text' para el contenido del correo

    # Vectorización con TF-IDF
    vectorizador = TfidfVectorizer(max_features=5000)
    X = vectorizador.fit_transform(datos["texto_limpio"]).toarray()

    y = datos["target"]

    return X, y, vectorizador


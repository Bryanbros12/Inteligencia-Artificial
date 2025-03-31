import joblib
from preproceso import limpiar_texto

# Cargar modelo y vectorizador
modelo = joblib.load("modelo_spam.pkl")
vectorizador = joblib.load("vectorizador_tfidf.pkl")

def clasificar_correo(texto):
    """Clasifica un correo como spam o no spam."""
    texto_limpio = limpiar_texto(texto)
    texto_vectorizado = vectorizador.transform([texto_limpio]).toarray()
    prediccion = modelo.predict(texto_vectorizado)[0]
    return "Spam" if prediccion == 1 else "No Spam"

if __name__ == "__main__":
    correo_prueba = input("Ingresa el correo a analizar: ")
    resultado = clasificar_correo(correo_prueba)
    print(f"El correo es: {resultado}")

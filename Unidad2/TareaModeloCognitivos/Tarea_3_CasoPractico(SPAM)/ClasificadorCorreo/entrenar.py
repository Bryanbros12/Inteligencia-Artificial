import joblib
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from preproceso import cargar_y_preprocesar

# Cargar datos preprocesados
X, y, vectorizador = cargar_y_preprocesar()

# Dividir los datos en conjunto de entrenamiento (80%) y prueba (20%)
X_entrenamiento, X_prueba, y_entrenamiento, y_prueba = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar el modelo Naive Bayes
modelo = MultinomialNB()
modelo.fit(X_entrenamiento, y_entrenamiento)

# Evaluar el modelo
y_prediccion = modelo.predict(X_prueba)
precision = accuracy_score(y_prueba, y_prediccion)

print(f"Precisión del modelo: {precision:.4f}")
print("Reporte de clasificación:\n", classification_report(y_prueba, y_prediccion))

# Guardar el modelo y el vectorizador
joblib.dump(modelo, "modelo_spam.pkl")
joblib.dump(vectorizador, "vectorizador_tfidf.pkl")

print("Modelo y vectorizador guardados correctamente.")

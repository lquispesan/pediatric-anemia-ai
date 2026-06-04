import pandas as pd

# Herramientas para dividir el dataset
from sklearn.model_selection import train_test_split

# Algoritmo de clasificación
from sklearn.ensemble import RandomForestClassifier

# Métricas de evaluación
from sklearn.metrics import accuracy_score, classification_report


# ==========================================================
# CARGA DEL DATASET
# ==========================================================

# Leer el archivo CSV que contiene los registros hematológicos
df = pd.read_csv("anemia.csv")

# Mostrar información básica del dataset
print("\n" + "="*60)
print("INFORMACIÓN GENERAL DEL DATASET")
print("="*60)

print(f"Cantidad de registros: {len(df)}")
print(f"Cantidad de columnas: {len(df.columns)}")

print("\nColumnas disponibles:")
print(df.columns.tolist())


# ==========================================================
# DEFINICIÓN DE VARIABLES
# ==========================================================

# Variables de entrada (features)
X = df[['Gender', 'Hemoglobin', 'MCH', 'MCHC', 'MCV']]

# Variable objetivo (target)
y = df['Result']


# ==========================================================
# DIVISIÓN DEL DATASET
# ==========================================================

# 80% para entrenamiento
# 20% para pruebas
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\n" + "="*60)
print("DIVISIÓN DEL DATASET")
print("="*60)

print(f"Registros para entrenamiento: {len(X_train)}")
print(f"Registros para prueba: {len(X_test)}")


# ==========================================================
# ENTRENAMIENTO DEL MODELO
# ==========================================================

# Crear modelo Random Forest
modelo = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# Entrenar modelo
modelo.fit(X_train, y_train)


# ==========================================================
# PREDICCIÓN
# ==========================================================

# Realizar predicciones usando los datos de prueba
predicciones = modelo.predict(X_test)


# ==========================================================
# EVALUACIÓN DEL MODELO
# ==========================================================

accuracy = accuracy_score(y_test, predicciones)

print("\n" + "="*60)
print("RESULTADOS DEL MODELO")
print("="*60)

print(f"Accuracy: {accuracy:.4f}")
print(f"Accuracy (%): {accuracy*100:.2f}%")

print("\nReporte de Clasificación:")
print(classification_report(y_test, predicciones))


# ==========================================================
# ANÁLISIS EXPLORATORIO
# ==========================================================

print("\n" + "="*60)
print("PROMEDIOS POR CLASE")
print("="*60)

print(df.groupby("Result").mean())


print("\n" + "="*60)
print("ESTADÍSTICAS DE HEMOGLOBINA POR CLASE")
print("="*60)

print(df.groupby("Result")["Hemoglobin"].describe())


# ==========================================================
# IMPORTANCIA DE VARIABLES
# ==========================================================

print("\n" + "="*60)
print("IMPORTANCIA DE VARIABLES")
print("="*60)

for nombre, importancia in zip(X.columns, modelo.feature_importances_):
    print(f"{nombre:<12} -> {importancia:.4f}")
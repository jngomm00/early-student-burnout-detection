import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

# Aqui importo los algoritmos
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
import xgboost as xgb

# 1. Carga y preparacion de los datos
df = pd.read_csv('./data-clean/master_student_data.csv')
cols_to_drop = [col for col in ['id_student', 'code_module', 'code_presentation', 'final_result', 'target'] if
                col in df.columns]

X = df.drop(columns=cols_to_drop)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

# 2. ESCALADO DE DATOS para KNN, Regresión Logística y Redes Neuronales
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. Inicializo los modelos
modelos = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    "Regresión Logística": LogisticRegression(max_iter=1000, random_state=42),
    "KNN (K-Vecinos)": KNeighborsClassifier(n_neighbors=5),
    "Red Neuronal (MLP)": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=42),
    "XGBoost": xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
}

# 4. Entrenar y evaluar
resultados = []

print("Iniciando la competición de modelos... (esto puede tardar unos segundos)\n")

for nombre, modelo in modelos.items():
    print(f"Entrenando {nombre}...")

    # Entrenar
    modelo.fit(X_train_scaled, y_train)

    # Predicciones
    y_pred = modelo.predict(X_test_scaled)

    # Métricas
    acc = accuracy_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred, pos_label=1)
    prec = precision_score(y_test, y_pred, pos_label=1)
    f1 = f1_score(y_test, y_pred, pos_label=1)

    resultados.append(
        {"Modelo": nombre, "Accuracy": acc, "Recall (Riesgo)": rec, "Precisión (Riesgo)": prec, "F1-Score": f1})

# 5. Resultados
df_resultados = pd.DataFrame(resultados).sort_values(by="Recall (Riesgo)", ascending=False)
print("\n--- CLASIFICACIÓN FINAL DE MODELOS ---")
print(df_resultados.to_string(index=False))

# 6. Graficas
df_melted = df_resultados.melt(id_vars="Modelo", var_name="Métrica", value_name="Puntuación")

plt.figure(figsize=(12, 6))
sns.barplot(x="Modelo", y="Puntuación", hue="Métrica", data=df_melted, palette="viridis")
plt.title("Comparativa de Algoritmos (Detección a 90 días)", fontsize=16)
plt.ylim(0, 1.0)
plt.ylabel("Puntuación (0 a 1)")
plt.xlabel("Algoritmo Utilizado")
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()
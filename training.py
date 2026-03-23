from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report

# 1. Cargo la tabla
df = pd.read_csv('./data-clean/master_student_data.csv')

# 2. Limpieza de los datos
# Quitamos columnas que no son números o que harían trampa (el target)
columnas_a_ignorar = ['id_student', 'code_module', 'code_presentation', 'final_result', 'target']
cols_to_drop = [col for col in columnas_a_ignorar if col in df.columns]

X = df.drop(columns=cols_to_drop)
y = df['target']

# 3. Dividimos los datos (80% entrenamiento, 20% pruebas)
# stratify=y asegura que mantengamos la misma proporción de aprobados/reprobados en ambos grupos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

print(f"Entrenando el modelo con {len(X_train)} estudiantes...")
print(f"Evaluando el modelo con {len(X_test)} estudiantes...\n")

# 4. Crear y Entrenar el Modelo (Random Forest)
# n_estimators=100 significa que creará 100 árboles de decisión y tomará la mayoría de votos
modelo_rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
modelo_rf.fit(X_train, y_train)

# 5. Hacer predicciones (El examen)
predicciones = modelo_rf.predict(X_test)

# 6. Mostrar los Resultados
# 1. GRÁFICA DE LA MATRIZ DE CONFUSIÓN (Heatmap)
cm = confusion_matrix(y_test, predicciones)

plt.figure(figsize=(8, 6))
# Usamos un mapa de calor azul
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', annot_kws={"size": 16},
            xticklabels=['Éxito (0)', 'Riesgo (1)'],
            yticklabels=['Éxito (0)', 'Riesgo (1)'])

plt.title('Matriz de Confusión: Predicciones vs Realidad', fontsize=15)
plt.xlabel('Lo que predijo el modelo', fontsize=12)
plt.ylabel('Lo que realmente pasó', fontsize=12)
plt.tight_layout()
plt.show()


# 2. GRÁFICA DE MÉTRICAS (Precisión, Recall, F1-Score)
# Extraemos el reporte en formato diccionario
reporte_dict = classification_report(y_test, predicciones, output_dict=True)

# Preparamos los datos para la gráfica
datos_metricas = {
    'Métrica': ['Precisión', 'Recall', 'F1-Score', 'Precisión', 'Recall', 'F1-Score'],
    'Valor': [
        reporte_dict['0']['precision'], reporte_dict['0']['recall'], reporte_dict['0']['f1-score'],
        reporte_dict['1']['precision'], reporte_dict['1']['recall'], reporte_dict['1']['f1-score']
    ],
    'Clase': ['Éxito (0)', 'Éxito (0)', 'Éxito (0)', 'Riesgo (1)', 'Riesgo (1)', 'Riesgo (1)']
}

df_metricas = pd.DataFrame(datos_metricas)

plt.figure(figsize=(10, 6))
# ////////////////////////////// Verde para Éxito, Rojo para Riesgo
sns.barplot(x='Métrica', y='Valor', hue='Clase', data=df_metricas, palette=['#2ECC71', '#E74C3C'])

# Añadimos el Accuracy general en el título
accuracy_general = reporte_dict['accuracy']
plt.title(f'Rendimiento del Modelo (Accuracy General: {accuracy_general:.2%})', fontsize=15)
plt.ylim(0, 1.1) # Límite del eje Y hasta 1.1
plt.ylabel('Puntuación (0 a 1)', fontsize=12)
plt.xlabel('Métricas de Evaluación', fontsize=12)
plt.legend(title='Estado del Estudiante', loc='lower right')

# Poner los valores exactos encima de las barras
for p in plt.gca().patches:
    plt.gca().annotate(f"{p.get_height():.2f}",
                       (p.get_x() + p.get_width() / 2., p.get_height()),
                       ha='center', va='center',
                       xytext=(0, 9),
                       textcoords='offset points')

plt.tight_layout()
plt.show()

# 1. Extraer la importancia calculada por el Random Forest
importancias = modelo_rf.feature_importances_

# 2. Emparejar cada puntuación con el nombre de su columna (de la variable X)
df_importancias = pd.DataFrame({
    'Variable': X.columns,
    'Importancia': importancias
})

# 3. Ordenar de mayor a menor y quedarnos con el Top 10
top_10_variables = df_importancias.sort_values(by='Importancia', ascending=False).head(10)

# 4. Graficar los resultados
plt.figure(figsize=(10, 6))
sns.barplot(x='Importancia', y='Variable', data=top_10_variables, palette='magma')

plt.title('Top 10 Señales de Alerta de Burnout (Días 0 a 90)', fontsize=14)
plt.xlabel('Peso de la Variable en la Decisión del Modelo', fontsize=12)
plt.ylabel('Característica (Comportamiento/Rendimiento)', fontsize=12)
plt.tight_layout()
plt.show()

print("Top 10 Variables más importantes:")
print(top_10_variables)
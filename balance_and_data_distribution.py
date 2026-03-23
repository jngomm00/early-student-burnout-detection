import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt



# 1. Cargar el dataset limpio
df = pd.read_csv('./data-clean/master_student_data.csv')

# 2. Gráfico de barras del Target
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='target', palette=['#2ECC71', '#E74C3C'])
plt.title('Distribución de Estudiantes (0: Éxito, 1: Riesgo/Burnout)')
plt.xlabel('Estado del Estudiante')
plt.ylabel('Cantidad')
plt.xticks([0, 1], ['Éxito (0)', 'Riesgo (1)'])
plt.show()

# Ver los números exactos
print("Conteo exacto por clase:")
print(df['target'].value_counts())
print("\nPorcentaje:")
print(df['target'].value_counts(normalize=True) * 100)



# 1. Identificar las columnas de las semanas clicks_w1, clicks_w2, etc...

week_columns = [col for col in df.columns if str(col).startswith('clicks_w')]

# 2. Agrupar por el target y calcular el promedio de clics por semana
trend_data = df.groupby('target')[week_columns].mean().T

# 3. Crear el gráfico de líneas
plt.figure(figsize=(10, 6))

# Línea de los que tienen éxito
plt.plot(trend_data.index, trend_data[0], marker='o', color='#2ECC71', linewidth=2, label='Éxito (0)')

# Línea de los que están en riesgo/burnout
plt.plot(trend_data.index, trend_data[1], marker='x', color='#E74C3C', linewidth=2, label='Riesgo/Burnout (1)')

plt.title('Tendencia Promedio de Clics Semanales (Días 0 a 90)', fontsize=14)
plt.xlabel('Semanas del Curso', fontsize=12)
plt.ylabel('Promedio de Clics', fontsize=12)
plt.xticks(rotation=45)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
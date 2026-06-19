import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Cargar los datos (ajusta la ruta a tu archivo studentInfo.csv)
# Utilizo la ruta que has manejado en scripts anteriores
df_info = pd.read_csv('./../../dataset/oulad/studentInfo.csv')

# 2. Crear una nueva columna agrupando si terminaron o abandonaron
# "Withdrawn" = No acabó. El resto (Pass, Distinction, Fail) = Sí acabó.
df_info['estado_finalizacion'] = df_info['final_result'].apply(
    lambda x: 'Abandonó (Withdrawn)' if x == 'Withdrawn' else 'Finalizó el Curso'
)

# 3. Configurar el lienzo y el estilo
plt.figure(figsize=(8, 6))
sns.set_theme(style="whitegrid")

# 4. Generar el gráfico de barras
# Asignamos rojo al abandono y verde a la finalización para mantener la coherencia visual
ax = sns.countplot(
    data=df_info,
    x='estado_finalizacion',
    palette=['#E74C3C', '#2ECC71'],
    order=['Abandonó (Withdrawn)', 'Finalizó el Curso']
)

# 5. Personalizar textos y etiquetas
plt.title('Proporción de Estudiantes: Finalización vs Abandono', fontsize=15, pad=15)
plt.xlabel('Estado del Estudiante', fontsize=12)
plt.ylabel('Cantidad de Estudiantes', fontsize=12)

# 6. Añadir los números exactos encima de cada barra
for p in ax.patches:
    altura = p.get_height()
    ax.annotate(f'{int(altura)}',
                (p.get_x() + p.get_width() / 2., altura),
                ha='center', va='center',
                xytext=(0, 8),
                textcoords='offset points',
                fontsize=12, fontweight='bold')

plt.tight_layout()
plt.show()

# 7. Mostrar también los datos precisos por consola
print("--- DESGLOSE DE FINALIZACIÓN ---")
conteo = df_info['estado_finalizacion'].value_counts()
porcentajes = df_info['estado_finalizacion'].value_counts(normalize=True) * 100

resumen = pd.DataFrame({
    'Cantidad': conteo,
    'Porcentaje (%)': porcentajes.round(2)
})
print(resumen)
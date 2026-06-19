# merging_data_by_weeks.py

import pandas as pd

# 1. CARGA DE DATOS (Filtrados previamente a 90 días)
df_info = pd.read_csv('./../../dataset/oulad/raw/studentInfo_cleaned.csv')
df_vle = pd.read_csv('./../../dataset/oulad/raw/studentVle_90days.csv')
df_assess_90 = pd.read_csv('./../../dataset/oulad/raw/studentAssessment_90days.csv')



print("Archivos cargados. Iniciando transformación...")

# /**/            PROCESAMIENTO DE CLICS (COMPORTAMIENTO)
# Creamos la columna de semanas: 0-6=W1, 7-13=W2...
df_vle['week'] = (df_vle['date'] // 7) + 1

# Pivotamos: de formato largo a formato ancho (una columna por semana)
vle_weekly = df_vle.pivot_table(
    index=['id_student', 'code_module', 'code_presentation'],
    columns='week',
    values='sum_click',
    aggfunc='sum',
    fill_value=0
).reset_index()

# Renombrar columnas: week 1 -> clicks_w1
vle_weekly.columns = [f'clicks_w{c}' if isinstance(c, (int, float)) else c for c in vle_weekly.columns]

# 1. Convertir la columna 'score' a numérico (los caracteres extraños se vuelven NaN)
df_assess_90['score'] = pd.to_numeric(df_assess_90['score'], errors='coerce')
# PROCESAMIENTO DE NOTAS
# Calculamos promedio de notas y total de tareas entregadas hasta el día 90
assess_summary = df_assess_90.groupby(['id_student']).agg(
    avg_score=('score', 'mean'),
    total_assessments=('id_assessment', 'count')
).reset_index()

#  FINAL (EL DATAFRAME final )
# Unimos Info con Comportamiento
master_df = pd.merge(df_info, vle_weekly, on=['id_student', 'code_module', 'code_presentation'], how='left')

# Unimos con Notas
master_df = pd.merge(master_df, assess_summary, on='id_student', how='left')

# LIMPIEZA POST-UNIÓN
# Si un estudiante no tiene clics o notas en 90 días, los NaNs deben ser 0
# (Es un indicador fuerte de desapego/burnout)
master_df.fillna(0, inplace=True)

# Guardamos el dataset listo para el modelo
master_df.to_csv('./../../dataset/oulad/raw/master_student_data.csv', index=False)

print("\n¡Tabla Maestra creada con éxito!")
print(f"Dimensiones finales: {master_df.shape}")
print(master_df.head())
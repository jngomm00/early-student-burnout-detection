import pandas as pd

# Cargar datos
df_info = pd.read_csv('./data/studentInfo.csv')

# 1. Eliminar columnas sensibles / sociodemográficas
cols_to_drop = ['gender', 'region', 'highest_education', 'imd_band', 'age_band', 'disability']
df_info_clean = df_info.drop(columns=cols_to_drop)

# 2. Crear la variable objetivo 'target' (1: Riesgo, 0: Éxito)
# Consideramos burnout/riesgo a quienes abandonan o suspenden
df_info_clean['target'] = df_info_clean['final_result'].map({
    'Withdrawn': 1,
    'Fail': 1,
    'Pass': 0,
    'Distinction': 0
})

# Guardar
df_info_clean.to_csv('./data-clean/studentInfo_cleaned.csv', index=False)
print("Paso 1 completado: studentInfo_cleaned.csv creado.")

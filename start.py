import pandas as pd

# 1. Cargar la info de los estudiantes
df_info = pd.read_csv('./data/studentInfo.csv')

# 2. Cargar el registro de clics
df_vle = pd.read_csv('./data/studentVle.csv')

print(f"Total de registros de actividad: {len(df_vle)}")
print(f"Estudiantes únicos registrados: {df_info['id_student'].nunique()}")

# Echa un vistazo a la columna objetivo
print("\nDistribución de resultados:")
print(df_info['final_result'].value_counts())

import pandas as pd
# Cargar el gigante (puede tardar unos segundos)
df_vle = pd.read_csv('./data/studentVle.csv')

# Filtrar por nuestra ventana de 90 días
df_vle_90 = df_vle[df_vle['date'] <= 90]

# Guardar versión reducida
df_vle_90.to_csv('./data-clean/studentVle_90days.csv', index=False)
print(f"Paso 2 completado: De {len(df_vle)} registros bajamos a {len(df_vle_90)}.")
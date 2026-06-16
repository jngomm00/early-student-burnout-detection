
#list.py

import pandas as pd

def estadisticas_modulo_2013(ruta_archivo='./../../dataset/oulad/studentInfo.csv'):
    # Cargar los datos
    df = pd.read_csv(ruta_archivo)

    # Filtrar solo las presentaciones de 2013
    mask_2013 = df['code_presentation'].str.contains('2013', na=False)
    df_2013 = df[mask_2013].copy()

    # Crear una columna booleana que vale True si es abandono y False si no lo es
    df_2013['es_abandono'] = df_2013['final_result'] == 'Withdrawn'

    # Agrupar por módulo y calcular el total de alumnos y la suma de abandonos
    resultado = df_2013.groupby('code_module').agg(
        total_alumnos=('id_student', 'count'),
        num_abandonos=('es_abandono', 'sum')
    ).reset_index()

    # Calcular los alumnos que no abandonaron (el opuesto a Withdrawn)
    resultado['num_no_abandonos'] = resultado['total_alumnos'] - resultado['num_abandonos']

    # Calcular el ratio: withdrawn / no abandono
    resultado['ratio_abandono_vs_no'] = (resultado['num_abandonos'] / resultado['num_no_abandonos']).round(2)

    # Calcular los porcentajes
    resultado['pct_abandono'] = ((resultado['num_abandonos'] / resultado['total_alumnos']) * 100).round(2)
    resultado['pct_no_abandono'] = ((resultado['num_no_abandonos'] / resultado['total_alumnos']) * 100).round(2)

    # Reordenar las columnas para mostrar la información solicitada
    resultado = resultado[[
        'code_module',
        'num_abandonos',
        'num_no_abandonos',
        'total_alumnos',
        'ratio_abandono_vs_no',
        'pct_abandono',
        'pct_no_abandono'
    ]]

    # Ordenar por el número de abandonos de mayor a menor
    resultado = resultado.sort_values(by='num_abandonos', ascending=False).reset_index(drop=True)

    return resultado


if __name__ == '__main__':
    try:
        df_resultados = estadisticas_modulo_2013()
        print(df_resultados.to_string(index=False))
    except FileNotFoundError:
        print("Error: El archivo 'studentInfo.csv' no se encuentra en la ruta especificada.")
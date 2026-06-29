import pandas as pd
import numpy as np
import os

def crear_dataset_enriquecido_completo_60d(ruta_base_datos, ruta_salida):
    print("Iniciando generación del dataset enriquecido a 60 días (con variables sociodemográficas)...")

    print("Cargando archivos CSV...")
    info_df = pd.read_csv(os.path.join(ruta_base_datos, 'studentInfo.csv'))
    vle_df = pd.read_csv(os.path.join(ruta_base_datos, 'studentVle.csv'))
    assessments = pd.read_csv(os.path.join(ruta_base_datos, 'assessments.csv'))
    student_assessment = pd.read_csv(os.path.join(ruta_base_datos, 'studentAssessment.csv'))
    registration = pd.read_csv(os.path.join(ruta_base_datos, 'studentRegistration.csv'))

    modulos_objetivo = ['BBB', 'DDD', 'FFF']
    info_clean = info_df[info_df['code_module'].isin(modulos_objetivo)].copy()

    print("Procesando interacciones (VLE) para los primeros 60 días...")
    vle_60 = vle_df[vle_df['date'] <= 60].copy()

    vle_60['week'] = (vle_60['date'] // 7).clip(lower=0)

    vle_diario = vle_60.groupby(['id_student', 'code_module', 'code_presentation', 'week', 'date'])[
        'sum_click'].sum().reset_index()

    max_clicks_dia = vle_diario.groupby(['id_student', 'code_module', 'code_presentation'])['sum_click'].max().rename(
        'max_clicks_1_dia').reset_index()

    vle_semanal = vle_diario.groupby(['id_student', 'code_module', 'code_presentation', 'week']).agg(
        clicks_semanales=('sum_click', 'sum'),
        dias_activos_semana=('date', 'nunique')
    ).reset_index()

    print("Generando columnas individuales por semana...")
    vle_semanas_pivot = vle_semanal.pivot_table(
        index=['id_student', 'code_module', 'code_presentation'],
        columns='week',
        values='clicks_semanales',
        fill_value=0
    ).reset_index()

    nombres_semanas = {col: f'clicks_semana_{int(col)}' for col in vle_semanas_pivot.columns if
                       isinstance(col, (int, float))}
    vle_semanas_pivot = vle_semanas_pivot.rename(columns=nombres_semanas)
    lista_columnas_semanales = list(nombres_semanas.values())

    vle_features = vle_semanal.groupby(['id_student', 'code_module', 'code_presentation']).agg(
        total_clicks_60d=('clicks_semanales', 'sum'),
        media_clicks_semanales=('clicks_semanales', 'mean'),
        total_dias_activos=('dias_activos_semana', 'sum'),
        semanas_con_actividad=('week', 'nunique'),
        semanas_actividad_plena=('dias_activos_semana', lambda x: (x >= 5).sum())
    ).reset_index()

    TOTAL_SEMANAS_60D = 9 # Los 60 días abarcan de la semana 0 a la 8 inclusive
    vle_features['semanas_sin_clicks'] = TOTAL_SEMANAS_60D - vle_features['semanas_con_actividad']
    vle_features['dias_sin_clicks_60d'] = 60 - vle_features['total_dias_activos']

    vle_features = pd.merge(vle_features, max_clicks_dia, on=['id_student', 'code_module', 'code_presentation'], how='left')
    vle_features = pd.merge(vle_features, vle_semanas_pivot, on=['id_student', 'code_module', 'code_presentation'], how='left')

    print("Procesando evaluaciones y retrasos...")
    assessments['date'] = pd.to_numeric(assessments['date'].replace('?', np.nan))
    assessments_60 = assessments[assessments['date'] <= 60].copy()
    student_ass_60 = pd.merge(student_assessment, assessments_60, on='id_assessment', how='inner')

    student_ass_60['date_submitted'] = pd.to_numeric(student_ass_60['date_submitted'], errors='coerce')
    student_ass_60['retraso_dias'] = student_ass_60['date_submitted'] - student_ass_60['date']
    student_ass_60['es_entrega_tardia'] = (student_ass_60['retraso_dias'] > 0).astype(int)
    student_ass_60['score'] = pd.to_numeric(student_ass_60['score'], errors='coerce')

    ass_features = student_ass_60.groupby(['id_student', 'code_module', 'code_presentation']).agg(
        entregas_realizadas_60d=('id_assessment', 'count'),
        nota_media_60d=('score', 'mean'),
        retraso_medio_dias=('retraso_dias', 'mean'),
        total_entregas_tardias=('es_entrega_tardia', 'sum')
    ).reset_index()

    print("Procesando fechas de registro...")
    reg_features = registration[['id_student', 'code_module', 'code_presentation', 'date_registration']].copy()
    reg_features['date_registration'] = pd.to_numeric(reg_features['date_registration'].replace('?', np.nan))

    print("Consolidando matriz de características...")
    df_master = info_clean.copy()

    df_master = pd.merge(df_master, vle_features, on=['id_student', 'code_module', 'code_presentation'], how='left')
    df_master = pd.merge(df_master, ass_features, on=['id_student', 'code_module', 'code_presentation'], how='left')
    df_master = pd.merge(df_master, reg_features, on=['id_student', 'code_module', 'code_presentation'], how='left')

    columnas_a_rellenar = [
        'total_clicks_60d', 'media_clicks_semanales', 'total_dias_activos',
        'semanas_con_actividad', 'semanas_actividad_plena', 'max_clicks_1_dia',
        'entregas_realizadas_60d', 'nota_media_60d', 'retraso_medio_dias', 'total_entregas_tardias'
    ] + lista_columnas_semanales

    df_master[columnas_a_rellenar] = df_master[columnas_a_rellenar].fillna(0)

    df_master['semanas_sin_clicks'] = df_master['semanas_sin_clicks'].fillna(TOTAL_SEMANAS_60D)
    df_master['dias_sin_clicks_60d'] = df_master['dias_sin_clicks_60d'].fillna(60)

    df_master['target_burnout'] = df_master['final_result'].apply(lambda x: 1 if x == 'Withdrawn' else 0)
    df_master.drop(columns=['final_result', 'id_student'], inplace=True)

    print("Exportando archivos CSV...")
    if not os.path.exists(ruta_salida):
        os.makedirs(ruta_salida)

    df_train = df_master[df_master['code_presentation'].str.contains('2013')].copy()
    df_test = df_master[df_master['code_presentation'].str.contains('2014')].copy()

    df_train.drop(columns=['code_module', 'code_presentation'], inplace=True)
    df_test.drop(columns=['code_module', 'code_presentation'], inplace=True)

    archivo_train = os.path.join(ruta_salida, 'dataset_train_2013_full_60d.csv')
    archivo_test = os.path.join(ruta_salida, 'dataset_test_2014_full_60d.csv')

    df_train.to_csv(archivo_train, index=False)
    df_test.to_csv(archivo_test, index=False)

    print("-" * 50)
    print(f"Dataset de Entrenamiento a 60 días (2013) generado: {len(df_train)} filas.")
    print(f"Dataset de Prueba a 60 días (2014) generado: {len(df_test)} filas.")
    print(f"Columnas resultantes: {df_master.shape[1]}")
    print("Proceso finalizado con éxito.")


if __name__ == "__main__":
    RUTA_ORIGEN = './../../dataset/oulad/'
    RUTA_DESTINO = './../../dataset/oulad/generated'

    crear_dataset_enriquecido_completo_60d(RUTA_ORIGEN, RUTA_DESTINO)
import pandas as pd
import numpy as np
import os


def crear_dataset_enriquecido_30d(ruta_base_datos, ruta_salida):
    print("Iniciando generación del dataset enriquecido (Ventana: 30 días)...")

    # 1. Cargar todas las tablas necesarias
    print("Cargando archivos CSV...")
    info_df = pd.read_csv(os.path.join(ruta_base_datos, 'studentInfo.csv'))
    vle_df = pd.read_csv(os.path.join(ruta_base_datos, 'studentVle.csv'))
    assessments = pd.read_csv(os.path.join(ruta_base_datos, 'assessments.csv'))
    student_assessment = pd.read_csv(os.path.join(ruta_base_datos, 'studentAssessment.csv'))
    registration = pd.read_csv(os.path.join(ruta_base_datos, 'studentRegistration.csv'))

    # 2. Filtrado inicial y privacidad
    modulos_objetivo = ['BBB', 'DDD', 'FFF']
    info_df = info_df[info_df['code_module'].isin(modulos_objetivo)]

    columnas_sensibles = ['gender', 'region', 'imd_band', 'age_band', 'disability']
    info_clean = info_df.drop(columns=columnas_sensibles).copy()

    # 3. Procesamiento de VLE (Interacciones) - Ventana de 30 días
    print("Procesando interacciones (VLE) a 30 días...")
    vle_30 = vle_df[vle_df['date'] <= 30].copy()

    # Crear identificador de semana (semana 0 a semana 4 aprox. para 30 días)
    vle_30['week'] = (vle_30['date'] // 7).clip(lower=0)

    # Agrupación diaria para calcular métricas de días específicos
    vle_diario = vle_30.groupby(['id_student', 'code_module', 'code_presentation', 'week', 'date'])[
        'sum_click'].sum().reset_index()

    max_clicks_dia = vle_diario.groupby(['id_student', 'code_module', 'code_presentation'])['sum_click'].max().rename(
        'max_clicks_1_dia').reset_index()

    # Agrupación semanal
    vle_semanal = vle_diario.groupby(['id_student', 'code_module', 'code_presentation', 'week']).agg(
        clicks_semanales=('sum_click', 'sum'),
        dias_activos_semana=('date', 'nunique')
    ).reset_index()

    # Pivotar las columnas en semanas independientes
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

    # Guardamos los nombres para rellenar nulos más adelante
    lista_columnas_semanales = list(nombres_semanas.values())
    # -------------------------------------------------------------

    # Agrupación total del estudiante
    vle_features = vle_semanal.groupby(['id_student', 'code_module', 'code_presentation']).agg(
        total_clicks_30d=('clicks_semanales', 'sum'),
        media_clicks_semanales=('clicks_semanales', 'mean'),
        total_dias_activos=('dias_activos_semana', 'sum'),
        semanas_con_actividad=('week', 'nunique'),
        semanas_actividad_plena=('dias_activos_semana', lambda x: (x >= 5).sum())
    ).reset_index()

    # Cálculos derivados de inactividad a 30 días (Semanas 0, 1, 2, 3, 4 = 5 semanas)
    TOTAL_SEMANAS_30D = 5
    vle_features['semanas_sin_clicks'] = TOTAL_SEMANAS_30D - vle_features['semanas_con_actividad']
    vle_features['dias_sin_clicks_30d'] = 30 - vle_features['total_dias_activos']

    # Unir máximo de clicks y las columnas de cada semana
    vle_features = pd.merge(vle_features, max_clicks_dia, on=['id_student', 'code_module', 'code_presentation'],
                            how='left')
    vle_features = pd.merge(vle_features, vle_semanas_pivot, on=['id_student', 'code_module', 'code_presentation'],
                            how='left')

    # 4. Procesamiento de Evaluaciones (Assessments) - Ventana de 30 días
    print("Procesando evaluaciones y retrasos a 30 días...")
    assessments['date'] = pd.to_numeric(assessments['date'].replace('?', np.nan))
    assessments_30 = assessments[assessments['date'] <= 30].copy()
    student_ass_30 = pd.merge(student_assessment, assessments_30, on='id_assessment', how='inner')

    student_ass_30['date_submitted'] = pd.to_numeric(student_ass_30['date_submitted'], errors='coerce')
    student_ass_30['retraso_dias'] = student_ass_30['date_submitted'] - student_ass_30['date']
    student_ass_30['es_entrega_tardia'] = (student_ass_30['retraso_dias'] > 0).astype(int)
    student_ass_30['score'] = pd.to_numeric(student_ass_30['score'], errors='coerce')

    ass_features = student_ass_30.groupby(['id_student', 'code_module', 'code_presentation']).agg(
        entregas_realizadas_30d=('id_assessment', 'count'),
        nota_media_30d=('score', 'mean'),
        retraso_medio_dias=('retraso_dias', 'mean'),
        total_entregas_tardias=('es_entrega_tardia', 'sum')
    ).reset_index()

    # 5. Procesamiento de Registro
    print("Procesando fechas de registro...")
    reg_features = registration[['id_student', 'code_module', 'code_presentation', 'date_registration']].copy()
    reg_features['date_registration'] = pd.to_numeric(reg_features['date_registration'].replace('?', np.nan))

    # 6. Consolidación del Dataset Maestro
    print("Consolidando matriz de características...")
    df_master = info_clean.copy()

    df_master = pd.merge(df_master, vle_features, on=['id_student', 'code_module', 'code_presentation'], how='left')
    df_master = pd.merge(df_master, ass_features, on=['id_student', 'code_module', 'code_presentation'], how='left')
    df_master = pd.merge(df_master, reg_features, on=['id_student', 'code_module', 'code_presentation'], how='left')

    # Añadimos las nuevas columnas semanales a la lista de columnas que deben rellenarse con 0 si son nulas
    columnas_a_rellenar = [
                              'total_clicks_30d', 'media_clicks_semanales', 'total_dias_activos',
                              'semanas_con_actividad',
                              'semanas_actividad_plena', 'max_clicks_1_dia', 'entregas_realizadas_30d',
                              'nota_media_30d', 'retraso_medio_dias', 'total_entregas_tardias'
                          ] + lista_columnas_semanales

    df_master[columnas_a_rellenar] = df_master[columnas_a_rellenar].fillna(0)

    df_master['semanas_sin_clicks'] = df_master['semanas_sin_clicks'].fillna(TOTAL_SEMANAS_30D)
    df_master['dias_sin_clicks_30d'] = df_master['dias_sin_clicks_30d'].fillna(30)

    # 7. Creación de Target
    df_master['target_burnout'] = df_master['final_result'].apply(lambda x: 1 if x == 'Withdrawn' else 0)
    df_master.drop(columns=['final_result', 'id_student'], inplace=True)

    # 8. Partición y exportación
    print("Exportando archivos CSV...")
    if not os.path.exists(ruta_salida):
        os.makedirs(ruta_salida)

    df_train = df_master[df_master['code_presentation'].str.contains('2013')].copy()
    df_test = df_master[df_master['code_presentation'].str.contains('2014')].copy()

    df_train.drop(columns=['code_module', 'code_presentation'], inplace=True)
    df_test.drop(columns=['code_module', 'code_presentation'], inplace=True)

    # Añadimos el sufijo _30d para evitar sobrescribir el dataset original
    archivo_train = os.path.join(ruta_salida, 'dataset_train_2013_enriquecido_30d.csv')
    archivo_test = os.path.join(ruta_salida, 'dataset_test_2014_enriquecido_30d.csv')

    df_train.to_csv(archivo_train, index=False)
    df_test.to_csv(archivo_test, index=False)

    print("-" * 50)
    print(f"Dataset de Entrenamiento (2013 - 30d) generado: {len(df_train)} filas.")
    print(f"Dataset de Prueba (2014 - 30d) generado: {len(df_test)} filas.")
    print("Proceso finalizado con éxito.")


if __name__ == "__main__":
    RUTA_ORIGEN = './../../dataset/oulad/'
    RUTA_DESTINO = './../../dataset/oulad/generated'

    crear_dataset_enriquecido_30d(RUTA_ORIGEN, RUTA_DESTINO)
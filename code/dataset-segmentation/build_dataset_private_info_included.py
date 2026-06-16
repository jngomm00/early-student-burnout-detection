import pandas as pd
import numpy as np
import os

def crear_dataset_enriquecido_completo(ruta_base_datos, ruta_salida):
    print("Iniciando generación del dataset enriquecido (con variables sociodemográficas)...")

    # 1. Cargar todas las tablas necesarias
    print("Cargando archivos CSV...")
    info_df = pd.read_csv(os.path.join(ruta_base_datos, 'studentInfo.csv'))
    vle_df = pd.read_csv(os.path.join(ruta_base_datos, 'studentVle.csv'))
    assessments = pd.read_csv(os.path.join(ruta_base_datos, 'assessments.csv'))
    student_assessment = pd.read_csv(os.path.join(ruta_base_datos, 'studentAssessment.csv'))
    registration = pd.read_csv(os.path.join(ruta_base_datos, 'studentRegistration.csv'))

    # 2. Filtrado inicial (Sin eliminación de datos sensibles)
    modulos_objetivo = ['BBB', 'DDD', 'FFF']

    # Mantenemos las columnas (gender, region, imd_band, age_band, disability, etc.)
    info_clean = info_df[info_df['code_module'].isin(modulos_objetivo)].copy()

    # 3. Procesamiento de VLE (Interacciones) - Ventana de 90 días
    print("Procesando interacciones (VLE)...")
    vle_90 = vle_df[vle_df['date'] <= 90].copy()

    # Crear identificador de semana (semana 0 a semana 12 aprox. para 90 días)
    vle_90['week'] = (vle_90['date'] // 7).clip(lower=0)

    # Agrupación diaria para calcular métricas de días específicos
    vle_diario = vle_90.groupby(['id_student', 'code_module', 'code_presentation', 'week', 'date'])[
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
    lista_columnas_semanales = list(nombres_semanas.values())

    # Agrupación total del estudiante
    vle_features = vle_semanal.groupby(['id_student', 'code_module', 'code_presentation']).agg(
        total_clicks_90d=('clicks_semanales', 'sum'),
        media_clicks_semanales=('clicks_semanales', 'mean'),
        total_dias_activos=('dias_activos_semana', 'sum'),
        semanas_con_actividad=('week', 'nunique'),
        semanas_actividad_plena=('dias_activos_semana', lambda x: (x >= 5).sum())
    ).reset_index()

    # Cálculos derivados de inactividad
    TOTAL_SEMANAS_90D = 13
    vle_features['semanas_sin_clicks'] = TOTAL_SEMANAS_90D - vle_features['semanas_con_actividad']
    vle_features['dias_sin_clicks_90d'] = 90 - vle_features['total_dias_activos']

    # Unir máximo de clicks y las columnas de cada semana
    vle_features = pd.merge(vle_features, max_clicks_dia, on=['id_student', 'code_module', 'code_presentation'],
                            how='left')
    vle_features = pd.merge(vle_features, vle_semanas_pivot, on=['id_student', 'code_module', 'code_presentation'],
                            how='left')

    # 4. Procesamiento de Evaluaciones (Assessments) - Ventana de 90 días
    print("Procesando evaluaciones y retrasos...")
    assessments['date'] = pd.to_numeric(assessments['date'].replace('?', np.nan))
    assessments_90 = assessments[assessments['date'] <= 90].copy()
    student_ass_90 = pd.merge(student_assessment, assessments_90, on='id_assessment', how='inner')

    student_ass_90['date_submitted'] = pd.to_numeric(student_ass_90['date_submitted'], errors='coerce')
    student_ass_90['retraso_dias'] = student_ass_90['date_submitted'] - student_ass_90['date']
    student_ass_90['es_entrega_tardia'] = (student_ass_90['retraso_dias'] > 0).astype(int)
    student_ass_90['score'] = pd.to_numeric(student_ass_90['score'], errors='coerce')

    ass_features = student_ass_90.groupby(['id_student', 'code_module', 'code_presentation']).agg(
        entregas_realizadas_90d=('id_assessment', 'count'),
        nota_media_90d=('score', 'mean'),
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
                              'total_clicks_90d', 'media_clicks_semanales', 'total_dias_activos',
                              'semanas_con_actividad',
                              'semanas_actividad_plena', 'max_clicks_1_dia', 'entregas_realizadas_90d',
                              'nota_media_90d', 'retraso_medio_dias', 'total_entregas_tardias'
                          ] + lista_columnas_semanales

    df_master[columnas_a_rellenar] = df_master[columnas_a_rellenar].fillna(0)

    df_master['semanas_sin_clicks'] = df_master['semanas_sin_clicks'].fillna(TOTAL_SEMANAS_90D)
    df_master['dias_sin_clicks_90d'] = df_master['dias_sin_clicks_90d'].fillna(90)

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

    # Modificación en el nombre de salida para distinguir el dataset completo
    archivo_train = os.path.join(ruta_salida, 'dataset_train_2013_full.csv')
    archivo_test = os.path.join(ruta_salida, 'dataset_test_2014_full.csv')

    df_train.to_csv(archivo_train, index=False)
    df_test.to_csv(archivo_test, index=False)

    print("-" * 50)
    print(f"Dataset de Entrenamiento Completo (2013) generado: {len(df_train)} filas.")
    print(f"Dataset de Prueba Completo (2014) generado: {len(df_test)} filas.")
    print(f"Columnas resultantes: {df_master.shape[1]}")
    print("Proceso finalizado con éxito.")


if __name__ == "__main__":
    RUTA_ORIGEN = './../../dataset/oulad/'
    RUTA_DESTINO = './../../dataset/oulad/generated'

    crear_dataset_enriquecido_completo(RUTA_ORIGEN, RUTA_DESTINO)
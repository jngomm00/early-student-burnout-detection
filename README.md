# Diseño e Implementación de un Sistema Inteligente para la Detección del Agotamiento Académico Preservando la Privacidad

Este repositorio contiene el código fuente de un Trabajo de Fin de Grado centrado en la creación de un sistema predictivo mediante Machine Learning. El objetivo principal es identificar de forma temprana a los estudiantes con riesgo de abandono (o *burnout*) **dentro de los primeros 30, 60 o 90 días del curso**.

El sistema se ha construido bajo el paradigma de **Privacy by Design (Privacidad desde el Diseño)**: basa sus predicciones exclusivamente en la huella digital y conductual del estudiante (interacción temporal con el Entorno Virtual de Aprendizaje y cumplimiento de entregas), omitiendo deliberadamente cualquier variable sociodemográfica sensible para evitar sesgos discriminatorios.

## Origen de los Datos

> **Nota Técnica:** Por motivos de optimización y límites de almacenamiento en GitHub, los archivos `.csv` originales (que superan los 400 MB) no están incluidos en este repositorio. Para reproducir los experimentos, debes descargar los datos y colocarlos en las rutas especificadas dentro del directorio `dataset/`.

Este proyecto se alimenta de dos fuentes de datos principales:

1. **OULAD (Open University Learning Analytics Dataset):** Motor principal del proyecto. Contiene registros relacionales de evaluaciones y millones de interacciones diarias (logs del VLE).
   * 🔗 **[Descargar OULAD Dataset aquí](https://analyse.kmi.open.ac.uk/open-dataset)**
   * 📂 **Ruta destino:** `dataset/oulad/raw/`

2. **Kaggle - Student Performance Dataset:** Utilizado para pruebas de estrés de generalización externa y validación de la arquitectura frente a una estructura tabular diferente.
   * 🔗 **[Descargar Dataset de Kaggle aquí](https://www.kaggle.com/datasets/nabeelqureshitiii/student-performance-dataset)**
   * 📂 **Ruta destino:** `dataset/kaggle/`

---

## Arquitectura y Estructura del Proyecto

El repositorio desacopla estrictamente los datos de la lógica de procesamiento, dividiéndose en dos grandes bloques estructurales:

### 1. Gestión de Datos (`dataset/`)
* `kaggle/`: Fuente externa para la prueba empírica de contraste.
* `oulad/raw/`: Directorio para los archivos CSV originales en bruto.
* `oulad/generated/`: Directorio de salida del pipeline de datos. Almacena las matrices finales, enriquecidas mediante *Feature Engineering* y divididas por ventanas temporales de observación (`_30d`, `_60d`, `full`) y por cohortes cronológicas (2013 para entrenamiento, 2014 para test).

### 2. Código Fuente (`code/`)
La lógica de la aplicación se fragmenta en módulos y notebooks independientes:

* **`dataset-segmentation/` (Motor de ingesta y transformación):**
  * `dropout_statistics.py`: Análisis estadístico para aislar los módulos curriculares más representativos.
  * `build_dataset_private_info_included.py` (y variantes de 30/60 días): Orquestan la carga relacional, el cálculo de nuevas variables conductuales (días inactivos, picos de clics, retraso en entregas) y la exportación al directorio `generated/`.

* **`cp/` (Casos de Prueba y Modelado):**
  * Jupyter Notebooks secuenciales (de `cp01` a `cp14`) organizados en subdirectorios por algoritmo (XGBoost, Random Forest, SVM, MLP, KNN, Regresión Logística, Naive Bayes).
  * Incluyen la implementación de técnicas de remuestreo para mitigar el desbalanceo de clases (**ADASYN** y **Undersampling**).
  * Optimizaciones multiobjetivo de hiperparámetros mediante `GridSearchCV`.
  * Notebooks dedicados al análisis de explicabilidad e importancia de variables utilizando **SHAP**.

---

## Stack Tecnológico Principal

* **Lenguaje:** Python
* **Manipulación de datos:** `pandas`, `numpy`
* **Machine Learning:** `scikit-learn`, `xgboost`, `imbalanced-learn`
* **Explicabilidad (XAI):** `shap`
* **Visualización:** `matplotlib`, `seaborn`

---

## Ejecución

Para replicar el entorno y ejecutar los pipelines:
1. Asegúrate de colocar los archivos de datos en `dataset/oulad/raw/`.
2. Ejecuta los scripts de `code/dataset-segmentation/` para generar las matrices enriquecidas.
3. Abre y ejecuta secuencialmente los Jupyter Notebooks ubicados en `code/cp/` para reproducir el entrenamiento, la optimización y la evaluación de los distintos modelos.


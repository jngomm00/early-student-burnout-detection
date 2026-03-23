# Predicción Temprana de Abandono y Burnout Estudiantil con Machine Learning

Este repositorio contiene el código fuente de un trabajo centrado en la creación de un sistema de alerta temprana mediante Machine Learning. El objetivo principal es identificar a los estudiantes con alto riesgo de abandonar sus estudios (o sufrir *burnout*) **dentro de los primeros 90 días del curso**, basándose en su interacción temporal con el Entorno Virtual de Aprendizaje (VLE) y sus resultados preliminares.

## Origen de los Datos

> **Nota Técnica:** Por motivos de optimización y límites de peso en GitHub, los archivos `.csv` originales (que superan los 400 MB) no están incluidos en este repositorio. Para reproducir el código, debes descargar los datos desde los siguientes enlaces y colocarlos en las carpetas `data/` y `kaggle/`.

Este proyecto se alimenta de dos fuentes de datos principales:

1. **OULAD (Open University Learning Analytics Dataset):** Es el motor principal del proyecto. Contiene datos demográficos y registros de clics diarios (VLE) de estudiantes reales.
   * 🔗 **[Descargar OULAD Dataset aquí](https://analyse.kmi.open.ac.uk/open_dataset)**

2. **Kaggle - Student Performance Dataset:** Utilizado para pruebas de contraste y experimentación secundaria con modelos de predicción de calificaciones.
   * 🔗 **[Descargar Dataset de Kaggle aquí](PON_AQUI_EL_ENLACE_EXACTO_DE_KAGGLE)**

---

## Estructura del Proyecto y Pipeline de Datos

El proyecto está diseñado de forma modular, separando la limpieza de datos del entrenamiento de los modelos:

* **Limpieza y Preprocesamiento:**
  * `cleaning_student_info_dataset.py`, `cleaning_student_assesment.py`, `cleaning_student_vle.py`: Scripts encargados de limpiar nulos, normalizar variables categóricas y preparar los datos en bruto.
  * `merging_data_by_weeks.py`: Agrupa los millones de clics de los estudiantes en ventanas temporales (ej. primeros 90 días) para crear series temporales de comportamiento.
* **Análisis y Modelado:**
  * `balance_and_data_distribution.py`: Análisis exploratorio para lidiar con el desbalanceo de clases (la mayoría de estudiantes aprueba, pocos abandonan).
  * `training.py` / `training_various_models.py`: Scripts para entrenar, evaluar y comparar diferentes algoritmos (Random Forest, Logistic Regression, etc.).
* **Ejecución:**
  * `training.py` / `training_various_models.py`: Puntos de entrada para ejecutar el pipeline completo.

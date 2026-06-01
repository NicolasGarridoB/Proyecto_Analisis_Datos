# Predicción de Éxito en Apps de Google Play Store

Este proyecto desarrolla una solución integral de Ciencia de Datos y Machine Learning utilizando el conjunto de datos de aplicaciones de la Google Play Store. El flujo de trabajo abarca desde la adquisición y una limpieza profunda de datos "crudos", hasta la implementación de modelos predictivos supervisados y técnicas de aprendizaje no supervisado.

## 🎯 Objetivo del Proyecto
El objetivo principal es predecir si una aplicación alcanzará un rendimiento o calificación ("Rating") alto en la tienda, analizando variables clave como su categoría, cantidad de reseñas, volumen de instalaciones, tipo (gratis/pago), precio y clasificación de contenido.

---

## 📂 Estructura del Proyecto

El desarrollo se divide en dos fases principales contenidas en los Jupyter Notebooks:

### 1. Fase 1: Análisis Estructurado y Limpieza de Datos
* **Archivo:** `01_analisis_y_limpieza_googleplay.ipynb`
* **Descripción:** Transformación de datos en bruto (con texto, símbolos e inconsistencias) en un formato apto para modelos analíticos.
* **Hitos clave:**
  * Descarga automatizada del dataset original mediante la API de `kagglehub`.
  * Eliminación de caracteres especiales en campos numéricos (ej. remoción de `+` y `,` en instalaciones, símbolos monetarios en precios).
  * Tratamiento estratégico de valores ausentes (nulos) y registros duplicados.
  * Estandarización preliminar de magnitudes para comparaciones equitativas.
  * Exportación del dataset limpio (`googleplaystore_limpio.csv`).

### 2. Fase 2: Modelamiento Avanzado y Machine Learning (EV2)
* **Archivo:** `EV_PARCIAL_2_SCY1101_300D_R_Cuadrado (1)(1).ipynb`
* **Descripción:** Implementación de pipelines de entrenamiento para clasificación y segmentación de datos.
* **Hitos clave:**
  * **Preprocesamiento Automatizado:** Uso de `ColumnTransformer` y `Pipeline` de Scikit-Learn para aplicar `OneHotEncoder` a variables categóricas y `StandardScaler` a numéricas de manera controlada y evitar el data leakage.
  * **Modelos Supervisados:** Implementación y evaluación de algoritmos de clasificación:
    * Regresión Logística (`LogisticRegression`)
    * Árboles de Decisión (`DecisionTreeClassifier`)
    * Bosques Aleatorios (`RandomForestClassifier`)
  * **Optimización:** Búsqueda de hiperparámetros mediante `GridSearchCV` y `RandomizedSearchCV` con validación cruzada.
  * **Modelos No Supervisados:** Análisis exploratorio mediante reducción de dimensionalidad con PCA y agrupamiento clúster utilizando el algoritmo K-Means.

---

## 📊 Conclusiones del Estudio

* **Calidad de los Datos:** El proceso de curación inicial garantizó un dataset final consistente con 10,346 registros libres de duplicados y valores nulos, listo para producción.
* **Rendimiento de Modelos:** Al evaluar las métricas competitivas, **Random Forest** demostró una capacidad superior frente a la Regresión Logística. Esto se debe a su habilidad para capturar relaciones no lineales complejas entre características como el volumen de reseñas (*Reviews*) y las instalaciones (*Installs*).
* **Optimización:** Aunque el ajuste de hiperparámetros mediante `GridSearchCV` incrementó ligeramente el *Accuracy* global, el análisis detallado del *F1-score* en las clases minoritarias demostró la importancia de evaluar los modelos de forma multidimensional y no basarse en una sola métrica.

---

## 🛠️ Tecnologías y Librerías Utilizadas

El proyecto fue desarrollado en entornos Python 3 utilizando las siguientes librerías core:

* **Manipulación de Datos:** `pandas`, `numpy`
* **Descarga de Datos:** `kagglehub`
* **Machine Learning & Preprocesamiento:** `scikit-learn`
* **Visualización de Datos:** `matplotlib`, `seaborn`

---

## 🚀 Cómo Ejecutar el Proyecto

1. Clonar este repositorio:
   ```bash
   git clone [https://github.com/TU_USUARIO/TU_REPOSITORIO.git](https://github.com/TU_USUARIO/TU_REPOSITORIO.git)

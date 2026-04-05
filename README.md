# Análisis y Limpieza: Google Play Store Dataset

Este repositorio contiene un pipeline completo de procesamiento de datos sobre el dataset de aplicaciones de Google Play Store. El objetivo es transformar datos crudos con inconsistencias en un dataset limpio y ordenado.

## Estructura del Proyecto

- `notebooks/`: Contiene el archivo `01_analisis_y_limpieza_googleplay.ipynb` con el análisis paso a paso.
- `data/`: 
    - `googleplaystore.csv`: Dataset original.
    - `googleplaystore_limpio.csv`: Dataset resultante tras la limpieza y normalización.
- `src/`: Scripts de Python (`cleaning.py`) con funciones modulares para el tratamiento de caracteres y nulos.
- `outputs/`: Visualizaciones generadas.
- `docs/`: Informe técnico detallado sobre las decisiones tomadas durante el preprocesamiento.


## Procesamiento de Datos Realizado
1. **Limpieza de Símbolos**: Remoción de `$`, `+` y `,` en las columnas de Precio e Instalaciones.
2. **Imputación de Nulos**: Se utilizó la **mediana** para completar los Ratings faltantes, asegurando robustez frente a valores atípicos.
3. **Normalización**: Uso de `StandardScaler` para equilibrar magnitudes numéricas.
4. **Encoding**: Transformación de variables categóricas (Categoría, Género) a formato numérico.

## Cómo ejecutar
1. Clona este repositorio o descarga las carpetas.
2. Asegúrate de tener Python instalado y ejecuta:
   ```bash
   pip install -r requirements.txt
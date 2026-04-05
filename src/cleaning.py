import pandas as pd
import numpy as np

def limpiar_caracteres_especiales(df):
    """
    Elimina símbolos como '$', '+', y ',' de las columnas Price e Installs
    y las convierte a tipos numéricos.
    """
    # Limpiar columna Installs
    if 'Installs' in df.columns:
        df['Installs'] = df['Installs'].str.replace('+', '').str.replace(',', '')
        df['Installs'] = pd.to_numeric(df['Installs'], errors='coerce')
    
    # Limpiar columna Price
    if 'Price' in df.columns:
        df['Price'] = df['Price'].str.replace('$', '').str.replace(',', '')
        df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
    
    return df

def tratar_nulos_calificaciones(df):
    """
    Imputa los valores nulos de la columna Rating usando la mediana.
    """
    if 'Rating' in df.columns:
        mediana = df['Rating'].median()
        df['Rating'] = df['Rating'].fillna(mediana)
    return df

def preprocesar_datos(df):
    """
    Ejecuta el pipeline completo de limpieza.
    """
    df = limpiar_caracteres_especiales(df)
    df = tratar_nulos_calificaciones(df)
    # Eliminamos cualquier otra fila que haya quedado con nulos tras la conversión
    df = df.dropna()
    return df
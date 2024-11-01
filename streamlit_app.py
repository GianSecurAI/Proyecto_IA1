import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import contextlib
import zipfile
import requests

st.title("Clasificador de Neumonía en Rayos X Pediátricos")
st.write(f"**Versión de TensorFlow:** {tf.__version__}")

# Descargar y descomprimir el modelo si no existe
def download_and_extract_model():
    model_url = 'https://drive.google.com/uc?id=1-3pFQ7FQsvu-CIY6unQUSinXffXCzMwH'
    zip_path = 'mobilenet_v2_model.zip'
    extract_folder = 'extracted_files'

    # Descargar el archivo zip si no existe
    if not os.path.exists(zip_path):
        try:
            with requests.get(model_url, stream=True) as r:
                r.raise_for_status()
                with open(zip_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
            st.success("Modelo descargado correctamente.")
        except Exception as e:
            st.error(f"Error al descargar el modelo: {e}")
            return False

    # Descomprimir el archivo si no existe la carpeta
    if not os.path.exists(extract_folder):
        os.makedirs(extract_folder)

        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_folder)
        except zipfile.BadZipFile:
            st.error("El archivo descargado está corrupto.")
            return False
    
    return os.path.join(extract_folder, 'mobilenet_v2_model.keras')

# Descargar y preparar el modelo
modelo_path = download_and_extract_model()

# Verificar si el archivo del modelo existe
if not modelo_path or not os.path.exists(modelo_path):
    st.error("No se encontró el archivo del modelo.")
else:
    st.success("Archivo del modelo encontrado.")

    # Cargar el modelo preentrenado
    try:
        model = load_model(modelo_path)
        st.success("Modelo cargado correctamente.")
    except Exception as e:
        st.error(f"Error al cargar el modelo: {e}")
        model = None

    # Verificación de carga de archivo de imagen
    uploaded_file = st.file_uploader("Elige una imagen de Rayos X...", type=["jpg", "jpeg", "png"], label_visibility="hidden")
    
    if uploaded_file is not None and model is not None:
        # Mostrar la imagen subida
        st.image(uploaded_file, width=300, caption="Imagen cargada")
    
        # Preprocesamiento de la imagen para hacer la predicción
        img = image.load_img(uploaded_file, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0
    
        # Realizar la predicción con redirección de salida para evitar UnicodeEncodeError
        with open(os.devnull, 'w') as devnull:
            with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
                prediction = model.predict(img_array)
    
        # Mostrar resultados
        st.success('**Neumonía Detectada**.' if prediction[0][0] > 0.5 else '**No se Detecta Neumonía**.')

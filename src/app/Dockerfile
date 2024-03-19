# Usa la imagen base de Python
FROM python:3.11-slim

# Copia el archivo requirements.txt al contenedor
COPY requirements.txt .

# Instala las dependencias del proyecto
RUN pip install --no-cache-dir -r requirements.txt

# Copia el contenido de la carpeta actual al contenedor en /app
COPY . /app
# Establece el directorio de trabajo en /app
WORKDIR /app
# Expone el puerto 8501
EXPOSE 8501

# Comando para ejecutar la aplicación Streamlit
ENTRYPOINT ["streamlit", "run", "app_streamlit.py", "–server.port=8501"]

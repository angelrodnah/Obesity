# Usa la imagen base de Python
FROM python:3.11-slim

# Establece el directorio de trabajo en /app
WORKDIR /app

# Copia el archivo requirements.txt al contenedor
COPY requirements.txt .

# Instala las dependencias del proyecto
RUN pip install --no-cache-dir -r requirements.txt

# Copia el contenido de la carpeta actual al contenedor en /app
COPY . .

# Expone el puerto 5000
EXPOSE 8080

# Comando para ejecutar la aplicación Streamlit
ENTRYPOINT ["streamlit", "run", "app_streamlit.py", "–server.port=8080", "–server.address=0.0.0.0"]

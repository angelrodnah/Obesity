@echo off

rem Nombre de la imagen Docker
set IMAGE_NAME=obesity

rem Construir la imagen Docker
echo Construyendo la imagen Docker...
docker build -t %IMAGE_NAME% .

rem Ejecutar el contenedor Docker
echo Ejecutando el contenedor Docker...
docker run -p 8080:8080 %IMAGE_NAME%
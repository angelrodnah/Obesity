import pickle 
import streamlit as st
import pandas as pd

# Título y subtítulo de la aplicación
titulo = '¿Obeso o no? 💪'
subtitulo = 'Predice el nivel de obesidad con aprendizaje automático'

def main():
    # Configuración de la página
    st.set_page_config(layout="wide", page_icon='💪', page_title='Obeso o No')
    st.title(titulo)
    st.write(subtitulo)

    # Formulario para ingresar los datos
    formulario = st.form("Datos de entrada")
    genero = formulario.selectbox('Género', ['Masculino', 'Femenino'])
    edad = formulario.number_input('¿Qué edad tienes?', min_value=1, max_value=100)
    antecedentes_familiares = formulario.selectbox('¿Tienes antecedentes familiares de sobrepeso?', ['Sí', 'No'])
    fumador = formulario.selectbox('¿Fumas?', ['Sí', 'No'])
    altura = formulario.number_input('Ingresa tu altura (en metros)')
    peso = formulario.number_input('Ingresa tu peso (en kilogramos)')
    comidas = formulario.selectbox('¿Cuántas comidas haces al día?', list(range(1, 11)))
    actividad_fisica = formulario.number_input('¿Cuántos días haces ejercicio a la semana?')
    enviar = formulario.form_submit_button('¡Predecir tu nivel de obesidad!')

    if enviar:
        # Recopilación de los datos ingresados
        datos = {
            'Género': genero,
            'Edad': edad,
            'Altura': altura,
            'Peso': peso,
            'Antecedentes familiares de sobrepeso': antecedentes_familiares,
            'Comidas': comidas,
            'Actividad física (días por semana)': actividad_fisica,
            'Fumador': fumador
        }

        # Preparación de los datos para la predicción
        datos = pd.Series(datos).to_frame(name=0).T
        datos['Género'] = datos['Género'].replace({'Masculino': 1, 'Femenino': 0})
        datos['Antecedentes familiares de sobrepeso'] = datos['Antecedentes familiares de sobrepeso'].replace({'Sí': 1, 'No': 0})
        datos['Fumador'] = datos['Fumador'].replace({'No': 0, 'Sí': 1})

        # Carga del modelo y realización de la predicción
        with open('model.pkl', 'rb') as f:
            modelo = pickle.load(f)
        prediccion = modelo.predict(datos)[0]

        # Mostrar el resultado de la predicción
        st.success('Tu predicción de nivel de obesidad: ' + prediccion)

if __name__ == '__main__':
    main()

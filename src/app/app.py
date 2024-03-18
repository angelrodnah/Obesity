import pandas as pd
import numpy as np
from sklearn import pipeline
import streamlit as st
import plotly.express as px
from PIL import Image
import os as so
import sys
import joblib
appdir=so.path.dirname(so.path.abspath(__file__))
so.chdir(appdir)
sys.path.append(appdir+'/utils')
from app.utils.funcs import XEncoder
from funcs import *
from values import *



imagen1=Image.open('./img/nutricion-tratar-obesidad.webp')
imagen2 = Image.open('./img/Obesity.webp')
st.set_page_config(page_title="Conoce tu riesgo de obesidad :question:", layout="wide")
st.header(":blue[Tus hábitos: ]:pizza: :hamburger: :hotdog: :cake: :arrow_right::arrow_right::arrow_right: :red[:heartbeat: ¿No estarás en riesgo de obesidad?]  :heartbeat:  :skull:")

tab_inicio, tab_conjunto_datos, tab_graficos, tab_modelo = st.tabs(["Inicio", "Sobre el Conjunto de Datos", "Gráficos", "Predicción del Nivel de Obesidad"])

# TAB INICIO
inf, video = tab_inicio.columns(2, gap="large")
inf.markdown("El objetivo principal de este proyecto es desarrollar un modelo predictivo que pueda anticipar la probabilidad de que un individuo se vuelva obeso basándose en un análisis de datos exhaustivo. \
            La obesidad está asociada con una serie de resultados adversos, incluido un mayor riesgo de enfermedades cardiovasculares, diabetes tipo 2, problemas respiratorios y articulares. \
            Además, puede afectar la salud mental, lo que lleva a un aumento de las tasas de depresión y ansiedad. Por eso, abordar y prevenir la obesidad es vital para la salud y el bienestar en general.")
inf.markdown("Usando el conjunto de datos, que abarca variables cruciales como antecedentes familiares de sobrepeso, hábitos alimenticios (incluido el consumo frecuente de alimentos altos en calorías,\
    frecuencia de consumo de verduras y consumo de alimentos entre comidas), factores de estilo de vida (hábitos de fumar, consumo diario de agua, monitoreo del consumo de calorías, \
        frecuencia de actividad física, tiempo usando dispositivos tecnológicos y consumo de alcohol), y tipo de transporte utilizado, el objetivo es descubrir patrones y correlaciones \
            que contribuyen a la obesidad. A través de la exploración de estos factores multifacéticos, esta aplicación constituye una herramienta proactiva para permitir a las personas\
                tomar decisiones informadas sobre su estilo de vida y potencialmente reducir el riesgo de obesidad en el futuro.")

inf.image(imagen1, width=200, use_column_width=True, clamp=False, channels='RGB', output_format='auto')
video_file = open('./video/Obesidad_ Un grave problema de salud pública.mp4', 'rb')
video_bytes = video_file.read()
video.video(video_bytes)

# TAB CONJUNTO DE DATOS
tab_conjunto_datos.subheader("Sobre el Conjunto de Datos")
column_inf, column_conjunto_datos  = tab_conjunto_datos.columns(2, gap="large")

column_conjunto_datos.image(imagen2, width=100, use_column_width=True, clamp=False, channels='RGB', output_format='auto')

column_inf.subheader("Conjunto de Datos")
column_conjunto_datos.subheader("Preguntas de la encuesta utilizada para la recopilación inicial de información")
column_inf.markdown(
    """
    - El conjunto de datos utilizado para el análisis se obtuvo de [Kaagle](https://www.kaggle.com/competitions/playground-series-s4e2).
    - Incluye datos para estimar los niveles de obesidad en personas de 14 a 61 años con diversos hábitos alimenticios y condiciones físicas en México, Perú y Colombia.
    - Los datos se recopilaron utilizando una encuesta en una plataforma web con 17 atributos y 2111 registros.
    - Los datos se preprocesaron, incluida la eliminación de datos faltantes y atípicos, y la normalización de datos.
    - Los atributos de hábitos alimenticios incluyen FAVC, FCVC, NCP, CAEC, CH20 y CALC.
    - Los atributos de condición física incluyen SCC, FAF, TUE y MTRANS. Otras variables incluyen Género, Edad, Altura y Peso.
    - Los registros están etiquetados con la variable de clase NObesidad, lo que permite la clasificación en 7 grupos. El proceso de etiquetado se realizó en función de la OMS y la Normatividad Mexicana. Se realizó un balance de clases utilizando el filtro SMOTE utilizando la herramienta Weka.
    - Los autores del conjunto de datos señalan que el 23% de los registros se recopilaron directamente y el 77% restante se generaron sintéticamente.
    
    Valor de los datos:
    - Estos datos presentan información de diferentes lugares como México, Perú y Colombia, y se pueden usar para construir estimaciones de los niveles de obesidad basadas en el comportamiento nutricional de varias regiones.
    - Los datos se pueden utilizar para estimar el nivel de obesidad de las personas utilizando siete categorías, lo que permite un análisis detallado del nivel de afectación de un individuo.
    - La estructura y la cantidad de datos se pueden utilizar para diferentes tareas en minería de datos como: clasificación, predicción, segmentación y asociación.
    - Los datos se pueden utilizar para construir herramientas de software para la estimación de los niveles de obesidad. Los datos pueden validar el impacto de varios factores que propician la aparición de problemas de obesidad.
    """)
# Carga de datasets
df=pd.read_pickle("./data/df.pkl")
#column_inf.write(df)
#column_inf.write(revdicts)
dff=redondea_vars(df)
dff=codifica_columnas(dff,revdicts)
dff=traduce_columnas(dff,diccionario_columnas)
column_inf.dataframe(dff)
column_conjunto_datos.dataframe(df2)

# TAB GRÁFICOS
## Gráfico 1
tab_graficos.subheader("Distribuciones de Variables Numéricas por Género")
columnas_numericas = [ 'Edad', 'Altura', 'Peso', "Indice de Masa Corporal"]
columnas_numericas_seleccionadas = tab_graficos.multiselect(label="Selecciona columnas numéricas", options=columnas_numericas[0:], default=columnas_numericas[0:])
for col in columnas_numericas_seleccionadas:
    fig1 = px.histogram(dff, x=col, marginal="box", color="Género")
    fig1.update_layout(template='plotly_dark', title_x=0.5, yaxis_title='Cuenta', xaxis_title=f"{col}", title=f"Distribución de {col}")
    tab_graficos.plotly_chart(fig1, use_container_width=True)

## Gráfico 2
tab_graficos.subheader("Correlación entre Altura y Peso")
fig2 = px.scatter(data_frame=dff,y="Altura",x="Peso",size="Indice de Masa Corporal",color="Género",trendline="ols")
fig2.update_layout(template='plotly_dark')
tab_graficos.plotly_chart(fig2, use_container_width=True)

## Gráfico 3
def grafico_cruzado(data, columna_objetivo, columna_categorica, categorias):
    fig3 = px.histogram(
        data[data[columna_objetivo].isin(categorias)],
        x=columna_objetivo,
        color=columna_categorica,
        barmode='group',
        category_orders={columna_categorica: categorias},

    )
    tab_graficos.plotly_chart(fig3, use_container_width=True)

columnas_categoricas = ['Género', 
                        'Frecuencia de Actividad Física',
                        'Consumo de Alimentos Altos en Calorías',
                        'Consumo de Verduras',
                        'Número de Comidas Principales',
                        'Comer Entre Comidas',
                        'Fumar',
                        'Consumo Diario de Agua',
                        'Monitoreo de Calorías',
                        'Consumo de Alcohol',
                        'Uso de Dispositivos Tecnológicos',
                        'Antecedentes Familiares de Sobrepeso',
                        'Preferencia de Transporte']


columna_categorica_seleccionada = tab_graficos.multiselect(label="Selecciona una variable", options=columnas_categoricas, default=["Género"])

categorias = ["Peso Insuficiente", "Peso Normal", "Nivel de Sobrepeso I", "Nivel de Sobrepeso II", "Obesidad Tipo I", "Obesidad Tipo II", "Obesidad Tipo III"]

if columna_categorica_seleccionada:
    tab_graficos.subheader(f"Distribución de Niveles de Obesidad por {columna_categorica_seleccionada[0]}")
grafico_cruzado(dff, "Nivel de Obesidad", columna_categorica_seleccionada[0], categorias)

## Gráfico 4
tab_graficos.subheader("Frecuencia de Actividad Física por Nivel de Obesidad")
fig4 = px.box(dff, x="Nivel de Obesidad", y="Frecuencia de Actividad Física", points="all")
tab_graficos.plotly_chart(fig4, use_container_width=True)

## Gráfico 5
tab_graficos.subheader("Gráfico de Dispersión de Edad e IMC")
fig5 = px.scatter(dff, x="Edad", y="Indice de Masa Corporal", color="Nivel de Obesidad")
tab_graficos.plotly_chart(fig5, use_container_width=True)

## Gráfico 6
tab_graficos.subheader("Comparación de IMC Basada en Género y Preferencia de Transporte")
transporte_seleccionado = tab_graficos.multiselect(label="Selecciona un tipo de transporte", options=dff['Preferencia de Transporte'].unique() ,default=["Transporte Público"])
df_filtrado = dff[dff['Preferencia de Transporte'].isin(transporte_seleccionado)]

fig6 = px.bar(
    df_filtrado,
    x="Nivel de Obesidad",
    y="Indice de Masa Corporal",
    color="Género",
    facet_col="Preferencia de Transporte",
)
tab_graficos.plotly_chart(fig6, use_container_width=True)

# TAB MODELO

tab_modelo.subheader("Modelo de Clasificación de Nivel de Obesidad")
columnas = list(dict_var_modelo)
#columnas_seleccionadas = tab_modelo.multiselect(label="Selecciona columnas", options=columnas[0:], default=columnas[0:])

# Página de inicio de presentación
tab_modelo.markdown("## Bienvenido a Conoce Tu Obesidad")
tab_modelo.markdown("Esta aplicación te ayudará a determinar tu riesgo de obesidad.")
tab_modelo.markdown("Por favor, ingresa la siguiente información:")

# Crear un diccionario para almacenar las respuestas
respuestas = {}

# Dividir la página en dos columnas
col1, col2 = tab_modelo.columns(2)

# Calcular la mitad del número total de preguntas
mitad_preguntas = len(preguntas) // 2

# Mostrar los selectbox y campos de entrada en dos columnas
for i, pregunta in enumerate(preguntas):
    # Determinar en qué columna mostrar la pregunta
    if i < mitad_preguntas:
        columna = col1
    else:
        columna = col2
    
    # Si la variable es 'Age', utilizar un intslider
    if variables[i] in ['Age', 'Height', 'Weight']:
        widget_key = f"{variables[i]}_{i}"
        if variables[i] in ['Height', 'Weight']:
            fformat = '%.2f'
        else:
            fformat = '%.0f'
        # Utilizar un campo de entrada numérica en lugar de un selectbox
        respuesta_numerica = columna.number_input(pregunta, format=fformat, key=widget_key)
        # Almacenar la respuesta numérica en el diccionario de respuestas
        respuestas[variables[i]] = respuesta_numerica
    else:
        # Agregar un sufijo único (por ejemplo, el índice) a la clave del widget
        widget_key = f"{variables[i]}_{i}"
        # Utilizar un selectbox para las otras opciones de respuesta
        respuesta_select = columna.selectbox(pregunta, options=opciones_respuesta[i], key=widget_key)
        # Almacenar la respuesta del selectbox en el diccionario de respuestas
        respuestas[variables[i]] = respuesta_select

# Botón para almacenar las respuestas en un DataFrame
if tab_modelo.button('Descubre tu Nivel de Obesidad'):
    # Almacenar las respuestas en un DataFrame
    df_respuestas = pd.DataFrame(respuestas, index=[0])
    tab_modelo.write('## Respuestas marcadas:')
    df_respuestas_pantalla=traduce_columnas(df_respuestas,diccionario_columnas)
    tab_modelo.dataframe(df_respuestas_pantalla)
    # Codificar las respuestas en formato numérico
    
    X_respuestas = codifica_columnas(df_respuestas,dicts)
    #Cargarpkl del pipeline
    pipeline = joblib.load('./model/encoder_pipeline.pkl')
    X=XEncoder(pipeline,X_respuestas)
    tab_modelo.write(X_respuestas.describe())
    tab_modelo.write(X)
    '''# Realizar la predicción
    y_pred = modelo.predict(X_respuestas)
    # Mostrar la predicción
    tab_modelo.write('## Predicción:')
    tab_modelo.write(y_pred[0])'''



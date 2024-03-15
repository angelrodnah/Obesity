import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import joblib
import time
from PIL import Image
import os

cdir=os.getcwd()

imagen1=Image.open('./img/nutricion-tratar-obesidad.webp')
imagen2 = Image.open('./img/Obesity.webp')
st.set_page_config(page_title="Conoce tu riesgo de obesidad :question:", layout="wide")
st.header(":blue[Conoce Tu Obesidad: ] :red[ Estás en Riesgo?]")

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

column_inf.image(imagen2, width=100, use_column_width=True, clamp=False, channels='RGB', output_format='auto')

column_conjunto_datos.markdown(
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

column_conjunto_datos.subheader("Preguntas de la encuesta utilizada para la recopilación inicial de información")
def obtener_datos2():
    df = pd.read_pickle("./data/df.pkl")
    return df

df2 = obtener_datos2()
column_conjunto_datos.dataframe(df2)

column_inf.subheader("Conjunto de Datos")
def obtener_datos():
    df = pd.read_pickle("./data/df.pkl")
    return df

df = obtener_datos()
column_inf.dataframe(df)

df["Nivel de Obesidad"] = df["NObeyesdad"]
df["Frecuencia de Actividad Física"] = df["FAF"]
df["Fumar"] = df["SMOKE"]
df["Consumo de Alimentos Altos en Calorías"] = df["FAVC"]
df["Consumo de Verduras"] = df["FCVC"]
df["Número de Comidas Principales"] = df["NCP"]
df["Comer Entre Comidas"] = df["CAEC"]
df["Consumo Diario de Agua"] = df["CH2O"]
df["Monitoreo de Calorías"] = df["SCC"]
df["Uso de Dispositivos Tecnológicos"] = df["TUE"]
df["Consumo de Alcohol"] = df["CALC"]
df["Preferencia_de_Transporte"] = df["MTRANS"]
df["Antecedentes Familiares de Sobrepeso"] = df["family_history_with_overweight"]
df["IMC"] = df["Weight"]/(df["Height"]*df["Height"])

# TAB GRÁFICOS
## Gráfico 1
tab_graficos.subheader("Distribuciones de Variables Numéricas por Género")
columnas_numericas = ["Edad", "Altura", "Peso", "IMC"]
columnas_numericas_seleccionadas = tab_graficos.multiselect(label="Selecciona columnas numéricas", options=columnas_numericas[0:], default=columnas_numericas[0:])
for col in columnas_numericas_seleccionadas:
    fig1 = px.histogram(df, x=col, marginal="box", color="Género")
    fig1.update_layout(template='plotly_dark', title_x=0.5, yaxis_title='Cuenta', xaxis_title=f"{col}", title=f"Distribución de {col}")
    tab_graficos.plotly_chart(fig1, use_container_width=True)

## Gráfico 2
tab_graficos.subheader("Correlación entre Altura y Peso")
fig2 = px.scatter(data_frame=df,y="Altura",x="Peso",size="IMC",color="Género",trendline="ols")
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
columnas_categoricas = ['Género', 'Frecuencia de Actividad Física', 'Consumo de Alimentos Altos en Calorías', 'Consumo de Verduras', 'Número de Comidas Principales', 'Comer Entre Comidas','Fumar', 'Consumo Diario de Agua', 'Monitoreo de Calorías', 'Consumo de Alcohol', 'Uso de Dispositivos Tecnológicos', 'Antecedentes Familiares de Sobrepeso', 'Preferencia_de_Transporte']
columna_categorica_seleccionada = tab_graficos.multiselect(label="Selecciona una variable", options=columnas_categoricas, default=["Género"])
categorias = ["Peso Insuficiente", "Peso Normal", "Nivel de Sobrepeso I", "Nivel de Sobrepeso II", "Obesidad Tipo I", "Obesidad Tipo II", "Obesidad Tipo III"]
if columna_categorica_seleccionada:
    tab_graficos.subheader(f"Gráfico Cruzado: Distribución de Niveles de Obesidad por {columna_categorica_seleccionada[0]}")
grafico_cruzado(df, "Nivel de Obesidad", columna_categorica_seleccionada[0], categorias)

## Gráfico 4
tab_graficos.subheader("Frecuencia de Actividad Física por Nivel de Obesidad")
fig4 = px.box(df, x="Nivel de Obesidad", y="Frecuencia de Actividad Física", points="all")
tab_graficos.plotly_chart(fig4, use_container_width=True)

## Gráfico 5
tab_graficos.subheader("Gráfico de Dispersión de Edad e IMC")
fig5 = px.scatter(df, x="Edad", y="IMC", color="Nivel de Obesidad")
tab_graficos.plotly_chart(fig5, use_container_width=True)

## Gráfico 6
tab_graficos.subheader("Comparación de IMC Basada en Género y Preferencia_de_Transporte")
transporte_seleccionado = tab_graficos.multiselect(label="Selecciona un tipo de transporte", options=df.Preferencia_de_Transporte.unique(), default=["Transporte Público"])
df_filtrado = df[df.Preferencia_de_Transporte.isin(transporte_seleccionado)]

fig6 = px.bar(
    df_filtrado,
    x="Nivel de Obesidad",
    y="IMC",
    color="Género",
    facet_col="Preferencia_de_Transporte",
    labels={"IMC": "Índice de Masa Corporal"}
)
tab_graficos.plotly_chart(fig6, use_container_width=True)

# TAB MODELO

df3 = pd.read_csv("ObesityDataSet.csv")
df3.columns = [col.upper() for col in df3.columns]
# cargando el modelo guardado
modelo = joblib.load("rta_model_deploy3.joblib")
codificador = joblib.load("onehot_encoder.joblib")

# creando la lista de opciones para el menú desplegable
opciones_GENERO = ["Selecciona una opción", "Femenino", "Masculino"]
opciones_ANTECEDENTES_FAMILIARES_CON_SOBREPESO = ["Selecciona una opción", 'No', 'Sí']
opciones_FAVC   = ["Selecciona una opción", 'No', 'Sí']
opciones_FCVC   = ["Selecciona una opción", 'Nunca', 'A veces', 'Siempre']
opciones_NCP   = ["Selecciona una opción", 'Una', 'Dos', 'Tres', 'Más de tres']
opciones_CAEC   = ["Selecciona una opción", 'No', 'A veces', 'Frecuentemente', 'Siempre']
opciones_SMOKE   = ["Selecciona una opción", 'No', 'Sí']
opciones_CH2O   = ["Selecciona una opción", 'Menos de 1L', 'Entre 1L y 2L', 'Más de 2L']
opciones_SCC   = ["Selecciona una opción", 'No', 'Sí']
opciones_FAF   = ["Selecciona una opción", 'No tengo', '1 o 2 días', '3 o 4 días', '4 o 5 días']
opciones_TUE   = ["Selecciona una opción", '0-2 horas', '3-5 horas', 'Más de 5 horas']
opciones_CALC  = ["Selecciona una opción", 'No bebo', 'A veces', 'Frecuentemente', 'Siempre']
opciones_MTRANS  = ["Selecciona una opción", 'Automóvil', 'Motocicleta', 'Bicicleta', 'Transporte Público', 'Caminar']

def principal():
    tab_modelo.subheader("Por favor, ingresa la siguiente información:")
    GENERO = tab_modelo.selectbox('¿Cuál es tu género?', options=opciones_GENERO)
    EDAD = tab_modelo.text_input('Ingresa tu edad', placeholder='Por ejemplo, 25')

    ANTECEDENTES_FAMILIARES_CON_SOBREPESO = tab_modelo.selectbox('¿Tiene un familiar que ha sufrido o sufre de sobrepeso?',
                                                         options=opciones_ANTECEDENTES_FAMILIARES_CON_SOBREPESO)
    FAVC = tab_modelo.selectbox('¿Consumes alimentos altos en calorías con frecuencia?', options=opciones_FAVC)
    FCVC = tab_modelo.selectbox('¿Sueles comer verduras en tus comidas?', options=opciones_FCVC)
    NCP = tab_modelo.selectbox('¿Cuántas comidas principales tienes al día?', options=opciones_NCP)
    CAEC = tab_modelo.selectbox('¿Comes algo entre comidas?', options=opciones_CAEC)
    SMOKE = tab_modelo.selectbox('¿Fumas?', options=opciones_SMOKE)
    CH2O = tab_modelo.selectbox('¿Cuánta agua bebes diariamente?', options=opciones_CH2O)
    SCC = tab_modelo.selectbox('¿Monitoreas las calorías que consumes diariamente?', options=opciones_SCC)
    FAF = tab_modelo.selectbox('¿Con qué frecuencia haces actividad física?', options=opciones_FAF)
    TUE = tab_modelo.selectbox(
        '¿Cuánto tiempo usas dispositivos tecnológicos como móvil, videojuegos, televisión, ordenador y otros?',
        options=opciones_TUE)
    CALC = tab_modelo.selectbox('¿Con qué frecuencia bebes alcohol?', options=opciones_CALC)
    MTRANS = tab_modelo.selectbox('¿Qué transporte usas habitualmente?', options=opciones_MTRANS)
    st.cache(allow_output_mutation=True, hash_funcs={np.ndarray: lambda x: hash(x.tobytes())})
    if tab_modelo.button('Resultado de Predicción del Nivel de Obesidad'):
        array_entrada = np.array([GENERO, ANTECEDENTES_FAMILIARES_CON_SOBREPESO, FAVC, CALC, SMOKE, SCC, CAEC, MTRANS, FCVC, NCP, CH2O, FAF, TUE],
            ndmin=2)
        matriz_codificada = codificador.transform(array_entrada)
        array_denso = matriz_codificada.toarray()
        array_codificado = list(array_denso.ravel())

        array_numérico = [EDAD]
        arreglo_prediccion = np.array(array_numérico + array_codificado, dtype=np.float64).reshape(1, -1)
        prediccion = modelo.predict(arreglo_prediccion)

        with st.spinner('Por favor espera...'):
            time.sleep(1)

        if prediction == 0:
            texto = 'Peso Insuficiente'
        elif prediction == 1:
            texto = 'Peso Normal'
        elif prediction == 2:
            texto = 'Obesidad I'
        elif prediction == 3:
            texto = 'Obesidad II'
        elif prediction == 4:
            texto = 'Obesidad Tipo III'
        elif prediction == 5:
            texto = 'Nivel de Sobrepeso I'
        elif prediction == 6:
            texto = 'Nivel de Sobrepeso II'
        else:
            texto = 'Valor de predicción inválido'
        tab_modelo.markdown(f'<p style="text-align:center; font-size:26px; font-weight:bold;">{texto}</p>',
                           unsafe_allow_html=True)

    st.write("**Desarrollado Por: Angel Rodrigo Naharro**", unsafe_allow_html=True)

if __name__ == '__main__':
    principal()

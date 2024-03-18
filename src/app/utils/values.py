import pandas as pd

def rev_dict(diccionario):
    return {valor: clave for clave, valor in diccionario.items()}

diccionario_columnas = {
    'FAVC': 'Consumo de Alimentos Altos en Calorías',
    'FCVC': 'Consumo de Verduras',
    'NCP': 'Número de Comidas Principales',
    'CAEC': 'Comer Entre Comidas',
    'CH2O': 'Consumo Diario de Agua',
    'CALC': 'Consumo de Alcohol',
    'SCC': 'Monitoreo de Calorías',
    'FAF': 'Frecuencia de Actividad Física',
    'TUE': 'Uso de Dispositivos Tecnológicos',
    'MTRANS': 'Preferencia de Transporte',
    'Age':'Edad',
    'Height': 'Altura',
    'Weight':'Peso',
    'SMOKE':'Fumador',
    'Gender':'Género',
    'IMC':'Indice de Masa Corporal',
    'HA': 'Hábitos alimienticios',
    'UT': 'Uso de Tecnología',
    'family_history_with_overweight':'Antecedentes Familiares de Sobrepeso',
    'NObeyesdad': "Nivel de Obesidad"   }


dict_gender={'Female': 'Femenino', 'Male': 'Masculino'}
dict_mtrans={'Automobile': 'Automóvil', 'Motorbike': 'Moto', 'Public_Transportation': 'Transporte Público', 'Bike': 'Bicicleta', 'Walking': 'Caminar'}
dict_yesno={'no': 'No', 'yes': 'Si'}


dict_target={'Insufficient_Weight': 0,
 'Normal_Weight': 1,
 'Overweight_Level_I': 2,
 'Overweight_Level_II': 3,
  'Obesity_Type_I': 4,
 'Obesity_Type_II': 5,
 'Obesity_Type_III': 6,}

dict_prediction={0: 'Peso insuficiente',
 1: 'Peso normal',
 2: 'Sobrepeso Nivel I',
 3: 'Sobrepeso Nivel II',
 4: 'Obesidad Tipo I',
 5: 'Obesidad Tipo II',
 6: 'Obesidad Tipo III'}

target = "NObeyesdad"

decod_target= "Nivel de obesidad"

dict_CAEC={
    'no': 0,
    'Sometimes': 1,
    'Frequently': 2,
    'Always': 3}

dict_CALC={
    'no': 0,
    'Sometimes': 1,
    'Frequently': 2,
    'Always': 3}



selvars=['Age',
 'CAEC',
 'CALC',
 'CH2O',
 'FAF',
 'FCVC',
 'Gender_Male',
 'HA',
 'Height',
 'IMC',
 'MTRANS_Public_Transportation',
 'NCP',
 'TUE',
 'UT',
 'Weight',
 'family_history_with_overweight_yes']

dict_var_modelo = {
    'Edad': 'Age',
    'Comer Entre Comidas': 'CAEC',
    'Consumo de Alcohol': 'CALC',
    'Consumo Diario de Agua': 'CH2O',
    'Frecuencia de Actividad Física': 'FAF',
    'Consumo de Verduras': 'FCVC',
    'Género': 'Gender_Male',
    'Hábitos alimienticios': 'HA',
    'Altura': 'Height',
    'Indice de Masa Corporal': 'IMC',
    'Preferencia de Transporte': 'MTRANS_Public_Transportation',
    'Número de Comidas Principales': 'NCP',
    'Uso de Dispositivos Tecnológicos': 'TUE',
    'Uso de Tecnología': 'UT',
    'Peso': 'Weight',
    'Antecedentes Familiares de Sobrepeso': 'family_history_with_overweight_yes'
}

variables = [
    'Gender',
    'Age',
    'Height',
    'Weight',
    'family_history_with_overweight',
    'FAVC',
    'FCVC',
    'NCP',
    'CAEC',
    'SMOKE',    
    'CH2O',
    'SCC',
    'FAF',
    'TUE',
    'CALC',
    'MTRANS' ]

preguntas = [
    "¿Cuál es tu género?",
    "Ingresa tu edad",
    "Ingresa tu altura (m)",
    "Ingresa tu peso (Kg)",
    "¿Tiene un familiar que ha sufrido o sufre de sobrepeso?",
    "¿Consumes alimentos altos en calorías con frecuencia?",
    "¿Sueles comer verduras en tus comidas?",
    "¿Cuántas comidas principales tienes al día?",
    "¿Comes algo entre comidas?",
    "¿Fumas?",
    "¿Cuánta agua bebes diariamente?",
    "¿Monitoreas las calorías que consumes diariamente?",
    "¿Con qué frecuencia haces actividad física?",
    "¿Cuánto tiempo usas dispositivos tecnológicos como móvil, videojuegos, televisión, ordenador y otros?",
    "¿Con qué frecuencia bebes alcohol?",
    "¿Qué transporte usas habitualmente?"
]

opciones_respuesta = [
    ["Femenino", "Masculino"],
    "-",
    "-",
    "-",
    ["No", "Sí"],
    ["No", "Sí"],
    ["Nunca", "A veces", "Siempre"],
    ["Una", "Dos", "Tres", "Más de tres"],
    ["No", "A veces", "Frecuentemente", "Siempre"],
    ["No", "Sí"],
    ["Menos de 1L", "Entre 1L y 2L", "Más de 2L"],
    ["No", "Sí"],
    ["No tengo", "1 o 2 días", "3 o 4 días", "4 o 5 días"],
    ["0-2 horas", "3-5 horas", "Más de 5 horas"],
    ["No bebo", "A veces", "Frecuentemente", "Siempre"],
    ["Automóvil", "Motocicleta", "Bicicleta", "Transporte Público", "Caminar"]
]
opciones_respuesta = [[option] if not isinstance(option, list) else option for option in opciones_respuesta]



df2= pd.DataFrame({
    'Columna': variables,
    'Pregunta': preguntas,
    'Opciones de Respuesta': opciones_respuesta
})


var_values={
'FCVC':[1,2,3],
'NCP':[1,2,3,4],
'CH2O':[1,2,3],
'FAF':[0,1,2,3],
'CAEC':[0,1,2,3],
'CALC':[0,1,2,3],
'TUE':[0,1,2],
}

dicts = {}
for a, b, c in zip(
    df2.loc[df2.Columna.isin(list(var_values.keys())), 'Columna'],
    df2.loc[df2.Columna.isin(list(var_values.keys())), 'Opciones de Respuesta'],
    df2.loc[df2.Columna.isin(list(var_values.keys())), 'Columna'].apply(lambda x: var_values[x])
):
    dicts[a] = dict(zip(b, c))
dicts['Gender']=rev_dict(dict_gender)
dicts['MTRANS']=rev_dict(dict_mtrans)
cols_y_n=['family_history_with_overweight','FAVC','SMOKE','SCC']
for col in cols_y_n:
    dicts[col]=rev_dict(dict_yesno)


revdicts = {}
for a, b, c in zip(
    df2.loc[df2.Columna.isin(list(var_values.keys())), 'Columna'],
    df2.loc[df2.Columna.isin(list(var_values.keys())), 'Opciones de Respuesta'],
    df2.loc[df2.Columna.isin(list(var_values.keys())), 'Columna'].apply(lambda x: var_values[x])
):
    revdicts[a] = dict(zip(c, b))
revdicts['Gender']=dict_gender
revdicts['MTRANS']=dict_mtrans
cols_y_n=['family_history_with_overweight','FAVC','SMOKE','SCC']
for col in cols_y_n:
    revdicts[col]=dict_yesno
revdicts['NObeyesdad']=dict_prediction


vars2round = [ 'Age',
                    'FCVC',
                    'NCP',
                    'CAEC',
                    'CH2O',
                    'FAF',
                    'TUE',
                    'CALC']


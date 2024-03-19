import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from values import *





def XEncoder(Pipeline,X):
    
    categorical_columns = X.columns[X.dtypes=="object"].tolist()
    numeric_columns_to_transform = X.columns[X.dtypes!="object"].tolist()    
    X_trans = Pipeline.transform(X)
    transformed_columns = []

    # Obtener columnas transformadas de OHE
    if categorical_columns:
        ohe_columns = Pipeline.named_steps['preprocessor'].named_transformers_['categorical'].named_steps['onehot'].get_feature_names_out()
        transformed_columns.extend(ohe_columns)

    # Obtener columnas transformadas de variables numéricas
    numeric_transformer = Pipeline.named_steps['preprocessor'].named_transformers_['numeric'].named_steps['transformation']
    numeric_columns_transformed = [col for name, transformer, col in numeric_transformer.transformers_ if transformer != 'drop']
    transformed_columns.extend(numeric_columns_transformed)

    # Obtener columnas restantes
    remaining_columns = list(X.columns.difference(categorical_columns + numeric_columns_to_transform))
    transformed_columns.extend(remaining_columns)
    transformed_columns=list(pd.DataFrame(transformed_columns)[0].explode().dropna())
        
    return pd.DataFrame(X_trans, columns=transformed_columns)

def traduce_columnas(df, diccionario):
    dff=df.copy(True)
    dff.columns = [diccionario.get(col, col) for col in dff.columns]
    return dff

def codifica_columnas(df, diccionario):
    dff=df.copy(True)
    for col in dff.columns:
        if col in diccionario.keys():
            dff[col] = dff[col].map(diccionario[col])
    Feature_Engineering(dff)
    return dff

def Feature_Engineering(df):
        try:
            df['IMC'] = df['Weight'] / (df['Height'] ** 2)
            df['HA'] = df['FCVC'] * df['NCP']
            df['UT'] = df['TUE'] / df['Age']
        except:
            pass
        return df

def redondea_vars(df):
    dff=df.copy(deep=True)
    for var in vars2round:
        dff[var] = dff[var].apply(lambda x: round(x, 0))
        dff[var] = dff[var].astype(int)

    return dff
import os as so
import sys
sys.path.append(utils_path)
import pandas as pd
import numpy as np
import pickle
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer, PowerTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import python_clustering
from utils import *
from pyod.models.ecod import ECOD
from sklearn.ensemble import  GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import pickle
from sklearn.ensemble import StackingClassifier
import pickle

if __name__ == '__main__':
    # %%
    """# Análisis del Riesgo de Obesidad y problemas CardioVasculares
    """
    
    #---
    
    # Carga funciones propias reutilizables  de librerías
    utils_path = so.path.join(so.getcwd(), '..', 'utils')
    class PATH():
        RAW = './data/raw//'
        PROCESSED= './data/processed//'
        MODELS = './models//'
        REPORTS = './reports//'
    
    #---
    
    # %%
    """## Carga datos
    """
    
    #---
    
    df_train = pd.read_csv(PATH.RAW + 'train.csv')
    df_test = pd.read_csv(PATH.RAW +'test.csv')
    
    #---
    
    # %%
    """## Diccionarios"""
    
    #---
    
    diccionario_columnas = {
        'FAVC': 'Consumo frecuente de alimentos altos en calorías',
        'FCVC': 'Frecuencia de consumo de vegetales',
        'NCP': 'Número de comidas principales',
        'CAEC': 'Consumo de alimentos entre comidas',
        'CH20': 'Consumo de agua diario',
        'CALC': 'Consumo de alcohol',
        'SCC': 'Monitoreo del consumo de calorías',
        'FAF': 'Frecuencia de actividad física',
        'TUE': 'Tiempo utilizando dispositivos tecnológicos',
        'MTRANS': 'Transporte utilizado',
        'Age':'Edad',
        'Height': 'Altura',
        'Weight':'Peso',
        'SMOKE':'Fumador',
        'Gender':'Género',
        'IMC':'Indice de Masa Corporal',
        'HA': 'Hábitos alimienticios',
        'UT': 'Uso de Tecnología',
        'family_history_with_overweight':'Historial familiar con sobrepeso'
    }
    
    #---
    
    # %%
    """## Target
        """
    
    #---
    
    target = "NObeyesdad"
    
    #---
    
    dict_target={'Insufficient_Weight': 0,
     'Normal_Weight': 1,
     'Overweight_Level_I': 2,
     'Overweight_Level_II': 3,
      'Obesity_Type_I': 4,
     'Obesity_Type_II': 5,
     'Obesity_Type_III': 6,}
    
    #---
    
    dictrev_target={v:k for k,v in  dict_target.items()}
    
    #---
    
    # %%
    """No hay infinitos"""
    
    #---
    
    categoricas = df.columns[df.dtypes=="object"].tolist()
    numericas = df.columns[df.dtypes!="object"].tolist()
    
    #---
    
    df[target].value_counts() / len(df[target])
    
    #---
    
    # %%
    """# Feature Engineering
    """
    
    #---
    
    # %%
    """Hay columnas que tienen orden y que deben convertirse a variables discretas. Las mapeo también."""
    
    #---
    
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
    
    #---
    
    def Feature_Engineering(df):
        df.set_index('id', inplace=True)
        df['IMC'] = df['Weight'] / (df['Height'] ** 2)
        df['HA'] = df['FCVC'] * df['NCP']
        df['UT'] = df['TUE'] / df['Age']
        df['CALC']=df['CALC'].map(dict_CALC)
        df['CAEC']=df['CAEC'].map(dict_CAEC)
        return df
    
    #---
    
    df_train = Feature_Engineering(df_train)
    df_test = Feature_Engineering(df_test)
    df = Feature_Engineering(df)
    numericas = df.columns[df.dtypes!="object"].tolist()
    
    #---
    
    diccionario_columnas = {
        'FAVC': 'Consumo frecuente de alimentos altos en calorías',
        'FCVC': 'Frecuencia de consumo de vegetales',
        'NCP': 'Número de comidas principales',
        'CAEC': 'Consumo de alimentos entre comidas',
        'CH20': 'Consumo de agua diario',
        'CALC': 'Consumo de alcohol',
        'SCC': 'Monitoreo del consumo de calorías',
        'FAF': 'Frecuencia de actividad física',
        'TUE': 'Tiempo utilizando dispositivos tecnológicos',
        'MTRANS': 'Transporte utilizado',
        'Age':'Edad',
        'Height': 'Altura',
        'Weight':'Peso',
        'SMOKE':'Fumador',
        'Gender':'Género',
        'IMC':'Indice de Masa Corporal',
        'HA': 'Hábitos alimienticios',
        'UT': 'Uso de Tecnología',
        'family_history_with_overweight':'Historial familiar con sobrepeso'
    }
    
    #---
    
    # %%
    """# Transformaciones (Encoding)
         """
    
    #---
    
    categoricas = df.columns[df.dtypes=="object"].tolist()
    numericas = df.columns[df.dtypes!="object"].tolist()
    
    #---
    
    df = drop_cols(df, max_cardi=20, max_miss=30)
    
    #---
    
    def sqrt_transform(X):
        return np.sqrt(X)
    
    def log_transform(X):
        return np.log1p(X)
    
    #---
    
    # target no necesita transformación
    y = df[target]
    
    #---
    
    # %%
    """Guardo df, df_train y df_test antes de transformarlas"""
    
    #---
    
    # Save df as pickle
    df.to_pickle(PATH.PROCESSED + 'df.pkl')
    df_train.to_pickle(PATH.PROCESSED + 'df_train.pkl')
    df_test.to_pickle(PATH.PROCESSED + 'df_test.pkl')
    
    #---
    
    # Load df as pickle
    df = pd.read_pickle(PATH.PROCESSED + 'df.pkl')
    df_train = pd.read_pickle(PATH.PROCESSED + 'df_train.pkl')
    df_test = pd.read_pickle(PATH.PROCESSED + 'df_test.pkl')
    
    #---
    
    # %%
    """Pipeline para transformar las columnas. Para ver la significancia, lo tengo que pasar a numéricas y ya veré si posteriormente elimino columnas antes de quitar outliners."""
    
    #---
    
    dist_class = classify_distributions(df, threshold=0.05)
    
    #---
    
    def Encoder(df, target_col, threshold=0.05):
        if target_col not in df.columns:
            X = df.copy()
        else:
            X = df.drop(columns=[target_col])
    
        dist_class = classify_distributions(X, threshold)
        
        categorical_columns = list(X.select_dtypes(include=['object']).columns)
    
        if categorical_columns:
            categorical_pipeline = Pipeline([
                ('onehot', OneHotEncoder(drop='first'))
            ])
        else:
            categorical_pipeline = None
        
        numeric_columns_to_transform = [col for col, (dist,model) in dist_class.items()]
        untouched_columns = list(set(X.columns) - set(categorical_columns) - set(numeric_columns_to_transform))
    
        numeric_transformers = [
            ('sqrt_transform', FunctionTransformer(sqrt_transform), [col for col, (dist, _) in dist_class.items() if dist == 'positive_increasing']),
            ('log_transform', FunctionTransformer(log_transform), [col for col, (dist, _) in dist_class.items() if dist == 'positive_decreasing']),
            ('yeojohnson_transform', PowerTransformer(method='yeo-johnson'), [col for col, (dist, _) in dist_class.items() if dist not in ['positive_increasing', 'positive_decreasing']]),
            ('untouched', 'passthrough', untouched_columns)
        ]
    
        numeric_pipeline = Pipeline([
            ('transformation', ColumnTransformer(
                transformers=numeric_transformers,
                remainder='passthrough')
            ),
            ('scaler', StandardScaler())
        ])
    
        if categorical_pipeline:
            preprocessor = ColumnTransformer([
                ('categorical', categorical_pipeline, categorical_columns),
                ('numeric', numeric_pipeline, numeric_columns_to_transform)
            ])
        else:
            preprocessor = numeric_pipeline
    
        final_pipeline = Pipeline([
            ('preprocessor', preprocessor)
        ])
    
        required_columns = set([col for col, _ in dist_class.items()])
        if not required_columns.issubset(X.columns):
            missing_columns = required_columns - set(X.columns)
            raise ValueError(f"Missing columns: {missing_columns}")
    
        X_trans = final_pipeline.fit_transform(X)
        transformed_columns = []
    
        # Obtener columnas transformadas de OHE
        if categorical_columns:
            ohe_columns = final_pipeline.named_steps['preprocessor'].named_transformers_['categorical'].named_steps['onehot'].get_feature_names_out()
            transformed_columns.extend(ohe_columns)
    
        # Obtener columnas transformadas de variables numéricas
        numeric_transformer = final_pipeline.named_steps['preprocessor'].named_transformers_['numeric'].named_steps['transformation']
        numeric_columns_transformed = [col for name, transformer, col in numeric_transformer.transformers_ if transformer != 'drop']
        transformed_columns.extend(numeric_columns_transformed)
    
        # Obtener columnas restantes
        remaining_columns = list(X.columns.difference(categorical_columns + numeric_columns_to_transform))
        transformed_columns.extend(remaining_columns)
        transformed_columns=list(pd.DataFrame(transformed_columns)[0].explode().dropna())
        
        return pd.DataFrame(X_trans, columns=transformed_columns)
    
    
        
    
    #---
    
    X_trans=Encoder(df, target, threshold=0.05)
    
    #---
    
    #load df_test as pickle
    df_test = pd.read_pickle(PATH.PROCESSED + 'df_test.pkl')
    
    #---
    
    X_test=Encoder(df_test, target, threshold=0.05)
    
    #---
    
    #save X_test as pickle
    X_test.to_pickle(PATH.PROCESSED + 'X_test.pkl')
    
    #---
    
    # %%
    """# División train y test
        """
    
    #---
    
    # %%
    """En la división aprovecho a quitar los outliiers para trabajos los modelos baselines"""
    
    #---
    
    def outlier_split(X_trans,y, test_size=0.2):
        Xout=X_trans.copy()
        outclf = ECOD(n_jobs=-1)
        outclf.fit(Xout)
        mask = outclf.labels_  # binary labels (0: inliers, 1: outliers)
        X_trans_out=X_trans[mask==0]
        X_trans_out=pd.DataFrame(data=X_trans_out, columns=X_trans.columns)
        y_out=[y for  i,y in enumerate(y) if mask[i]==0]
        X_train, X_val, y_train, y_val = train_test_split(X_trans_out, y_out, test_size=test_size, random_state=123)
        return X_train, X_val, y_train, y_val
    
    #---
    
    X_train, X_val, y_train, y_val=outlier_split(X_trans,y, test_size=0.2)
    
    #---
    
    trainval=['X_train', 'X_val','y_train', 'y_val']
    for data in trainval:
        if  isinstance(data,pd.DataFrame):
            globals()[data]=reduce_memory_usage(globals()[data])
        else:
            globals()[data]=reduce_memory_usage(pd.DataFrame((globals()[data])))
    
    #---
    
    # Guardar X_train, X_val, y_train, y_val
    X_train.to_pickle(PATH.PROCESSED + 'X_train.pkl')
    X_val.to_pickle(PATH.PROCESSED + 'X_val.pkl')
    y_train.to_pickle(PATH.PROCESSED + 'y_train.pkl')
    y_val.to_pickle(PATH.PROCESSED + 'y_val.pkl')
    
    #---
    
    # Leer X_train, X_val, y_train, y_val
    X_train = pd.read_pickle(PATH.PROCESSED + 'X_train.pkl')
    X_val = pd.read_pickle(PATH.PROCESSED + 'X_val.pkl')
    y_train = pd.read_pickle(PATH.PROCESSED + 'y_train.pkl')
    y_val = pd.read_pickle(PATH.PROCESSED + 'y_val.pkl')
    X_test = pd.read_pickle(PATH.PROCESSED + 'X_test.pkl')
    
    #---
    
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
    
    #---
    
    # %%
    """# Baselines
    """
    
    #---
    
    # Modelos de Clasificación
    
    #---
    
    # Reduzco el modelo a las variables seleccionadas
    X_train_fs=X_train[selvars]
    X_val_fs=X_val[selvars]
    
    #---
    
    # %%
    """### Finetuning por RandomSearchCV"""
    
    #---
    
    models4tune=modelos = {
        'XGBoost': XGBClassifier(),
        'Gradient Boosting': GradientBoostingClassifier(),
        'LightGBM':LGBMClassifier(),
        'CatBoost': CatBoostClassifier()
    }
    
    #---
    
    # %%
    """# Metamodelo"""
    
    #---
    
    # load tuned_models from pickle in PATH.MODELS:
    with open(PATH.MODELS+'tuned_models.pkl', 'rb') as f:
        tuned_models_pkl = pickle.load(f)
    
    #---
    
    base_models=[
        ('XGBoost',tuned_models_pkl['XGBoost']),
        ('Gradient Boosting',tuned_models_pkl['Gradient Boosting']),
        ('LightGBM',tuned_models_pkl['LightGBM']),
        ('CatBoost',tuned_models_pkl['CatBoost'])]
    
    #---
    
    X_val_fs=X_val[selvars]
    
    #---
    
    #Uno los dataframe X_train_fs y X_val_fs
    X_train_meta=pd.concat([X_train_fs,X_val_fs],axis=0)
    y_train_meta=pd.concat([y_train,y_val],axis=0)
    
    #---
    
    meta_model=tuned_models_pkl['Gradient Boosting']
    
    #---
    
    stacking_model = StackingClassifier(estimators=base_models, final_estimator=meta_model)
    stacking_model.fit(X_train_meta, y_train_meta)
    
    #---
    
    # Save stacking model to pickle in PATH.MODELS:
    with open(PATH.MODELS+'stacking_model.pkl', 'wb') as f:
        pickle.dump(stacking_model, f)




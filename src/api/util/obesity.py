# Librerías

import pandas as pd
import dill

from pydantic import BaseModel, Field, confloat, conint, constr


class PersonObesity(BaseModel):
    age: int = 18
    gender: str = Field(..., description="Género de la persona", pattern="^(female|male)$")
    weight: confloat(gt=0) 
    height: confloat(gt=0)
    waist_circum_preferred: confloat(gt=0) 
    hip_circum: confloat(gt=0) 



def make_obesity_predictions(person: PersonObesity):
    person = person.dict()
    df = pd.DataFrame([person])
    df_transformed = custom_calculus(df)
    with open('./model/encoder_pipeline.pkl','rb') as f:
        piplEncoder = dill.load(f)
    with open('./model/XGBClassifier.pkl', 'rb') as f:
        XGBoostClassifier = dill.load(f)
    
    X_respuestas = codifica_columnas(df_respuestas,dicts)
    lstscols=['FCVC','NCP','TUE']
    X_respuestas[lstscols] = X_respuestas[lstscols].astype('int64')
    
    Feature_Engineering(X_respuestas)

    #Cargar pkl del pipeline
    X=XEncoder(piplEncoder,X_respuestas)

    # Load XGBoostClassifier from pickle in PATH.MODELS:
    x=np.array(XEncoder(piplEncoder,X_respuestas)[selvars])
    y_pred = XGBoostClassifier.predict(x)[0]
    y_pred_proba = pd.DataFrame(100*XGBoostClassifier.predict_proba(x),columns=dict_prediction.values())
    
    return y_pred, y_pred_proba
 




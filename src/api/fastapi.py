from fastapi import FastAPI
import util.obesity as ob

app = FastAPI() 

@app.get("/")
def read_root():
    print('hola')
    return {'message': 'API Predicción Obesidad'}

# Return obesity prediction for a person
@app.post('/obesity_prediction')
async def obesity_prediction(person: ob.PersonObesity):
    result = ob.make_obesity_prediction(person)
    return result

import pickle
from model import predict_resistance
from typing import Dict
from pydantic import BaseModel
from fastapi import FastAPI
from fastapi.responses import JSONResponse
import uvicorn
from mangum import Mangum


class YachtModel(BaseModel):
    length_wl: float
    beam_wl: float
    draft: float
    displacement: float
    centre_of_buoyancy: float
    prismatic_coefficient: float
    velocity: float


app = FastAPI()

handler = Mangum(app)

@app.post('/predict')
def predict(yacht: YachtModel):

    with open('./model.bin', 'rb') as f_in:
        model = pickle.load(f_in)
        f_in.close()

    prediction = predict_resistance(yacht.dict(), model)

    result = {
        'resistance': prediction
    }
    return JSONResponse(result)

if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=9000)

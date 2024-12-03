from fastapi import FastAPI, UploadFile, HTTPException
from pydantic import BaseModel
from typing import List
import pandas as pd
import io
import pickle

app = FastAPI()


class Item(BaseModel):
    name: str
    year: int
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: float
    engine: int
    max_power: int
    torque: int
    seats: float


class Items(BaseModel):
    cars: List[Item]


def predict_price(item: Item) -> float:
    model = pickle.load(open("best_model.sav", "rb"))
    scaler = pickle.load(open("scaler.sav", "rb"))

    features = [
        item.year,
        item.km_driven,
        item.mileage,
        item.engine,
        item.max_power,
        item.seats,
    ]

    scaled_features = scaler.transform([features])
    prediction = model.predict(scaled_features)
    return prediction[0]


# 4. Эндпоинт для предсказания цены одного объекта
# @app.post("/predict_single/")
# async def predict_item(car: CarFeatures):

#     return


@app.post("/predict_item/")
async def predict_item(item: Item) -> dict:
    price = predict_price(item)
    return {"predicted_price": price}


# 5. Эндпоинт для обработки CSV файла
@app.post("/predict_items/")
async def predict_items(file: UploadFile):
    if not file.filename.endswith(".csv"):
        raise HTTPException(
            status_code=400, detail="Invalid file type. Please upload a CSV file."
        )
    try:
        content = await file.read()
        df = pd.read_csv(io.BytesIO(content))
    except Exception as e:
        raise HTTPException(status_code=400, detail="Failed to read CSV file.")

    # Предсказания для каждой строки
    predictions = []
    for _, row in df.iterrows():
        car = Item(
            name=row["name"],
            year=int(row["year"]),
            km_driven=int(row["km_driven"]),
            fuel=row["fuel"],
            seller_type=row["seller_type"],
            transmission=row["transmission"],
            owner=row["owner"],
            mileage=float(row["mileage"]),
            engine=int(row["engine"]),
            max_power=float(row["max_power"]),
            torque=row["torque"],
            seats=int(row["seats"]),
        )
        predictions.append(predict_price(car))

    df["predicted_price"] = predictions

    output = io.StringIO()
    df.to_csv(output, index=False)
    output.seek(0)

    return {"filename": "predictions.csv", "content": output.getvalue()}

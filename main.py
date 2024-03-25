#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
from fastapi import FastAPI, Depends
import uvicorn as uv

sys.path.insert(1, "./models/")
from model import EmailDetectionModel, PredictionOutput


__version__ = "0.1.0"


app = FastAPI(
    title="ML Endpoint 🤸🏻‍♀️",
    description="""<h2>ML Endpoint using FastAPI and ScikitLearn<h2>""",
    version=__version__,
)


model = EmailDetectionModel()


@app.get("/")
async def root() -> dict:
    return {"message": "Hello 👋"}


@app.post("/api/prediction")
async def prediction(
    output: PredictionOutput = Depends(model.predict),
) -> PredictionOutput:
    return output


@app.on_event("startup")
async def startup() -> None:
    model.load_model()


if __name__ == "__main__":
    uv.run("main:app", reload=True, port=8000)

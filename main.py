from fastapi import FastAPI, Request, File, UploadFile, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import prometheus_client as prom
from prometheus_fastapi_instrumentator import Instrumentator
import time
import os
import pandas as pd
from classification import *
import base64
from typing import Annotated

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
Instrumentator().instrument(app).expose(app)
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})

@app.post("/upload", response_class=HTMLResponse)
async def upload(request: Request, epochs: Annotated[int, Form()], file: UploadFile):
   if not file:
        return {"Error": "No upload file sent"}
   ext = file.filename.split('.')[1]
   if ext not in ["csv", "xlsx"]:
      return {"Error": "Has to be csv or xlsx"}
   df = pd.read_csv(file.file) if ext == 'csv' else pd.read_excel(file.file)
   if len(df.columns) != 3:
      return {"Error": "Dataset must have 3 columns (2 features, 1 target)"}
   results = classify(df, epochs)
   qnnAcc = float(results['qnnAcc'])
   img_buf = results['img_buf']
   nnAcc = float(results['nnAcc'])
   nnTime = results['nnTime']
   qnnTime = results['qnnTime']
   str_equivalent_image = base64.b64encode(img_buf.getvalue()).decode()
   img_tag = "<img src='data:image/png;base64," + str_equivalent_image + "'/>"
   return templates.TemplateResponse("results.html", {"request": request, "qnnAcc": qnnAcc, 'nnAcc': nnAcc, "qnnTime": qnnTime, "nnTime": nnTime, "img_tag": img_tag})
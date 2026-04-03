from fastapi import FastAPI
from app.models import StudentsInput
from app.services import generate_pairs

app = FastAPI(title="Kopeer API")

@app.get("/")
def root():
    return {"message" : "API Funcionando!!"}

@app.post("/generate_pairs")
def generate_pairs_endpoint(input_data: StudentsInput):
    result = generate_pairs(input_data.students, input_data.iterations)
    return result
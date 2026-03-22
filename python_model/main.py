from fastapi import FastAPI

app = FastAPI()

@app.post("/predict")
async def predict(data: dict):
    # 模拟模型推理
    return {"result": f"processed: {data.get('input')}"}

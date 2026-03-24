from fastapi import FastAPI
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

app = FastAPI()


@app.post("/predict")
async def predict(data: dict):
    # 模拟模型推理
    data = load_iris()
    clf = RandomForestClassifier()
    clf.fit(data.data, data.target)
    res = clf.predict([[5.1, 3.5, 1.4, 0.2]])  # 模拟一次预测
    return {"result": f"processed: {res}"}


def __main__():
    pass

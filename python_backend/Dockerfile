FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml .
RUN pip install uv && uv pip install --system fastapi uvicorn httpx

COPY main.py .

EXPOSE 8000
CMD ["python", "main.py"]
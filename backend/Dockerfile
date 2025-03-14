FROM python:3.9-slim-buster

WORKDIR /app

COPY requirements_frontend.txt .
RUN pip install --no-cache-dir -r requirements_frontend.txt

COPY frontend.py .

CMD ["python", "frontend.py"]

# Dockerfile for Backend
FROM python:3.9-slim-buster

WORKDIR /app

COPY requirements_backend.txt .
RUN pip install --no-cache-dir -r requirements_backend.txt

COPY backend.py .

CMD ["uvicorn", "backend:app", "--host", "0.0.0.0", "--port", "8000"]
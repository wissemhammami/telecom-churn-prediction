# Dockerfile — Telecom Churn Prediction
# Runs FastAPI on port 8000 and Streamlit on port 8501

FROM python:3.13-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000
EXPOSE 8501

CMD ["bash", "start.sh"]
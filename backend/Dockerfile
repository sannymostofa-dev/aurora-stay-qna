FROM python:3.13.15-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip &&     pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch &&     pip install --no-cache-dir -r requirements.txt

COPY . .

RUN python - <<'PY'
from sentence_transformers import SentenceTransformer
SentenceTransformer("HarishMaths/Hotel-Policy-Embedding")
PY

ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1

EXPOSE 7860

CMD ["gunicorn", "--bind", "0.0.0.0:7860", "--workers", "1", "--timeout", "180", "app:rag_api"]

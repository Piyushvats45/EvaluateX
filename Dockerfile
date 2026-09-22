# Zero-cost, portable deployment: builds a container running the FastAPI
# evaluation service on open-weight local models (no external API billed).
#
# Build:  docker build -t llm-eval-framework .
# Run:    docker run -p 8000:8000 llm-eval-framework
# Then:   curl http://localhost:8000/health
#
# Deploy for free on: Render (free web service tier), Fly.io (free
# allowance), or a Hugging Face Space with the "Docker" SDK.

FROM python:3.11-slim

WORKDIR /app

# System deps needed by torch/matplotlib at runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p outputs

EXPOSE 8000

ENV PYTHONUNBUFFERED=1

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]

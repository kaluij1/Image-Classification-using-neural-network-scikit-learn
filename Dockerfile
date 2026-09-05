# CPU image for the locked hold-for-review API.
# Does not train. Weights are mounted at runtime (gitignored).
# Dataset is not copied (CC BY-NC-SA 4.0).

FROM python:3.13-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir torch torchvision \
        --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY scripts/serve_api.py scripts/serve_api.py
COPY reports/baseline/metrics.json reports/baseline/metrics.json
COPY PROBLEM.md README.md ./

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"

CMD ["python", "scripts/serve_api.py", "--host", "0.0.0.0", "--port", "8000"]

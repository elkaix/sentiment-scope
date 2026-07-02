FROM python:3.13-slim

WORKDIR /app

# CPU-only torch wheel: ~10x smaller than the default CUDA build, and there
# is no GPU inside this container anyway. requirements-docker.txt excludes
# torch so we install it once from the CPU index, then the rest normally.
COPY requirements-docker.txt .
RUN pip install --no-cache-dir torch==2.12.1 --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements-docker.txt

COPY app ./app

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

# ── Etapa base ────────────────────────────────────────────────────────────────
# Python 3.11 slim: imagen oficial mínima, sin herramientas innecesarias.
FROM python:3.11-slim
 
# Evita que Python escriba archivos .pyc y que bufferee stdout/stderr.
# Sin PYTHONUNBUFFERED los logs no aparecen en tiempo real en Docker.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1
 
# ── Directorio de trabajo dentro del contenedor ───────────────────────────────
WORKDIR /app
 
# ── Dependencias ──────────────────────────────────────────────────────────────
# Se copia requirements.txt primero (antes que el código) para aprovechar
# la caché de capas de Docker: si el código cambia pero requirements.txt no,
# Docker no reinstala las librerías.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
 
# ── Código fuente ─────────────────────────────────────────────────────────────
COPY main.py .
COPY app/ ./app/
 
# ── Modelo entrenado ──────────────────────────────────────────────────────────
# El modelo se copia dentro de la imagen para que el contenedor sea autónomo.
# Si prefieres montar el modelo externamente (más flexible para actualizar
# versiones sin reconstruir la imagen), usa un volumen en docker run
# y omite esta línea.
COPY models/ ./models/
 
# ── Variables de entorno por defecto ─────────────────────────────────────────
# Pueden sobreescribirse al correr el contenedor con -e o --env-file.
ENV MODEL_DIR=./models \
    MODEL_VERSION=v1 \
    PORT=8000
 
# ── Puerto expuesto ───────────────────────────────────────────────────────────
# Documentativo — no publica el puerto solo, eso lo hace -p en docker run.
EXPOSE 8000
 
# ── Comando de arranque ───────────────────────────────────────────────────────
# Se usa la variable PORT para flexibilidad.
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT}"]
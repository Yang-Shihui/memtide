# ---- stage 1: build the React UI ----
FROM node:20-alpine AS ui
WORKDIR /build
COPY webui/package.json webui/package-lock.json ./
RUN npm ci
COPY webui/ .
RUN npm run build

# ---- stage 2: memtide runtime ----
FROM python:3.13-slim

WORKDIR /app

COPY pyproject.toml README.md ./
COPY memtide ./memtide
COPY scripts ./scripts
COPY --from=ui /build/dist ./memtide/static/

# psycopg is the PostgreSQL runtime driver; HTTP clients use Python stdlib
RUN pip install --no-cache-dir .

EXPOSE 8300

# deployment config comes from env (see docker-compose.yml / .env.example):
#   MEMTIDE_STORAGE, MEMTIDE_PG_DSN, MEMTIDE_VECTOR_BACKEND, MEMTIDE_QDRANT_URL,
#   LLM_BACKEND, LLM_BASE_URL, LLM_MODEL, LLM_API_KEY, DASHSCOPE_API_KEY
# The React management console is served at /console (override with
# MEMTIDE_STATIC_DIR); remove the dir to run API-only.
CMD ["python", "-m", "memtide", "serve", "--host", "0.0.0.0", "--port", "8300"]

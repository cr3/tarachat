FROM dustynv/pytorch:2.7-r36.4.0

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN apt-get update && apt-get install -y \
    python3-venv \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ENV UV_PYTHON_PREFERENCE=only-system

WORKDIR /app

# Venv inherits system-level torch (pre-installed with Jetson CUDA)
RUN uv venv --system-site-packages

COPY uv.lock pyproject.toml ./
RUN uv sync --frozen --no-install-project --no-install-package torch

COPY . ./
RUN uv sync --frozen --no-install-package torch

ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 80

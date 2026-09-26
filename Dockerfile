FROM python:3.12-slim-bookworm

# uv をインストール（公式イメージからバイナリをコピー）
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# MediaPipe / OpenCV の実行に必要なシステムライブラリ
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libegl1 \
        libgles2 \
    && rm -rf /var/lib/apt/lists/*

# 仮想環境はマウントされる /app の外に置く
ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    PATH="/opt/venv/bin:$PATH"

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --no-install-project

CMD ["bash"]

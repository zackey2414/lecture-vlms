# syntax=docker/dockerfile:1
# LLaVA-1.5 最小推論デモ。CLAUDE.md 準拠で、コンテナ内の uv が Python 依存を管理する。
# 実行には GPU が必要: `docker run --gpus all ...`（ホストに nvidia-container-toolkit 前提）。
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /app
ENV UV_LINK_MODE=copy \
    HF_HOME=/root/.cache/huggingface

# 依存だけ先に解決してレイヤをキャッシュ
COPY pyproject.toml ./
RUN uv sync --no-install-project

COPY demo.py ./

# 画像とHFキャッシュは実行時にマウントする（README 参照）。
# 既定の画像パスはコンテナ内に存在しないため、--image を明示すること。
ENTRYPOINT ["uv", "run", "--no-sync", "demo.py"]

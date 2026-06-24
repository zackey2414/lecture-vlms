# syntax=docker/dockerfile:1
# Qwen-VL-Chat 最小推論デモ（レガシー custom code）。コンテナ内 uv が Python 依存を管理する。
# 実行には GPU が必要: `docker run --gpus all ...`（ホストに nvidia-container-toolkit 前提）。
# 注意: Qwen-VL-Chat は transformers のバージョン整合がシビア。失敗時は README の対処を参照。
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /app
ENV UV_LINK_MODE=copy \
    HF_HOME=/root/.cache/huggingface

COPY pyproject.toml ./
RUN uv sync --no-install-project

COPY demo.py ./

# 画像とHFキャッシュは実行時にマウントする（README 参照）。--image を明示すること。
ENTRYPOINT ["uv", "run", "--no-sync", "demo.py"]

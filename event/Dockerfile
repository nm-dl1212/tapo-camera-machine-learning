FROM python:3.12-slim

RUN pip install uv
ENV UV_SYSTEM_PYTHON=true

WORKDIR /app

# 依存関係のインストール
RUN apt -y update \
    && apt -y install libopencv-dev

# 依存ファイルだけコピー
COPY pyproject.toml uv.lock ./

# 依存関係インストール（キャッシュ有効化のため）
RUN uv export --frozen --no-dev --format requirements-txt > requirements.txt \ 
    && uv pip install -r requirements.txt --no-cache-dir

# アプリ本体をコピー
COPY src/ ./src/

EXPOSE 8000

CMD ["uvicorn", "src.event_server:app", "--reload", "--host", "0.0.0.0"]

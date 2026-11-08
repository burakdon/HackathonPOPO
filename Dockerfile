# NeuroCue patient console: the FastAPI service in neurocue/demo/server.py plus its static page,
# listening on port 7860 (the Hugging Face Spaces convention). Everything the page computes comes
# from the same engine code the validation ran; the caches it needs are committed under data/cache.
#
#   docker build -t neurocue-demo .
#   docker run --rm -p 7860:7860 neurocue-demo      ->  http://localhost:7860
FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    PYTHONUNBUFFERED=1

# Spaces run the container as uid 1000; create that user and hand it the app directory.
RUN useradd -m -u 1000 user
WORKDIR /app

# Dependencies first, so edits to the code do not re-download torch. On Linux the lock file resolves
# torch from the CPU-only wheel index (see pyproject.toml), which keeps the image a few GB smaller.
COPY --chown=user:user pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

COPY --chown=user:user . .
USER user
ENV PATH="/app/.venv/bin:$PATH" \
    HOME=/home/user \
    NEUROCUE_HOST=0.0.0.0 \
    NEUROCUE_PORT=7860
EXPOSE 7860
CMD ["python", "-m", "neurocue.demo.server"]

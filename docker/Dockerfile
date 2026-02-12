# ---------------------------------------------------------------------------
# Stage 1: builder — install uv, sync workspace, build wheels
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS builder

LABEL stage="builder"

# Install uv (fast Python package manager)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /build

# Copy project manifests first for better layer caching
COPY pyproject.toml uv.lock ./
COPY packages/ packages/

# Copy remaining source
COPY . .

# Build all workspace packages into wheels
RUN uv build --all --out-dir /build/dist

# ---------------------------------------------------------------------------
# Stage 2: runtime — lean image with only the built wheels
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS runtime

LABEL maintainer="Memfun Contributors"
LABEL version="0.1.0"
LABEL description="Memfun — OSS autonomous coding agent: pluggable runtime + DSPy RLM + MCP + Agent Skills"

# Create non-root user
RUN groupadd --system memfun && \
    useradd --system --gid memfun --create-home memfun

# Install wheels built in the builder stage
COPY --from=builder /build/dist /tmp/dist
RUN pip install --no-cache-dir /tmp/dist/*.whl && \
    rm -rf /tmp/dist

# Switch to non-root user
USER memfun
WORKDIR /home/memfun

# Health check — verify the CLI is reachable
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD ["memfun", "version"]

ENTRYPOINT ["memfun"]

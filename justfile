ci:
    @echo "Running Ruff checks..."
    uv run ruff check

    @echo "Auto-formatting code with Ruff..."
    uv run ruff format

    @echo "Running tests..."
    uv run pytest

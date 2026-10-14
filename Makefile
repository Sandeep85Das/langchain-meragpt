.PHONY: test tests integration_test integration_tests lint format build

test tests:
	uv run --group test pytest --disable-socket --allow-unix-socket tests/unit_tests/

integration_test integration_tests:
	uv run --group test pytest tests/integration_tests/

lint:
	uv run --all-groups ruff check .
	uv run --all-groups ruff format . --diff
	uv run --all-groups mypy langchain_meragpt

format:
	uv run --all-groups ruff format .
	uv run --all-groups ruff check --fix .

build:
	uv build

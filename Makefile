.PHONY: test bench lint typecheck

test:
	pytest -q

bench:
	python -m evidenceorbit bench

lint:
	ruff check src tests

typecheck:
	mypy src

.PHONY: help install install-gpu test overfit train eval bench parity profile clean

help:
	@echo "Targets:"
	@echo "  install      - local (Mac/CPU/MPS) dev install"
	@echo "  install-gpu  - install on the rented CUDA box (adds ninja for kernel builds)"
	@echo "  test         - run unit tests (model shapes + overfit)"
	@echo "  overfit      - overfit the model on one batch (sanity check, CPU/MPS ok)"
	@echo "  train        - full training run (config-driven)"
	@echo "  eval         - validation L1 + sim rollout success rate"
	@echo "  parity       - custom CUDA kernel vs PyTorch reference (needs CUDA)"
	@echo "  bench        - latency before/after the kernel (needs CUDA)"
	@echo "  profile      - run NSight Systems on one forward+backward (needs CUDA)"

install:
	pip install -e ".[dev]"

install-gpu:
	pip install -e ".[data,dev]"
	pip install -r requirements-gpu.txt

test:
	pytest tests -q

overfit:
	python -m robopolicy.train --config configs/act_aloha.yaml --overfit-one-batch

train:
	python -m robopolicy.train --config configs/act_aloha.yaml

resume:
	python -m robopolicy.train --config configs/act_aloha.yaml --resume

eval:
	python -m robopolicy.eval --config configs/act_aloha.yaml --checkpoint outputs/last.pt

parity:
	pytest kernels/test_parity.py -q

bench:
	python kernels/bench.py

profile:
	bash profiling/run_nsight.sh

clean:
	rm -rf kernels/build outputs/*.tmp __pycache__ .pytest_cache

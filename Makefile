#!/bin/bash

run: prepare-environment download-inputs execute-pipeline

prepare-environment:
	python3 -m venv .venv
	source .venv/bin/activate
	pip3 install -r requirements.txt

download-inputs:
	python3 download_inputs.py

execute-pipeline:
	python3 pipeline.py

clean:
	rm -rf data/inputs data/outputs
	deactivate
	rm -rf .venv/

.PHONY: run prepare-environment download-inputs execute-pipeline clean

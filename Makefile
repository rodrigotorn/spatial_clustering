#!/bin/bash

run: prepare-environment download-inputs execute-pipeline

prepare-environment:
	pip3 install -r requirements.txt

download-inputs:
	python3 src/download_inputs.py

execute-pipeline:
	python3 pipeline.py

clean:
	rm -rf data/demographic_data data/geographic_data

.PHONY: run prepare-environment download-inputs execute-pipeline clean

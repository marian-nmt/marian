#!/bin/bash

dir_guard=$(mkdir -p $(@D))
second=$(word 2, $^)

.SECONDARY:

SRC=en
TRG=de

all: test


test: model
	python test.py

model:
	../scripts/download_models.py -w model -m $(SRC)-$(TRG)

.PHONY: test

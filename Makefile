SHELL=/bin/bash

prepare_data:
	python ./felix_work/create_4fold_dataset.py
	python ./yoyobar_work/run_prepare_data.py

reproduce_models:
	sh ./felix_work/reproduce.sh
	python ./yoyobar_work/run_train.py

run_inference:
	python ./inference.py


import os
from os.path import dirname, join, abspath

project_path = dirname(abspath(__file__))
models_path = join(project_path,"final_models")
dataset_path = join(project_path,"felix_work/")

# Paths for yoyobar
raw_data_dir = join(project_path,"yoyobar_work/clean_data/")
clean_data_dir = join(project_path,"yoyobar_work/clean_data/")
train_data_dir = join(project_path,"yoyobar_work/clean_data/train/")
checkpoint_dir = join(project_path,"yoyobar_work/checkpoints/")
final_model_dir = join(project_path,"yoyobar_work/final_model/")

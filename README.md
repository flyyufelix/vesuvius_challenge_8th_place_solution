# Kaggle Vesuvius Ink Detection Challenge
Code for 8th place solution in Kaggle Vesuvius Ink Detection Challenge.

Please refer to [this Kaggle forum post](https://www.kaggle.com/competitions/vesuvius-challenge-ink-detection/discussion/417383) for a description of the solution

## A Word on the Directory Structure
Since we only formed team in the last week of the competition, our work are mostly independently and are located in two separated folders respectively (i.e. `felix_work` and `yoyobar_work`). That said, we have made the pipeline for inference and models reproduction as simple as possible.

## Hardware Use

Felix:

CPU: Intel i7-13700KF
GPU: 3090 x 1
RAM: 64GB

Yoyobar:

CPU: 15 vCPU Intel(R) Xeon(R) Platinum 8375C CPU @ 2.90GHz
GPU: 4090 x 1
RAM: 80GB

## Environment
All requirements should be detailed in environment.yml. Using Anaconda is strongly recommended.

Use the command below to create a new conda environment named `vesuvius` where all the necessary packages are installed
```
conda env create -f environment.yml
conda activate vesuvius
```

## Prepare dataset
### Download competition dataset
As our work are most done separately, two independent data pipelines are required to run our code. Please follow the procedure below to place the competition data at the right place.

First download the competition dataset and place them inside `felix_work` folder:
```
$ cd felix_work
$ kaggle competitions download -c vesuvius-challenge-ink-detection
$ unzip vesuvius-challenge-ink-detection.zip
```
After unzipping, you will see two folders `train` and `test` that contains the train and test data for the competition are created.

Next, go to `yoyobar_work` folder and create symbolic links to point to the competition dataset we just downloaded
```
$ cd ../yoyobar_work
$ mkdir competition_data
$ cd competition_data
$ ln -s ../../felix_work/train train
$ ln -s ../../felix_work/test test
```
You will see two symbolic links `train` and `test` created inside the folder `yoyobar_work/competition_data` . We are now ready to run the data transformation pipeline.


### Data Transformation (e.g. create CV folds)
Navigate to the project root directory where the `Makefile` is located. Issue the following `make` command
```
$ make prepare_data
```

## Replicate Submission
This is to replicate our final submission notebook (public LB: 0.792859 / private LB: 0.655514).

We have created a kaggle dataset titled `vesuvius-8th-place-solution-models` with all the model checkpoints required to run the inference script.

### Download the kaggle dataset that contains all the model checkpoints
```
$ kaggle datasets download -d renman/vesuvius-8th-place-solution-models
```
Unzip `vesuvius-8th-place-solution-models` and you will see a `final_models` directory that contains all the models checkpoints

### Run inference code
```
$ make run_inference
```
The `submission.csv` file will be generated in the same directory

## Replicate Training
This is to run the training pipeline to regenerate all the models we have used for inference
```
$ make reproduce_models
```
Notice that all reproduced models are automatically being copied to the `final_models` directory to be consumed by the inference script. No manual model copy and pasting is required.

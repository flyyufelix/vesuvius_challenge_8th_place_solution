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
All requirements should be detailed in requirements.txt. Using Anaconda is strongly recommended.
```
conda create -n vesuvius python=3.8
conda activate vesuvius
pip install -r requirements.txt
```

## Prepare dataset
### Download competition dataset 
```
$ cd felix_work
$ kaggle competitions download -c vesuvius-challenge-ink-detection
$ unzip vesuvius-challenge-ink-detection.zip
```
After unzipping, you will see two folders `train` and `test` that contains the train and test data for the competition are created

### Data Transformation (e.g. create CV folds and resize images)
Navigate to the project root directory where the `Makefile` is located. Issue the following `make` command
```
$ make prepare_data
```

## Replicate Submission
This is to replicate our final submission notebook (public LB: 0.792859 / private LB: 0.55514).

We have created a kaggle dataset titled `vesuvius-8th-place-solution-models` with all the model checkpoints required to run the inference script.

### Download the kaggle dataset that contains all the model checkpoints
```
$ kaggle datasets download -d renman/vesuvius-8th-place-solution-models
```
You will see a `final_models` directory that contains all the models checkpoints

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

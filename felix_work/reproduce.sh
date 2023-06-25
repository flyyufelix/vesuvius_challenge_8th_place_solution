#!/bin/bash

# resnet34-224-models
python felix_work/train.py --exp_name=resnet34-224-models --backbone=resnet3d --model_depth=34 --nfolds=4 --validation_fold=1 --loss_fn=hybrid --input_size=224 --tile_size=224 --train_stride=112 --test_stride=112 --in_chans=18 --start_slice=22 --end_slice=40 --train_batch_size=8 --valid_batch_size=16 --epochs=48 
python felix_work/train.py --exp_name=resnet34-224-models --backbone=resnet3d --model_depth=34 --nfolds=4 --validation_fold=3 --loss_fn=hybrid --input_size=224 --tile_size=224 --train_stride=112 --test_stride=112 --in_chans=18 --start_slice=22 --end_slice=40 --train_batch_size=8 --valid_batch_size=16 --epochs=50 
python felix_work/train.py --exp_name=resnet34-224-models --backbone=resnet3d --model_depth=34 --nfolds=4 --validation_fold=4 --loss_fn=hybrid --input_size=224 --tile_size=224 --train_stride=112 --test_stride=112 --in_chans=18 --start_slice=22 --end_slice=40 --train_batch_size=8 --valid_batch_size=16 --epochs=17 
python felix_work/train.py --exp_name=resnet34-224-models --backbone=resnet3d --model_depth=34 --nfolds=4 --validation_fold=5 --loss_fn=hybrid --input_size=224 --tile_size=224 --train_stride=112 --test_stride=112 --in_chans=18 --start_slice=22 --end_slice=40 --train_batch_size=8 --valid_batch_size=16 --epochs=29 

# resnet34-192-models
python felix_work/train.py --exp_name=resnet34-192-models --backbone=resnet3d --model_depth=34 --nfolds=3 --validation_fold=1 --loss_fn=hybrid --input_size=192 --tile_size=192 --train_stride=96 --test_stride=96 --in_chans=18 --start_slice=22 --end_slice=40 --train_batch_size=8 --valid_batch_size=16 --epochs=30 
python felix_work/train.py --exp_name=resnet34-192-models --backbone=resnet3d --model_depth=34 --nfolds=3 --validation_fold=2 --loss_fn=hybrid --input_size=192 --tile_size=192 --train_stride=96 --test_stride=96 --in_chans=18 --start_slice=22 --end_slice=40 --train_batch_size=8 --valid_batch_size=16 --epochs=30 --select_best_model=True
python felix_work/train.py --exp_name=resnet34-192-models --backbone=resnet3d --model_depth=34 --nfolds=3 --validation_fold=3 --loss_fn=hybrid --input_size=192 --tile_size=192 --train_stride=96 --test_stride=96 --in_chans=18 --start_slice=22 --end_slice=40 --train_batch_size=8 --valid_batch_size=16 --epochs=28 

# mit-b3-attnPool-exp048-models
python felix_work/train.py --exp_name=mit-b3-attnPool-exp048-models --backbone=mit_b3 --nfolds=4 --validation_fold=1 --loss_fn=bce --input_size=224 --tile_size=224 --train_stride=112 --test_stride=112 --in_chans=11 --z_dims=3 --start_slice=25 --end_slice=36 --train_batch_size=16 --use_attn_pooling=True --valid_batch_size=16 --epochs=19
python felix_work/train.py --exp_name=mit-b3-attnPool-exp048-models --backbone=mit_b3 --nfolds=4 --validation_fold=3 --loss_fn=bce --input_size=224 --tile_size=224 --train_stride=112 --test_stride=112 --in_chans=11 --z_dims=3 --start_slice=25 --end_slice=36 --train_batch_size=16 --use_attn_pooling=True --valid_batch_size=16 --epochs=20
python felix_work/train.py --exp_name=mit-b3-attnPool-exp048-models --backbone=mit_b3 --nfolds=4 --validation_fold=4--loss_fn=bce --input_size=224 --tile_size=224 --train_stride=112 --test_stride=112 --in_chans=11 --z_dims=3 --start_slice=25 --end_slice=36 --train_batch_size=16 --use_attn_pooling=True --valid_batch_size=16 --epochs=20
python felix_work/train.py --exp_name=mit-b3-attnPool-exp048-models --backbone=mit_b3 --nfolds=4 --validation_fold=5--loss_fn=bce --input_size=224 --tile_size=224 --train_stride=112 --test_stride=112 --in_chans=11 --z_dims=3 --start_slice=25 --end_slice=36 --train_batch_size=16 --use_attn_pooling=True --valid_batch_size=16 --epochs=18

# mit-b3-models
python felix_work/train.py --exp_name=mit-b3-models --backbone=mit_b5 --nfolds=3 --validation_fold=1 --loss_fn=bce --input_size=224 --tile_size=224 --train_stride=112 --test_stride=112 --in_chans=6 --start_slice=29 --end_slice=35 --train_batch_size=16 --valid_batch_size=16 --epochs=18
python felix_work/train.py --exp_name=mit-b3-models --backbone=mit_b5 --nfolds=3 --validation_fold=2 --loss_fn=bce --input_size=224 --tile_size=224 --train_stride=112 --test_stride=112 --in_chans=6 --start_slice=29 --end_slice=35 --train_batch_size=16 --valid_batch_size=16 --epochs=20
python felix_work/train.py --exp_name=mit-b3-models --backbone=mit_b5 --nfolds=3 --validation_fold=3 --loss_fn=bce --input_size=224 --tile_size=224 --train_stride=112 --test_stride=112 --in_chans=6 --start_slice=29 --end_slice=35 --train_batch_size=16 --valid_batch_size=16 --epochs=20











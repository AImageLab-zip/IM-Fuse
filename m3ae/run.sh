CUDA_VISIBLE_DEVICES=1,2 python3 -u train_18.py --batch-size 2 --lr 0.0003 --model_type cnnnet --seed 500 --weight_kl 0.1 --feature_level 2 --epochs 300 --mdp 3 --val 20 --wd 0.0001 --deep_supervised True --patch_shape 128 --exp_name test1
 

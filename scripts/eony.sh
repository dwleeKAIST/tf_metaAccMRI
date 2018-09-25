export CUDA_VISIBLE_DEVICES=0
python3 g4_train.py --name 'DS4_meta_UnetwoBN_K10_simul' --model Unet_wo_BN  --gpu_ids 0 --k_shot 1 --lr 0.005 --lr_state 0.005 --k_shot 10 --k_shot_max 10 --nHidden 10 --DSrate 4 --nEpoch_state_update 1 --nEpoch_Wb_update 0 --ngf 64 --smallDB  


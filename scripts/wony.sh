export CUDA_VISIBLE_DEVICES=1
python3 g4_train.py --name 'DS4_meta_UnetwoBN_K1_Kitermax450_simul' --model Unet_wo_BN --gpu_ids 1 --k_shot 1 --lr 0.005 --k_shot_max 450 --nHidden 20 --DSrate 4 --nEpoch_state_update 1 --nEpoch_Wb_update 0 --ngf 64 --smallDB 
#python3 raki_train.py --name 'raki_DS4'  --model Gnet_ --dataset '7T' --batchSize 1 --lr 0.005 --gpu_ids 1 --DSrate 4 --ngf 1024 --w_decay 0.1  --debug_mode 

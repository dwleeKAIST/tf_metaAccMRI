export CUDA_VISIBLE_DEVICES=0
python3 g_train.py --name 'DS2_meta_K5_ngf512_WOpreproc'  --model Gnet_ --dataset '7T' --batchSize 1 --lr 0.005 --gpu_ids 0 --k_shot 5 --DSrate 2 --nEpoch_state_update 1 --nEpoch_Wb_update 1 --ngf 512  #--smallDB  #--debug_mode


#python3 g_train.py --name 'DS2_meta_K5iter_ngf512_kloss_retry'  --model Gnet_ --dataset '7T' --batchSize 1 --lr 0.005 --gpu_ids 0 --k_shot 5 --nHidden 10 --DSrate 2 --nEpoch_state_update 9 --nEpoch_Wb_update 1 --ngf 512  #--smallDB  #--debug_mode


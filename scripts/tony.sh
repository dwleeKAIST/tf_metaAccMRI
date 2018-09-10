export CUDA_VISIBLE_DEVICES=0
python3 g_train.py --name 'DS2_meta_K3'  --model Gnet_ --dataset '7T' --batchSize 1 --lr 0.005 --gpu_ids 0 --k_shot 1 --nHidden 10 --DSrate 2 --nEpoch_state_update 9 --nEpoch_Wb_update 1 --ngf 512  #--smallDB  #--debug_mode


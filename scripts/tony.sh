export CUDA_VISIBLE_DEVICES=0
python3 g_train.py --name 'DS2_meta_full_K50'  --model Gnet_ --dataset '7T' --batchSize 1 --lr 0.001 --gpu_ids 0 --k_shot 50 --nHidden 100 --DSrate 2 --nEpoch_state_update 3 --nEpoch_Wb_update 7 --ngf 32  #--smallDB  #--debug_mode


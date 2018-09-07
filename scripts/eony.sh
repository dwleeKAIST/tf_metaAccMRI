export CUDA_VISIBLE_DEVICES=0
python3 g_train.py --name 'metaGnet_debug' --debug_mode  --model Gnet_ --dataset '7T' --batchSize 1 --lr 0.01 --gpu_ids 0 --k_shot 10 --nHidden 10 --nEpoch_state_update 1

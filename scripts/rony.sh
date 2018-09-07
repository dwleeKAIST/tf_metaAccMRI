export CUDA_VISIBLE_DEVICES=0
python3 g_train.py --name 'DS2_metadebug_rony' --model Gnet_ --dataset '7T' --batchSize 1 --lr 0.01 --gpu_ids 1 --k_shot 10 --nHidden 100 --DSrate 2 --nEpoch_state_update 10 --ngf 32 --use_kloss --smallDB


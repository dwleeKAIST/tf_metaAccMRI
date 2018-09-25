export CUDA_VISIBLE_DEVICES=1
python3 g_train.py --name 'DS2_meta_K3iter_ngf512_full_iEpochPercent2' --model Gnet_ --dataset '7T' --batchSize 1 --lr 0.005 --gpu_ids 1 --k_shot 3 --nHidden 10 --DSrate 2 --nEpoch_state_update 9 --nEpoch_Wb_update 1 --ngf 512  #--clip 0.01


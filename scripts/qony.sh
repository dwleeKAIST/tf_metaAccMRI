export CUDA_VISIBLE_DEVICES=0
#python3 raki_train.py --name 'raki_DS4_zid16'  --model Gnet_DS4 --dataset '7T' --batchSize 1 --lr 0.005 --gpu_ids 1 --DSrate 4 --ngf 64 --nEpoch 250 --debug_mode --test_mode
python3 raki_train.py --name 'raki_7T_'  --dataset '7T' --batchSize 1 --lr 0.01 --gpu_ids 0 --DSrate 4 --ngf 64 --nEpoch 500 --dataroot './../../mrdata/T1w_pad_32ch_halfnY_std' --debug_mode #--test_mode
#python3 Graki_train.py --name 'Graki_DS4'  --model Gnet_DS4 --dataset '7T' --batchSize 1 --lr 0.005 --gpu_ids 1 --DSrate 4 --ngf 64 --nEpoch 20  --test_mode

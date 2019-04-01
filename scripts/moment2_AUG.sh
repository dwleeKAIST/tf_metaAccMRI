export CUDA_VISIBLE_DEVICES=1
python3 g4_train.py --name 'FS2_DS4_meta_GnetDS4_AUG_bu55' --model Gnet_DS4 --gpu_ids 1 --k_shot 20 --lr 0.0001 --lr_state 0.0001 --DSrate 4  --ngf 64 --nEpoch_state_update 1 --nEpoch_Wb_update 1 --dataroot './../../mrdata/T1w_pad_32ch_halfnY_std' --nEpoch 100 --Aug  --test_mode

#python3 g4_train.py --name 'FS2_DS4_meta_GnetDS4_ngf64_K10_Iinit0_full' --model Gnet_DS4 --gpu_ids 1 --k_shot 10 --lr 0.0001 --lr_state 0.0001 --k_shot 20 --DSrate 4 --nEpoch_state_update 1 --nEpoch_Wb_update 1 --ngf 64 --dataroot './../../mrdata/T1w_pad_32ch_halfnY_max' --nEpoch 50  --test_mode


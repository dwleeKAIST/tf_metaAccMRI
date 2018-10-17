export CUDA_VISIBLE_DEVICES=1
python3 g4_train.py --name 'FS2_DS4_GnetDS4_K30_ngf64_AUG2' --model Gnet_DS4 --gpu_ids 1 --k_shot 30 --lr 0.0001 --DSrate 4  --ngf 64 --dataroot './../../mrdata/T1w_pad_32ch_halfnY_std' --nEpoch 250 --Aug  #--test_mode

#python3 g4_train_stdCwise.py --name 'FS2_DS4_GnetDS4_K10_trainLossBP_stdCwise' --model Gnet_DS4 --gpu_ids 1 --k_shot 10 --lr 0.0001 --DSrate 4  --ngf 128 --dataroot './../../mrdata/T1w_pad_32ch_halfnY_stdCwise_' --nEpoch 250 --Aug  #--test_mode


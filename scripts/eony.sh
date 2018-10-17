export CUDA_VISIBLE_DEVICES=0
python3 g4_train.py --name 'FS2_DS4_GnetDS4_K35_ngf64_AUG2' --model Gnet_DS4  --gpu_ids 0 --lr 0.0001  --k_shot 35 --DSrate 4  --ngf 64 --dataroot './../../mrdata/T1w_pad_32ch_halfnY_std' --nEpoch 250  --Aug  # --test_mode 

#python3 g4_train_stdCwise.py --name 'FS2_DS4_GnetDS4_sel8_Ktest_ngf512_o5_Cwise' --model Gnet_DS4_8ch  --gpu_ids 0 --lr 0.00005  --k_shot 13 --DSrate 4  --ngf 512 --dataroot './../../mrdata/T1w_pad_32ch_halfnY_stdCwise_' --nEpoch 250  --Aug   #--test_mode 

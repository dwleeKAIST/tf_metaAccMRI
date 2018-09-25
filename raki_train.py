import os
import numpy as np
import tensorflow as tf
from util.util import tf_imgri2ssos, myNumExt, tf_kri2imgri, tf_pad0, DIM2CH2
import time
from ipdb import set_trace as st
from options.baseOptions import BaseOptions
from math import ceil
from model.myRNN import myDoubleLSTM as metaLearner
from model.RAKI import RAKI as myModel
# options to init
opt = BaseOptions().parse()
dtype= tf.float32
clip     = 0.00001
nB          = opt.batchSize


if opt.dataset=='7T':
    from data.DB7T import DB7T as myDB
else:
    st()

if opt.model == 'Gnet_':
    from model.learner import Gnet_ as Learner
else:
    st()
# init. DB first
DB_train = myDB(opt,'train')
DB_valid = myDB(opt,'valid')
opt      = DB_train.getInfo_RAKI(opt)
#
opt.nStep_train = ceil(len(DB_train)/opt.batchSize)
opt.nStep_valid = ceil(len(DB_valid)/opt.batchSize)
if opt.debug_mode:
    opt.nStep_train=1
    opt.nStep_valid=1
disp_step_train = ceil(opt.nStep_train/opt.disp_div_N)
disp_step_valid = ceil(opt.nStep_valid/opt.disp_div_N)

start_time = time.time()
myFS       = myModel(opt)

saver  = tf.train.Saver()
config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
#config.gpu_options.allow_growth=True
with tf.Session(config=config) as sess:
    latest_ckpt = tf.train.latest_checkpoint(opt.ckpt_dir)
    if latest_ckpt==None:
        print("Start! initially!")
        tf.global_variables_initializer().run()
        epoch_start=0
    else:
        print("Start from saved model -"+latest_ckpt)
        saver.restore(sess, latest_ckpt)
        epoch_start=myNumExt(latest_ckpt)+1
    summary_writer      = tf.summary.FileWriter(opt.log_dir, sess.graph)
    summary_writer_v    = tf.summary.FileWriter(opt.log_dir_v, sess.graph)   
    t_init = time.time()
    print("------- %d sec to initialize network-----" % (t_init-start_time) )
   
    # save the initailzed (Xavier) c2 to c2_init
    ## Fisrt, train the learner network
    for iEpoch in range(epoch_start, opt.nEpoch):
        if not opt.debug_mode:
            DB_train.shuffle()
        disp_cnt = 0
        sum_loss_train = 0.0
        t_i_1 = time.time()
        func_getBatch = DB_train.getBatch_G4_RAKI#_train
        out_argm = [myFS.loss_test, myFS.optimizer,  myFS.merged_all]
        out_arg  = [myFS.loss_test, myFS.optimizer ]
        for step in range(opt.nStep_train):
            _input_ACSk, _target_ACSk, _input_k, _target_k = func_getBatch(step*nB, (step+1)*nB)
            
            feed_dict={myFS.input_node_train: _input_ACSk, myFS.target_node_train: _target_ACSk, myFS.input_node_test: _input_k, myFS.target_node_test:_target_k,myFS.is_Training:True}
            if step%disp_step_train==0 or step==0:
                results = sess.run(out_argm,feed_dict=feed_dict)
                summary_writer.add_summary(results[-1], iEpoch*opt.disp_div_N+disp_cnt)
                disp_cnt+=1
            else:
                _,loss_test_train  = sess.run(out_arg,feed_dict=feed_dict)
            sum_loss_train += results[0]
        t_i_v = time.time()
        print('%d epoch -- loss : %.4f e-3, %d sec' %(iEpoch, sum_loss_train/opt.nStep_train*1000, t_i_v-t_i_1))
        disp_cnt = 0
        sum_loss_valid = 0.0

        for step in range(opt.nStep_valid):
            _input_ACSk, _target_ACSk, _input_k, _target_k = DB_valid.getBatch_G4_RAKI(step*opt.batchSize, (step+1)*opt.batchSize)
            feed_dict = {myFS.input_node_train: _input_ACSk, myFS.target_node_train: _target_ACSk, myFS.input_node_test: _input_k, myFS.target_node_test:_target_k,myFS.is_Training:False}
            if step%disp_step_valid==0 or step==0:
                loss_test_valid, merged = sess.run([myFS.loss_test, myFS.merged_all], feed_dict=feed_dict)
                summary_writer_v.add_summary(merged, iEpoch*opt.disp_div_N+disp_cnt)
                disp_cnt+=1
            else:
                loss_test_valid  = sess.run(myFS.loss_test, feed_dict=feed_dict)
            sum_loss_valid += loss_test_valid
        t_i = time.time()
        print('%d epoch -- loss : %.4f e-3, %d sec' %(iEpoch, sum_loss_valid/opt.nStep_valid*1000, t_i-t_i_v))
        if (iEpoch%50==0):
            path_saved = saver.save(sess, os.path.join(opt.ckpt_dir, "model.ckpt"), global_step=iEpoch)
    print(' Total time elpased : %d sec' %(t_init-time.time()))

   

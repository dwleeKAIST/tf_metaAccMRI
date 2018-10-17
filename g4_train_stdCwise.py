import os
import numpy as np
import tensorflow as tf
from util.util import tf_imgri2ssos, myNumExt, tf_kri2imgri, tf_pad0, DIM2CH2, wpng
import time
from ipdb import set_trace as st
from options.baseOptions import BaseOptions
from math import ceil
#from model.myRNN import myDoubleLSTM as metaLearner
from model.myRNN2 import myLSTM as metaLearner
from model.FewShot2stdCwise import FewShotG as myModel
#from model.FewShot import FewShotG as myModel
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
elif opt.model == 'Gnet2_':
    from model.learner import Gnet2_ as Learner
elif opt.model == 'Gnet_DS4_8ch':
    from model.learner import Gnet_DS4_8ch as Learner
elif opt.model == 'Gnet_DS4':
    from model.learner import Gnet_DS4 as Learner
elif opt.model == 'Gnet_DS4_wchange':
    from model.learner import Gnet_DS4 as Learner
elif opt.model == 'Gnetb_DS4':
    from model.learner import Gnetb_DS4 as Learner
elif opt.model == 'Gnet_DS6':
    from model.learner import Gnet_DS6 as Learner
elif opt.model == 'Unet_':
    from model.learner import Unet_ as Learner
elif opt.model == 'Unet_wo_BN':
    from model.learner import Unet_wo_BN as Learner
elif opt.model == 'tmp_net':
    from model.learner import tmp_net as Learner
else:
    st()
# init. DB first
DB_train = myDB(opt,'train')
DB_valid = myDB(opt,'valid')
DB_test  = myDB(opt,'test')
opt      = DB_train.getInfo(opt)
#
opt.nStep_train = ceil(len(DB_train)/opt.batchSize)
opt.nStep_valid = ceil(len(DB_valid)/opt.batchSize)
opt.nStep_test  = ceil(len(DB_test)/opt.batchSize)
if opt.debug_mode:
    opt.nStep_train=1
    opt.nStep_valid=1
disp_step_train = ceil(opt.nStep_train/opt.disp_div_N)
disp_step_valid = ceil(opt.nStep_valid/opt.disp_div_N)
if opt.DSrate==4:
    func_getBatch = DB_train.getBatch_G4_std_sel8
    func_getBatchv = DB_valid.getBatch_G4_std_sel8
    func_getBatcht = DB_test.getBatch_G4_std_sel8
elif opt.DSrate==6:
    func_getBatch = DB_train.getBatch_G6
    func_getBatchv = DB_valid.getBatch_G6
    func_getBatcht = DB_test.getBatch_G6
else:
    st()

## dummy data info
model_id  = 'DS'+str(opt.DSrate)+'nCh'+str(opt.ngf)
opt.d_spath1 = './model/dummy/'+opt.model+'_theta_shapes_'+model_id+'.npy'
opt.d_spath2 = './model/dummy/'+opt.model+'_ntheta_'+model_id+'.npy'
if os.path.isfile(opt.d_spath1):
    opt.dummy_theta_shapes=np.load(opt.d_spath1)
    opt.ntheta            =np.load(opt.d_spath2)
else:
    opt.dummy_theta_shapes=[]
    opt.ntheta            =[]
    st()
#init_ =  "./model/dummy/"+opt.model+model_id+ "initmodel.ckpt-0"
#
##
start_time = time.time()
myFS       = myModel(opt, metaLearner, Learner)

saver  = tf.train.Saver()
config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
#config.gpu_options.allow_growth=True
with tf.Session(config=config) as sess:
    latest_ckpt = tf.train.latest_checkpoint(opt.ckpt_dir)
    if latest_ckpt==None:
        print("Start! initially!")
        tf.global_variables_initializer().run()
        epoch_start=0
        #st()
        #if os.path.isfile(init_+".meta"):
        #    saver.restore(sess, init_)
    else:
        print("Start from saved model -"+latest_ckpt)
        saver.restore(sess, latest_ckpt)
        epoch_start=myNumExt(latest_ckpt)+1
    if not opt.test_mode:
        summary_writer      = tf.summary.FileWriter(opt.log_dir, sess.graph)
        summary_writer_v    = tf.summary.FileWriter(opt.log_dir_v, sess.graph)   
        t_init = time.time()
        print("------- %d sec to initialize network-----" % (t_init-start_time) )
        
        # save the initailzed (Xavier) c2 to c2_init
        myFS.save_state(sess)
        nE_update=(opt.nEpoch_state_update + opt.nEpoch_Wb_update)
        ## Fisrt, train the learner network
        for iEpoch in range(epoch_start, opt.nEpoch):
            if not opt.debug_mode:
                DB_train.shuffle()
            disp_cnt = 0
            sum_loss_train = 0.0
            t_i_1 = time.time()
            out_argm = [myFS.loss_test, myFS.optimizer_simul, myFS.save_states,  myFS.merged_all]
            out_arg  = [myFS.loss_test, myFS.optimizer_simul, myFS.save_states ]
       
            for step in range(opt.nStep_train):
                _input_ACSk, _target_ACSk, _input_k, _target_k, _scale_std = func_getBatch(step*nB, (step+1)*nB)
                
                myFS.restore_state(sess)
                feed_dict={myFS.input_node_train: _input_ACSk, myFS.target_node_train: _target_ACSk, myFS.input_node_test: _input_k, myFS.target_node_test:_target_k,myFS.is_Training:True, myFS.scale:_scale_std}
                if step%disp_step_train==0 or step==0:
                    results = sess.run(out_argm,feed_dict=feed_dict)
                    summary_writer.add_summary(results[-1], iEpoch*opt.disp_div_N+disp_cnt)
                    disp_cnt+=1
                else:
                    results  = sess.run(out_arg,feed_dict=feed_dict)
                myFS.save_state(sess) # save the c to c_init
                sum_loss_train += results[0]
            t_i_v = time.time()
            print('%d epoch -- loss : %.4f e-3, %d sec' %(iEpoch, sum_loss_train/opt.nStep_train*1000, t_i_v-t_i_1))

            ''' for Loop for validation goes here'''
            disp_cnt = 0
            sum_loss_valid = 0.0
            
            for step in range(opt.nStep_valid):
                myFS.restore_state(sess)
                _input_ACSk, _target_ACSk, _input_k, _target_k, _scale_std = func_getBatchv(step*opt.batchSize, (step+1)*opt.batchSize)
                feed_dict = {myFS.input_node_train: _input_ACSk, myFS.target_node_train: _target_ACSk, myFS.input_node_test: _input_k, myFS.target_node_test:_target_k,myFS.is_Training:False, myFS.scale:_scale_std}
                if step%disp_step_valid==0 or step==0:
                    loss_test_valid, merged = sess.run([myFS.loss_test, myFS.merged_all], feed_dict=feed_dict)
                    summary_writer_v.add_summary(merged, iEpoch*opt.disp_div_N+disp_cnt)
                    disp_cnt+=1
                else:
                    loss_test_valid  = sess.run(myFS.loss_test, feed_dict=feed_dict)
                sum_loss_valid += loss_test_valid
            t_i = time.time()
            print('%d epoch -- loss : %.4f e-3, %d sec' %(iEpoch, sum_loss_valid/opt.nStep_valid*1000, t_i-t_i_v))
            if (iEpoch%1==0):
                path_saved = saver.save(sess, os.path.join(opt.ckpt_dir, "model.ckpt"), global_step=iEpoch)
        print(' Total time elpased : %d sec' %(t_init-time.time()))

    '''test goes here'''
    if True:#else:
        out_arg  = [myFS.loss_test, myFS.target_test_ssos, myFS.net_out_test_ssos, myFS.net_out_ACSproj_test_ssos]
        sum_loss_test = 0.0
        t_i_t = time.time()
        for step in range(opt.nStep_test):
            myFS.restore_state(sess)
            _input_ACSk, _target_ACSk, _input_k, _target_k,_scale_std = func_getBatcht(step*opt.batchSize, (step+1)*opt.batchSize)
            feed_dict = {myFS.input_node_train: _input_ACSk, myFS.target_node_train: _target_ACSk, myFS.input_node_test: _input_k, myFS.target_node_test:_target_k,myFS.is_Training:False,myFS.scale:_scale_std}
            loss_test_test, tar_ssos, rec_ssos, rec_proj_ssos = sess.run(out_arg, feed_dict=feed_dict)
            sum_loss_test+= loss_test_test
            
            cmax = np.max(tar_ssos)
            pre_str = './result/'+opt.name+'/'+str(step+1)
            wpng(pre_str+'_tar.png', tar_ssos/cmax*255)
            wpng(pre_str+'_rec.png', rec_ssos/cmax*255)
            wpng(pre_str+'_recProj.png', rec_proj_ssos/cmax*255)
            wpng(pre_str+'_recProjEx20.png', np.abs((rec_proj_ssos-tar_ssos)/cmax*255*20))
            wpng(pre_str+'_recProjEx10.png', np.abs((rec_proj_ssos-tar_ssos)/cmax*255*10))

        t_i = time.time()
        print('test set : -- loss : %.4f e-3, %d sec' %(sum_loss_test/opt.nStep_test*1000, t_i-t_i_t))


import os
import numpy as np
from ipdb import set_trace as st
import tensorflow as tf
from util.util import tf_kri2imgri, DIM2CH2, tf_pad0, tf_imgri2ssos
import time

dtype = tf.float32

class FewShot:
    def __init__(self, opt, metaLearner, Learner):
        str_= "/device:GPU:"+str(opt.gpu_ids[0])
        print(str_)
        
        self.k_shot = opt.k_shot
        with tf.device(str_):
            ## parameters
            self.nCh_out = opt.nCh_out
 
            ##
            nEpochDecay = 20
            self.global_step = tf.Variable(0, dtype=dtype)
            self.lr_         = tf.train.exponential_decay(learning_rate=opt.lr, global_step=self.global_step, decay_steps=opt.nStep_train*nEpochDecay, decay_rate=0.99, staircase=True)
            tf.summary.scalar("train__monitor/learning rate ",  self.lr_)   
        
            ## def. of placeholder
            self.input_node_train  = tf.placeholder(dtype,[None, opt.nCh_in,  opt.nY, opt.nACS])
            self.target_node_train = tf.placeholder(dtype,[None, opt.nCh_out, opt.nY, opt.nACS])
            self.input_node_test   = tf.placeholder(dtype,[None, opt.nCh_in,  opt.nY, opt.nX])
            self.target_node_test  = tf.placeholder(dtype,[None, opt.nCh_out, opt.nY, opt.nX])
            self.is_Training       = tf.placeholder(tf.bool)
            
            self.target_imgri_train = tf_kri2imgri(self.target_node_train)
            self.target_imgri_test  = tf_kri2imgri(self.target_node_test)

            ## dymmy info extract
            if opt.dummy_theta_shapes==[]:
                st()
                _ = Learner(self.input_node_train, self.nCh_out, nCh=opt.ngf)
                dummy_theta = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='dummy')
                dummy_theta_shapes = [x.shape for x in dummy_theta]
                np.save(opt.d_spath1,dummy_theta_shapes)
                len_theta = []
                for aTheta in dummy_theta:
                    len_theta.append(int(tf.reshape(aTheta,[1,1,-1]).shape[2]))
                ntheta = np.sum(len_theta)
                np.save(opt.d_spath2,ntheta)
                print('please, restart the program')
                st()
                exit()
            else:
                self.dummy_theta_shapes = opt.dummy_theta_shapes
                self.ntheta             = opt.ntheta

            ## def meta-learner and init theta
            self.R = metaLearner(self.dummy_theta_shapes,Learner,  nHidden = opt.nHidden, ntheta=int(self.ntheta))
    
            with tf.variable_scope('state'):
                self.c1 = tf.Variable(tf.truncated_normal([self.ntheta,opt.nHidden],-0.1,0.1),name='c1')
                self.c2 = tf.Variable(tf.truncated_normal([self.ntheta,  1],-0.1,0.1),name='c2')
                self.f2 = tf.Variable(tf.ones([self.ntheta,1],name='f2'))
                self.i2 = tf.Variable(tf.zeros([self.ntheta,1],name='i2'))
                
            self.THETA_              = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.R.name)
            self.THETA_state         = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='state')
            #THETA                    = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
            self.c1_init = tf.Variable(tf.zeros([self.ntheta,opt.nHidden]),trainable=False)
            self.c2_init = tf.Variable(tf.zeros([self.ntheta,1]),trainable=False)
            self.f2_init = tf.Variable(tf.ones([self.ntheta,1]),trainable=False)
            self.i2_init = tf.Variable(tf.zeros([self.ntheta,1]),trainable=False)
            ##------------------------------------------------------------------------------
        
            restore_c1 = tf.assign(self.c1, self.c1_init)
            restore_c2 = tf.assign(self.c2, self.c2_init)
            restore_f2 = tf.assign(self.f2, self.f2_init)
            restore_i2 = tf.assign(self.i2, self.i2_init)
            self.restore_states = tf.group(restore_c1, restore_c2, restore_f2, restore_i2)
            
            save_c1 = tf.assign(self.c1_init,self.c1)
            save_c2 = tf.assign(self.c2_init,self.c2)
            #save_f2 = tf.assign(self.f2_init,self.f2)
            #save_i2 = tf.assign(self.i2_init,self.i2)
            self.save_states    = tf.group(save_c1, save_c2)#, save_f2, save_i2)
    
            self.c1, self.c2, self.f2, self.i2, self.loss_learner,self.grad_thetas = self.R.f(self.k_shot, self.input_node_train, self.target_node_train, self.c1, self.c2, self.f2, self.i2)
            
#            propagt_c1 = tf.assign(self.c1, self.c1_)
#            propagt_c2 = tf.assign(self.c2, self.c2_)
#            propagt_f2 = tf.assign(self.f2, self.f2_)
#            propagt_i2 = tf.assign(self.i2, self.i2_)      
#            self.propagt_state = tf.group(propagt_c1, propagt_c2, propagt_f2, propagt_i2)
#            
            ## Apply the last theta to test data
            self.net_out_train, _, self.loss_train =  self.R.learner_f(self.input_node_train,self.target_node_train,self.c2)
            self.cost_train     = tf.losses.mean_squared_error(labels=self.target_imgri_train,predictions=tf_kri2imgri(self.net_out_train))

            ## Apply the last theta to test data
            self.net_out_test, _, self.loss_test =  self.R.learner_f(self.input_node_test,self.target_node_test,self.c2)
            self.cost_test      = tf.losses.mean_squared_error(labels=self.target_imgri_test,predictions=tf_kri2imgri(self.net_out_test))
            
            self.optimizer_ = tf.train.AdamOptimizer(learning_rate = self.lr_)
            self.optimizer2_ = tf.train.AdamOptimizer(learning_rate = self.lr_*0.1)
            #optimizer_ = tf.train.RMSPropOptimizer(learning_rate = lr_) 
            #self.gvs        = self.optimizer_.compute_gradients(self.loss_test+self.loss_learner, var_list=self.THETA_)
            #self.gvs_state  = self.optimizer_.compute_gradients(self.loss_test+self.loss_learner, var_list=[self.THETA_state+self.THETA_])
            
            self.gvs        = self.optimizer2_.compute_gradients(self.loss_test, var_list=self.THETA_)
            self.gvs_state  = self.optimizer_.compute_gradients(self.loss_test, var_list=self.THETA_state)
            self.optimizer  = self.optimizer2_.apply_gradients(self.gvs,global_step=self.global_step)
            self.optimizer_state  = self.optimizer_.apply_gradients(self.gvs_state,global_step=self.global_step)
            #clipped_gvs= [(tf.clip_by_value(grad,-clip,clip),var) for grad, var in gvs]
            #optimizer  = optimizer_.apply_gradients(clipped_gvs)
            #    optimizer = tf.train.AdamOptimizer(learning_rate = lr_).minimize(loss_test,var_list=THETA_)
            #    optimizer_state = tf.train.AdamOptimizer(learning_rate=lr_).minimize(loss_test,var_list=[THETA_,THETA_state])
            ## monitor the of RNN
            tf.summary.scalar("LSTM1c/_init ",  tf.reduce_mean(tf.abs(self.c1_init)))
            tf.summary.scalar("LSTM2c/_init ",  tf.reduce_mean(tf.abs(self.c2_init)))
            tf.summary.scalar("LSTM2f/_init) ",  tf.reduce_mean(tf.abs(self.f2_init)))
            tf.summary.scalar("LSTM2i/_init) ",  tf.reduce_mean(tf.abs(self.i2_init)))    
            tf.summary.scalar("LSTM1c/_last) ",  tf.reduce_mean(tf.abs(self.c1)))
            tf.summary.scalar("LSTM2c/_last) ",  tf.reduce_mean(tf.abs(self.c2)))
            tf.summary.scalar("LSTM2f/_last) ",  tf.reduce_mean(tf.abs(self.f2)))
            tf.summary.scalar("LSTM2i/_last) ",  tf.reduce_mean(tf.abs(self.i2))) 
            tf.summary.scalar("LSTM1c/diff ",  tf.reduce_mean(tf.abs(self.c1-self.c1_init)))
            tf.summary.scalar("LSTM2c/diff ",  tf.reduce_mean(tf.abs(self.c2-self.c2_init)))
            tf.summary.scalar("LSTM2f/diff ",  tf.reduce_mean(tf.abs(self.f2-self.f2_init)))
            tf.summary.scalar("LSTM2i/diff ",  tf.reduce_mean(tf.abs(self.i2-self.i2_init)))
            #tf.summary.scalar("LSTM1/W1  ",   tf.reduce_mean(tf.abs(self.gvs[0][1])))
            #tf.summary.scalar("LSTM1/b1  ",   tf.reduce_mean(tf.abs(self.gvs[1][1])))
            #tf.summary.scalar("LSTM2/WF2  ",   tf.reduce_mean(tf.abs(self.gvs[2][1])))
            #tf.summary.scalar("LSTM2/bF2  ",   tf.reduce_mean(tf.abs(self.gvs[3][1])))
            tf.summary.scalar("LSTM1/Wg ",   tf.reduce_mean(tf.abs(self.gvs[0][0])))
            tf.summary.scalar("LSTM1/bg ",   tf.reduce_mean(tf.abs(self.gvs[1][0])))
            tf.summary.scalar("LSTM2/WFg ",  tf.reduce_mean(tf.abs(self.gvs[2][0])))
            tf.summary.scalar("LSTM2/bFg ",  tf.reduce_mean(tf.abs(self.gvs[3][0])))
            tf.summary.scalar("LSTM2/WIg ",  tf.reduce_mean(tf.abs(self.gvs[4][0])))
            tf.summary.scalar("LSTM2/bIg ",  tf.reduce_mean(tf.abs(self.gvs[5][0])))
            ##
             
            # display-tensorboard
            target_train_ssos   = tf_imgri2ssos(tf_kri2imgri(self.pad2ACS(self.target_node_train)))
            net_out_train_ssos  = tf_imgri2ssos(tf_kri2imgri(self.pad2ACS(self.net_out_train)))
            target_test_ssos    = tf_imgri2ssos(self.target_imgri_test)
            net_out_test_ssos   = tf_imgri2ssos(tf_kri2imgri(self.net_out_test)) 

            tf.summary.scalar("obj.func/train_last ",  self.cost_train)
            tf.summary.scalar("obj.func/test ", self.cost_test)
            scale_ = 255.0/tf.reduce_max(target_test_ssos)
            tf.summary.image("train_phase/target__ssos", tf.cast(target_train_ssos*scale_,dtype=tf.uint8))
            tf.summary.image("train_phase/netout__ssos", tf.cast(net_out_train_ssos*scale_,dtype=tf.uint8))
            tf.summary.image("train_phase/netout_errorx5", tf.cast(tf.abs(net_out_train_ssos-target_train_ssos)*5*scale_,dtype=tf.uint8))
            tf.summary.image("test_phase/target__ssos", tf.cast(target_test_ssos*scale_,dtype=tf.uint8))
            tf.summary.image("test_phase/netout__ssos", tf.cast(net_out_test_ssos*scale_,dtype=tf.uint8))
            tf.summary.image("test_phase/netout_errorx5", tf.cast(tf.abs(net_out_test_ssos-target_test_ssos)*5*scale_,dtype=tf.uint8))
            self.merged_all = tf.summary.merge_all() 

        
    def restore_state(self, sess):
        sess.run(self.restore_states)
        
    def save_state(self, sess):
        sess.run(self.save_states)
    def propagt_state(self, sess):
        sess.run(self.propagt_states)

    def pad2ACS(self,ACS_k):
#        myD2C_train     = DIM2CH2(shape4D=[1,opt.nCh,opt.nY,opt.nACS], inp_1st=True)
#            myD2C_test  =     DIM2CH2(shape4D=[1,opt.nCh,opt.nY,opt.nX], inp_1st=True)
        return tf_pad0(ACS_k, padX=128)

class FewShotG:
    def __init__(self, opt, metaLearner, Learner):
        str_= "/device:GPU:"+str(opt.gpu_ids[0])
        print(str_)
        
        self.k_shot = opt.k_shot
        with tf.device(str_):
            ## parameters
            self.nCh_out = opt.nCh_out
            self.myD2C_train = DIM2CH2(shape4D=[1,opt.nCh_in,opt.nY,opt.nACS], inp_1st=True)
            self.myD2C_test  = DIM2CH2(shape4D=[1,opt.nCh_in,opt.nY,opt.nX], inp_1st=True)
 
            ##
            nEpochDecay = 20
            self.global_step = tf.Variable(0, dtype=dtype)
            self.lr_         = tf.train.exponential_decay(learning_rate=opt.lr, global_step=self.global_step, decay_steps=opt.nStep_train*nEpochDecay, decay_rate=0.99, staircase=True)
            tf.summary.scalar("train__monitor/learning rate ",  self.lr_)   
             
            ## def. of placeholder
            self.input_node_train  = tf.placeholder(dtype,[None, opt.nCh_in,  opt.nY, int(opt.nACS/opt.DSrate)])
            self.target_node_train = tf.placeholder(dtype,[None, opt.nCh_out, opt.nY, int(opt.nACS/opt.DSrate)])
            self.input_node_test   = tf.placeholder(dtype,[None, opt.nCh_in,  opt.nY, int(opt.nX/opt.DSrate)])
            self.target_node_test  = tf.placeholder(dtype,[None, opt.nCh_out, opt.nY, int(opt.nX/opt.DSrate)])
            self.is_Training       = tf.placeholder(tf.bool)
            

            self.target_imgri_train = tf_kri2imgri(self.target_node_train)
            self.target_imgri_test  = tf_kri2imgri(self.target_node_test)

## dymmy info extract
            if opt.dummy_theta_shapes==[]:
                st()
                _ = Learner(self.input_node_train, self.nCh_out,nCh=opt.ngf)
                dummy_theta = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='dummy')
                dummy_theta_shapes = [x.shape for x in dummy_theta]
                np.save(opt.d_spath1,dummy_theta_shapes)
                len_theta = []
                for aTheta in dummy_theta:
                    len_theta.append(int(tf.reshape(aTheta,[1,1,-1]).shape[2]))
                ntheta = np.sum(len_theta)
                np.save(opt.d_spath2,ntheta)
                print('please, restart the program')
                st()
                exit()
            else:
                self.dummy_theta_shapes = opt.dummy_theta_shapes
                self.ntheta             = opt.ntheta

            ## def meta-learner and init theta
            self.R = metaLearner(self.dummy_theta_shapes,Learner,  nHidden = opt.nHidden, ntheta=int(self.ntheta),kloss=opt.use_kloss)
    
            with tf.variable_scope('state'):
                self.c1 = tf.Variable(tf.truncated_normal([self.ntheta,opt.nHidden],-0.1,0.1),name='c1')
                self.c2 = tf.Variable(tf.truncated_normal([self.ntheta,  1],-0.1,0.1),name='c2')
                self.f2 = tf.Variable(tf.ones([self.ntheta,1],name='f2'))
                self.i2 = tf.Variable(tf.zeros([self.ntheta,1],name='i2'))
                
            self.THETA_              = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.R.name)
            self.THETA_state         = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='state')
            #THETA                    = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
            self.c1_init = tf.Variable(tf.zeros([self.ntheta,opt.nHidden]),trainable=False)
            self.c2_init = tf.Variable(tf.zeros([self.ntheta,1]),trainable=False)
            self.f2_init = tf.Variable(tf.ones([self.ntheta,1]),trainable=False)
            self.i2_init = tf.Variable(tf.zeros([self.ntheta,1]),trainable=False)
            ##------------------------------------------------------------------------------
        
            restore_c1 = tf.assign(self.c1, self.c1_init)
            restore_c2 = tf.assign(self.c2, self.c2_init)
            restore_f2 = tf.assign(self.f2, self.f2_init)
            restore_i2 = tf.assign(self.i2, self.i2_init)
            self.restore_states = tf.group(restore_c1, restore_c2, restore_f2, restore_i2)
            
            save_c1 = tf.assign(self.c1_init,self.c1)
            save_c2 = tf.assign(self.c2_init,self.c2)
            #save_f2 = tf.assign(self.f2_init,self.f2)
            #save_i2 = tf.assign(self.i2_init,self.i2)
            self.save_states    = tf.group(save_c1, save_c2)#, save_f2, save_i2)
    
            self.c1, self.c2, self.f2, self.i2, self.loss_learner,self.grad_thetas = self.R.f(self.k_shot, self.input_node_train, self.target_node_train, self.c1, self.c2, self.f2, self.i2,self.is_Training)
            
#            propagt_c1 = tf.assign(self.c1, self.c1_)
#            propagt_c2 = tf.assign(self.c2, self.c2_)
#            propagt_f2 = tf.assign(self.f2, self.f2_)
#            propagt_i2 = tf.assign(self.i2, self.i2_)      
#            self.propagt_state = tf.group(propagt_c1, propagt_c2, propagt_f2, propagt_i2)
#            
            ## Apply the last theta to test data
            self.net_out_train, _, self.loss_train =  self.R.learner_f(self.input_node_train, self.target_node_train,self.c2)
            self.cost_train     = tf.losses.mean_squared_error(labels=self.target_imgri_train,predictions=tf_kri2imgri(self.net_out_train))

            ## Apply the last theta to test data
            self.net_out_test, _, self.loss_test =  self.R.learner_f(self.input_node_test,self.target_node_test,self.c2)
            self.cost_test      = tf.losses.mean_squared_error(labels=self.target_imgri_test,predictions=tf_kri2imgri(self.net_out_test))
            
            self.optimizer_ = tf.train.AdamOptimizer(learning_rate = self.lr_)
            self.optimizer2_ = tf.train.AdamOptimizer(learning_rate = self.lr_*0.1)
            #optimizer_ = tf.train.RMSPropOptimizer(learning_rate = lr_) 
            #self.gvs        = self.optimizer_.compute_gradients(self.loss_test+self.loss_learner, var_list=self.THETA_)
            #self.gvs_state  = self.optimizer_.compute_gradients(self.loss_test+self.loss_learner, var_list=[self.THETA_state+self.THETA_])
            #self.gvs_state  = self.optimizer_.compute_gradients(self.loss_test+self.loss_learner, var_list=[self.THETA_state])

            self.gvs_pre    = self.optimizer_.compute_gradients(self.loss_train, var_list=self.THETA_state)
            self.optimizer_pre = self.optimizer_.apply_gradients(self.gvs_pre,global_step=self.global_step)

            self.gvs        = self.optimizer_.compute_gradients(self.loss_test, var_list=self.THETA_)
            #self.gvs_state  = self.optimizer_.compute_gradients(self.loss_test+self.loss_learner, var_list=[self.THETA_state+self.THETA_])
            self.gvs_state  = self.optimizer2_.compute_gradients(self.loss_test, var_list=[self.THETA_state])

            self.optimizer  = self.optimizer_.apply_gradients(self.gvs,global_step=self.global_step)
            self.optimizer_state  = self.optimizer2_.apply_gradients(self.gvs_state,global_step=self.global_step)
            #clipped_gvs= [(tf.clip_by_value(grad,-clip,clip),var) for grad, var in gvs]
            #optimizer  = optimizer_.apply_gradients(clipped_gvs)
            #    optimizer = tf.train.AdamOptimizer(learning_rate = lr_).minimize(loss_test,var_list=THETA_)
            #    optimizer_state = tf.train.AdamOptimizer(learning_rate=lr_).minimize(loss_test,var_list=[THETA_,THETA_state])
            ## monitor the of RNN
            tf.summary.scalar("LSTM1c/_init ",  tf.reduce_mean(tf.abs(self.c1_init)))
            tf.summary.scalar("LSTM2c/_init ",  tf.reduce_mean(tf.abs(self.c2_init)))
            tf.summary.scalar("LSTM2f/_init) ",  tf.reduce_mean(tf.abs(self.f2_init)))
            tf.summary.scalar("LSTM2i/_init) ",  tf.reduce_mean(tf.abs(self.i2_init)))    
            tf.summary.scalar("LSTM1c/_last) ",  tf.reduce_mean(tf.abs(self.c1)))
            tf.summary.scalar("LSTM2c/_last) ",  tf.reduce_mean(tf.abs(self.c2)))
            tf.summary.scalar("LSTM2f/_last) ",  tf.reduce_mean(tf.abs(self.f2)))
            tf.summary.scalar("LSTM2i/_last) ",  tf.reduce_mean(tf.abs(self.i2))) 
            tf.summary.scalar("LSTM1c/diff ",  tf.reduce_mean(tf.abs(self.c1-self.c1_init)))
            tf.summary.scalar("LSTM2c/diff ",  tf.reduce_mean(tf.abs(self.c2-self.c2_init)))
            tf.summary.scalar("LSTM2f/diff ",  tf.reduce_mean(tf.abs(self.f2-self.f2_init)))
            tf.summary.scalar("LSTM2i/diff ",  tf.reduce_mean(tf.abs(self.i2-self.i2_init)))
            tf.summary.scalar("LSTM1/Wg ",   tf.reduce_mean(tf.abs(self.gvs[0][0])))
            tf.summary.scalar("LSTM1/bg ",   tf.reduce_mean(tf.abs(self.gvs[1][0])))
            tf.summary.scalar("LSTM2/WFg ",  tf.reduce_mean(tf.abs(self.gvs[2][0])))
            tf.summary.scalar("LSTM2/bFg ",  tf.reduce_mean(tf.abs(self.gvs[3][0])))
            tf.summary.scalar("LSTM2/WIg ",  tf.reduce_mean(tf.abs(self.gvs[4][0])))
            tf.summary.scalar("LSTM2/bIg ",  tf.reduce_mean(tf.abs(self.gvs[5][0])))
            ##
            target_train_k  = tf_pad0(self.myD2C_train.CH2D_(self.input_node_train,self.target_node_train),padX=opt.hPad)
            netoutL_train_k = tf_pad0(self.myD2C_train.CH2D_(self.input_node_train,self.net_out_train),padX=opt.hPad)
            target_test_k   = self.myD2C_test.CH2D_(self.input_node_test,self.target_node_test)
            netout_test_k   = self.myD2C_test.CH2D_(self.input_node_test,self.net_out_test)
             
            # display-tensorboard
            target_train_ssos   = tf_imgri2ssos(tf_kri2imgri(target_train_k))
            net_out_train_ssos  = tf_imgri2ssos(tf_kri2imgri(netoutL_train_k))
            target_test_ssos    = tf_imgri2ssos(tf_kri2imgri(target_test_k))
            net_out_test_ssos   = tf_imgri2ssos(tf_kri2imgri(netout_test_k)) 

            tf.summary.scalar("obj.func/train_last ",  self.cost_train)
            tf.summary.scalar("obj.func/test ", self.cost_test)
            scale_ = 255.0/tf.reduce_max(target_test_ssos)
            tf.summary.image("train_phase/target__ssos", tf.cast(target_train_ssos*scale_,dtype=tf.uint8))
            tf.summary.image("train_phase/netout__ssos", tf.cast(net_out_train_ssos*scale_,dtype=tf.uint8))
            tf.summary.image("train_phase/netout_errorx5", tf.cast(tf.abs(net_out_train_ssos-target_train_ssos)*5*scale_,dtype=tf.uint8))
            tf.summary.image("test_phase/target__ssos", tf.cast(target_test_ssos*scale_,dtype=tf.uint8))
            tf.summary.image("test_phase/netout__ssos", tf.cast(net_out_test_ssos*scale_,dtype=tf.uint8))
            tf.summary.image("test_phase/netout_errorx5", tf.cast(tf.abs(net_out_test_ssos-target_test_ssos)*5*scale_,dtype=tf.uint8))
            self.merged_all = tf.summary.merge_all() 

        
    def restore_state(self, sess):
        sess.run(self.restore_states)
        
    def save_state(self, sess):
        sess.run(self.save_states)
    def propagt_state(self, sess):
        sess.run(self.propagt_states)


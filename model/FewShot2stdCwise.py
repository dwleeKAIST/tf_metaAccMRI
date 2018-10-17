import os
import numpy as np
from ipdb import set_trace as st
import tensorflow as tf
from util.util import tf_kri2imgri, DIM2CH2, DIM2CH4,DIM2CH6, tf_pad0, tf_imgri2ssos
import time

dtype = tf.float32

class FewShotG:
    def __init__(self, opt, metaLearner, Learner):
        str_= "/device:GPU:"+str(opt.gpu_ids[0])
        print(str_)
        
        self.k_shot = opt.k_shot
        with tf.device(str_):
            ## parameters
            self.nCh_out = opt.nCh_out
            self.ACS_mask = opt.ACS_mask
            self.ACS_maskT= 1-self.ACS_mask

            if opt.DSrate ==2:
                self.myD2C_train = DIM2CH2(shape4D=[1,opt.nCh_in,opt.nY,opt.nACS])
                self.myD2C_test  = DIM2CH2(shape4D=[1,opt.nCh_in,opt.nY,opt.nX] )
            elif opt.DSrate==4:
                self.myD2C_train = DIM2CH4(shape4D=[1,opt.nCh_in,opt.nY,opt.nACS])
                self.myD2C_test  = DIM2CH4(shape4D=[1,opt.nCh_in,opt.nY,opt.nX])
            elif opt.DSrate==6:
                self.myD2C_train = DIM2CH6(shape4D=[1,opt.nCh_in,opt.nY,opt.nACS])
                self.myD2C_test  = DIM2CH6(shape4D=[1,opt.nCh_in,opt.nY,opt.nX])
            else:
                st()
 
            ##
            nEpochDecay = 20
            self.global_step = tf.Variable(0, dtype=dtype)
            self.lr_         = tf.train.exponential_decay(learning_rate=opt.lr, global_step=self.global_step, decay_steps=opt.nStep_train*nEpochDecay, decay_rate=0.99, staircase=True)
            self.lr_state_   = tf.train.exponential_decay(learning_rate=opt.lr_state, global_step=self.global_step, decay_steps=opt.nStep_train*nEpochDecay, decay_rate=0.99, staircase=True)
            tf.summary.scalar("train__monitor/learning rate ",  self.lr_)   
            tf.summary.scalar("train__monitor/learning rate for state ",  self.lr_state_)   
            
            ## def. of placeholder
            nB = 1
            self.input_node_train  = tf.placeholder(dtype,[nB, opt.nCh_in,  opt.nY, opt.dsnACS])
            self.target_node_train = tf.placeholder(dtype,[nB, opt.nCh_out, opt.nY, opt.dsnACS])
            self.input_node_test   = tf.placeholder(dtype,[nB, opt.nCh_in,  opt.nY, opt.dsnX])
            self.target_node_test  = tf.placeholder(dtype,[nB, opt.nCh_out, opt.nY, opt.dsnX])
            self.is_Training       = tf.placeholder(tf.bool)
            self.scale = tf.placeholder(dtype,[nB, opt.nCh_in])

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
            self.R = metaLearner(self.dummy_theta_shapes,Learner,  opt)
    
            with tf.variable_scope('state'):
                self.c = tf.get_variable('c',[self.ntheta,  1],initializer=tf.contrib.layers.xavier_initializer())
                self.h = tf.get_variable('h',[self.ntheta,  1],initializer=tf.contrib.layers.xavier_initializer())
                self.f = tf.Variable(tf.ones([self.ntheta,1],name='f'))
                self.i = tf.Variable(tf.zeros([self.ntheta,1],name='i'))
                #self.i = tf.Variable(tf.ones([self.ntheta,1],name='i'))
                #self.o = tf.Variable(tf.ones([self.ntheta,1],name='o'))
                self.o = tf.Variable(tf.zeros([self.ntheta,1],name='o'))               
            self.THETA_              = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.R.name)
            self.THETA_state         = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='state')
            
            self.c_init = tf.Variable(tf.zeros([self.ntheta,1]),trainable=False)
            self.h_init = tf.Variable(tf.zeros([self.ntheta,1]),trainable=False)
            self.f_init = tf.Variable(tf.ones([self.ntheta,1]),trainable=False)
            self.i_init = tf.Variable(tf.zeros([self.ntheta,1]),trainable=False)
            #self.i_init = tf.Variable(tf.ones([self.ntheta,1]),trainable=False)
            #self.o_init = tf.Variable(tf.ones([self.ntheta,1]),trainable=False)
            self.o_init = tf.Variable(tf.zeros([self.ntheta,1]),trainable=False)       
            restore_c = tf.assign(self.c, self.c_init)
            restore_h = tf.assign(self.h, self.h_init)
            restore_f = tf.assign(self.f, self.f_init)
            restore_i = tf.assign(self.i, self.i_init)
            restore_o = tf.assign(self.o, self.o_init)
            self.restore_states = tf.group(restore_c, restore_h, restore_f, restore_i, restore_o)
            
            save_c = tf.assign(self.c_init,self.c)
            save_h = tf.assign(self.h_init,self.h)
            save_f = tf.assign(self.f_init,self.f)
            save_i = tf.assign(self.i_init,self.i)
            save_o = tf.assign(self.o_init,self.o)
            self.save_states    = tf.group(save_c)#, save_h, save_f, save_i,save_o)
     
            self.c, self.h, self.f, self.i, self.o,  self.loss_learner, self.grad_thetas = self.R.f(self.k_shot, self.input_node_train, self.target_node_train, self.c, self.h, self.f, self.i, self.o,self.is_Training)
            
#            propagt_c1 = tf.assign(self.c1, self.c1_)
#            propagt_c2 = tf.assign(self.c2, self.c2_)
#            propagt_f2 = tf.assign(self.f2, self.f2_)
#            propagt_i2 = tf.assign(self.i2, self.i2_)      
#            self.propagt_state = tf.group(propagt_c1, propagt_c2, propagt_f2, propagt_i2)
#            
            ## Apply the last theta to test data
            self.net_out_train, _, self.loss_train =  self.R.learner_f(self.input_node_train, self.target_node_train,self.c)
            self.cost_train     = tf.losses.mean_squared_error(labels=self.target_imgri_train,predictions=tf_kri2imgri(self.net_out_train))

            ## Apply the last theta to test data
            self.net_out_test, _, self.loss_test =  self.R.learner_f(self.input_node_test,self.target_node_test,self.c)
            self.cost_test      = tf.losses.mean_squared_error(labels=self.target_imgri_test,predictions=tf_kri2imgri(self.net_out_test))
            
            self.optimizer_ = tf.train.AdamOptimizer(learning_rate = self.lr_)
            #self.gvs_simul  = self.optimizer_.compute_gradients(self.loss_test, var_list=[self.THETA_, self.THETA_state])
            self.gvs_simul  = self.optimizer_.compute_gradients(self.loss_test+self.loss_learner, var_list=[self.THETA_, self.THETA_state])
            self.optimizer_simul = self.optimizer_.apply_gradients(self.gvs_simul, global_step=self.global_step)
            
            ## monitor the of RNN
            tf.summary.scalar("LSTMc/_init ",  tf.reduce_mean(tf.abs(self.c_init)))
            tf.summary.scalar("LSTMf/_init) ",  tf.reduce_mean(tf.abs(self.f_init)))
            tf.summary.scalar("LSTMi/_init) ",  tf.reduce_mean(tf.abs(self.i_init)))    
            tf.summary.scalar("LSTMo/_init) ",  tf.reduce_mean(tf.abs(self.o_init)))
            tf.summary.scalar("LSTMh/_init) ",  tf.reduce_mean(tf.abs(self.h_init)))
            tf.summary.scalar("LSTMc/_last) ",  tf.reduce_mean(tf.abs(self.c)))
            tf.summary.scalar("LSTMf/_last) ",  tf.reduce_mean(tf.abs(self.f)))
            tf.summary.scalar("LSTMi/_last) ",  tf.reduce_mean(tf.abs(self.i)))
            tf.summary.scalar("LSTMo/_last) ",  tf.reduce_mean(tf.abs(self.o))) 
            tf.summary.scalar("LSTMh/_last) ",  tf.reduce_mean(tf.abs(self.h))) 
            ##
            tf.summary.histogram("gate_f", self.f)
            tf.summary.histogram("gate_i", self.i)
            tf.summary.histogram("gate_o", self.o)

            scale_tensor    = tf.tile( self.scale[:,:,tf.newaxis,tf.newaxis],[1,1,opt.nY, opt.nX])
            target_train_k  = tf_pad0(self.myD2C_train.CH2D_(self.input_node_train,self.target_node_train),padX=opt.hPad)*scale_tensor
            netoutL_train_k = tf_pad0(self.myD2C_train.CH2D_(self.input_node_train,self.net_out_train),padX=opt.hPad)*scale_tensor
            target_test_k   = self.myD2C_test.CH2D_(self.input_node_test,self.target_node_test)*scale_tensor
            netout_test_k   = self.myD2C_test.CH2D_(self.input_node_test,self.net_out_test)*scale_tensor
             
            # display-tensorboard
            target_train_ssos   = tf_imgri2ssos(tf_kri2imgri(target_train_k))
            net_out_train_ssos  = tf_imgri2ssos(tf_kri2imgri(netoutL_train_k))
            self.target_test_ssos    = tf_imgri2ssos(tf_kri2imgri(target_test_k))
            self.net_out_test_ssos   = tf_imgri2ssos(tf_kri2imgri(netout_test_k)) 
            self.net_out_ACSproj_test_ssos   = tf_imgri2ssos(tf_kri2imgri( self.ACS_mask*target_train_k + self.ACS_maskT*netout_test_k)  ) 
            tf.summary.scalar("obj.func/train_last ",  self.cost_train)
            tf.summary.scalar("obj.func/test ", self.cost_test)
            scale_ = 255.0/tf.reduce_max(self.target_test_ssos)
            tf.summary.image("train_phase/target__ssos", tf.cast(target_train_ssos*scale_,dtype=tf.uint8))
            tf.summary.image("train_phase/netout__ssos", tf.cast(net_out_train_ssos*scale_,dtype=tf.uint8))
            tf.summary.image("train_phase/netout_errorx20", tf.cast(tf.abs(net_out_train_ssos-target_train_ssos)*20*scale_,dtype=tf.uint8))
            tf.summary.image("test_phase/target__ssos", tf.cast(self.target_test_ssos*scale_,dtype=tf.uint8))
            tf.summary.image("test_phase/netout__ssos", tf.cast(self.net_out_test_ssos*scale_,dtype=tf.uint8))
            tf.summary.image("test_phase/netout_errorx20", tf.cast(tf.abs(self.net_out_test_ssos-self.target_test_ssos)*20*scale_,dtype=tf.uint8))
            tf.summary.image("test_phase/netoutACSproj__ssos", tf.cast(self.net_out_ACSproj_test_ssos*scale_,dtype=tf.uint8))
            tf.summary.image("test_phase/netoutACSproj_errorx20", tf.cast(tf.abs(self.net_out_ACSproj_test_ssos-self.target_test_ssos)*20*scale_,dtype=tf.uint8))
            self.merged_all = tf.summary.merge_all() 

        
    def restore_state(self, sess):
        sess.run(self.restore_states)
        
    def save_state(self, sess):
        sess.run(self.save_states)
    def propagt_state(self, sess):
        sess.run(self.propagt_states)


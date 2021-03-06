import os
import numpy as np
from ipdb import set_trace as st
import tensorflow as tf
from util.util import tf_kri2imgri, DIM2CH2, DIM2CH4, tf_pad0, tf_imgri2ssos
import time
from util.netUtil import gnet_DS4_32ch, gnet_DS4_4ch, gnet2_DS4_4ch,  gnet2_DS4

dtype = tf.float32
class RAKI:
    def __init__(self, opt):
        str_= "/device:GPU:"+str(opt.gpu_ids[0])
        print(str_)
        self.ACS_mask = opt.ACS_mask
        self.ACS_maskT=1-self.ACS_mask
        if opt.w_decay>0:
            self.reg = tf.contrib.layers.l2_regularizer(opt.w_decay)
        else:
            self.reg = None
        with tf.device(str_):
            ## parameters
            self.nCh_out = opt.nCh_out
            if opt.DSrate ==2:
                st()
                self.myD2C_train = DIM2CH2(shape4D=[1,opt.nCh_in,opt.nY,opt.nACS])
                self.myD2C_test  = DIM2CH2(shape4D=[1,opt.nCh_in,opt.nY,opt.nX] )
            elif opt.DSrate==4:
                self.myD2C_train = DIM2CH4(shape4D=[1,opt.nCh_in,opt.nY,opt.nACS])
                self.myD2C_test  = DIM2CH4(shape4D=[1,opt.nCh_in,opt.nY,opt.nX])
            else:
                st()
 
            ##
            nEpochDecay = 20
            self.global_step = tf.Variable(0, dtype=dtype)
            self.lr_         = tf.train.exponential_decay(learning_rate=opt.lr, global_step=self.global_step, decay_steps=opt.nStep_train*nEpochDecay, decay_rate=0.99, staircase=True)
            tf.summary.scalar("train__monitor/learning rate ",  self.lr_)   
            
            ## def. of placeholder
            nB = 1
            self.input_node_train  = tf.placeholder(dtype,[nB, opt.nCh_in,  opt.nY, opt.dsnACS])
            self.target_node_train = tf.placeholder(dtype,[nB, opt.nCh_out, opt.nY, opt.dsnACS])
            self.input_node_test   = tf.placeholder(dtype,[nB, opt.nCh_in,  opt.nY, opt.dsnX])
            self.target_node_test  = tf.placeholder(dtype,[nB, opt.nCh_out, opt.nY, opt.dsnX])
            self.is_Training       = tf.placeholder(tf.bool)
             
            self.net_out_train = gnet_DS4_32ch(self.input_node_train,opt.nCh_out, self.is_Training, nCh=opt.ngf, name_='raki',scope='Graki') 
            self.theta = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Graki')
            
            self.loss_train    = tf.losses.mean_squared_error(labels=self.target_node_train, predictions=self.net_out_train)

            self.net_out_test = gnet_DS4_32ch(self.input_node_test, opt.nCh_out, self.is_Training, nCh=opt.ngf, name_='raki', reuse=True, scope='Graki')
            self.loss_test = tf.losses.mean_squared_error(labels=self.target_node_test, predictions=self.net_out_test)

            self.optimizer_ = tf.train.AdamOptimizer(learning_rate = self.lr_)
            self.gvs        = self.optimizer_.compute_gradients(self.loss_train, var_list=self.theta)
            self.optimizer  = self.optimizer_.apply_gradients(self.gvs,global_step=self.global_step)
            
            ##
            target_train_k  = tf_pad0(self.myD2C_train.CH2D_(self.input_node_train,self.target_node_train),padX=opt.hPad)
            netoutL_train_k = tf_pad0(self.myD2C_train.CH2D_(self.input_node_train,self.net_out_train),padX=opt.hPad)

            target_test_k   = self.myD2C_test.CH2D_(self.input_node_test,self.target_node_test)
            netout_test_k   = self.myD2C_test.CH2D_(self.input_node_test,self.net_out_test)
            if opt.use_kproj:
                netout_test_k = netout_test_k*self.ACS_maskT + target_train_k*self.ACS_mask
            # display-tensorboard
            self.target_train_ssos   = tf_imgri2ssos(tf_kri2imgri(target_train_k))
            self.net_out_train_ssos  = tf_imgri2ssos(tf_kri2imgri(netoutL_train_k))
            self.target_test_ssos    = tf_imgri2ssos(tf_kri2imgri(target_test_k))
            self.net_out_test_ssos   = tf_imgri2ssos(tf_kri2imgri(netout_test_k)) 
            self.net_out_ACSproj_test_ssos = tf_imgri2ssos(tf_kri2imgri( self.ACS_mask*target_train_k + self.ACS_maskT*netout_test_k) )
            tf.summary.scalar("obj.func/train_last ",  self.loss_train)
            tf.summary.scalar("obj.func/test ", self.loss_test)
            scale_ = 255.0/tf.reduce_max(self.target_test_ssos)
            tf.summary.image("train_phase/target__ssos", tf.cast(self.target_train_ssos*scale_,dtype=tf.uint8))
            tf.summary.image("train_phase/netout__ssos", tf.cast(self.net_out_train_ssos*scale_,dtype=tf.uint8))
            tf.summary.image("train_phase/netout_errorx10", tf.cast(tf.abs(self.net_out_train_ssos-self.target_train_ssos)*10*scale_,dtype=tf.uint8))
            tf.summary.image("test_phase/target__ssos", tf.cast(self.target_test_ssos*scale_,dtype=tf.uint8))
            tf.summary.image("test_phase/netout__ssos", tf.cast(self.net_out_test_ssos*scale_,dtype=tf.uint8))
            tf.summary.image("test_phase/netout_errorx10", tf.cast(tf.abs(self.net_out_test_ssos-self.target_test_ssos)*10*scale_,dtype=tf.uint8))
            tf.summary.image("test_phase/netout_ACSproj__ssos", tf.cast(self.net_out_ACSproj_test_ssos*scale_,dtype=tf.uint8))
            tf.summary.image("test_phase/netout_ACSproj_errorx10", tf.cast(tf.abs(self.net_out_ACSproj_test_ssos-self.target_test_ssos)*10*scale_,dtype=tf.uint8))
            self.merged_all = tf.summary.merge_all() 

       

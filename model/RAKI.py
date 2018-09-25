import os
import numpy as np
from ipdb import set_trace as st
import tensorflow as tf
from util.util import tf_kri2imgri, DIM2CH2, DIM2CH4, tf_pad0, tf_imgri2ssos
import time
from util.netUtil import gnet

dtype = tf.float32
class RAKI:
    def __init__(self, opt):
        str_= "/device:GPU:"+str(opt.gpu_ids[0])
        print(str_)
        self.ACS_mask = opt.ACS_mask
        if opt.w_decay>0:
            self.reg = tf.contrib.layers.l2_regularizer(opt.w_decay)
        else:
            self.reg = None
        with tf.device(str_):
            ## parameters
            self.nCh_out = opt.nCh_out
            if opt.DSrate ==2:
                st()
                self.myD2C_train = DIM2CH2(shape4D=[1,opt.nCh_in*2,opt.nY,opt.nACS])
                self.myD2C_test  = DIM2CH2(shape4D=[1,opt.nCh_in*2,opt.nY,opt.nX] )
            elif opt.DSrate==4:
                self.myD2C_train = DIM2CH4(shape4D=[1,opt.nCh_in*2,opt.nY,opt.nACS])
                self.myD2C_test  = DIM2CH4(shape4D=[1,opt.nCh_in*2,opt.nY,opt.nX])
            else:
                st()
 
            ##
            nEpochDecay = 20
            self.global_step = tf.Variable(0, dtype=dtype)
            self.lr_         = tf.train.exponential_decay(learning_rate=opt.lr, global_step=self.global_step, decay_steps=opt.nStep_train*nEpochDecay, decay_rate=0.99, staircase=True)
            tf.summary.scalar("train__monitor/learning rate ",  self.lr_)   
            
            ## def. of placeholder
            self.input_node_train  = tf.placeholder(dtype,[None, opt.nCh_in*2,  opt.nY, opt.dsnACS])
            self.target_node_train = tf.placeholder(dtype,[None, opt.nCh_in,  opt.nCh_out, opt.nY, opt.dsnACS])
            self.input_node_test   = tf.placeholder(dtype,[None, opt.nCh_in*2,  opt.nY, opt.dsnX])
            self.target_node_test  = tf.placeholder(dtype,[None, opt.nCh_in,  opt.nCh_out, opt.nY, opt.dsnX])
            self.is_Training       = tf.placeholder(tf.bool)
            
            #self.target_imgri_train = tf_kri2imgri(self.target_node_train)
            #self.target_imgri_test  = tf_kri2imgri(self.target_node_test)
        
            ## Apply the last theta to test data
            net_out_train_ch0 = gnet(self.input_node_train, 2*(opt.DSrate-1), self.is_Training, nCh=opt.ngf, scope='gnet/0', reg_=self.reg)
            net_out_train_ch1 = gnet(self.input_node_train, 2*(opt.DSrate-1), self.is_Training, nCh=opt.ngf, scope='gnet/1', reg_=self.reg)
            net_out_train_ch2 = gnet(self.input_node_train, 2*(opt.DSrate-1), self.is_Training, nCh=opt.ngf, scope='gnet/2', reg_=self.reg)
            net_out_train_ch3 = gnet(self.input_node_train, 2*(opt.DSrate-1), self.is_Training, nCh=opt.ngf, scope='gnet/3', reg_=self.reg)
            net_out_train_ch4 = gnet(self.input_node_train, 2*(opt.DSrate-1), self.is_Training, nCh=opt.ngf, scope='gnet/4', reg_=self.reg)
            net_out_train_ch5 = gnet(self.input_node_train, 2*(opt.DSrate-1), self.is_Training, nCh=opt.ngf, scope='gnet/5', reg_=self.reg)
            net_out_train_ch6 = gnet(self.input_node_train, 2*(opt.DSrate-1), self.is_Training, nCh=opt.ngf, scope='gnet/6', reg_=self.reg)
            net_out_train_ch7 = gnet(self.input_node_train, 2*(opt.DSrate-1), self.is_Training, nCh=opt.ngf, scope='gnet/7', reg_=self.reg)           
            self.theta = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='gnet')
            
            loss_train_ch0 = tf.losses.mean_squared_error(labels=self.target_node_train[:,0,:,:,:], predictions=net_out_train_ch0)
            loss_train_ch1 = tf.losses.mean_squared_error(labels=self.target_node_train[:,1,:,:,:], predictions=net_out_train_ch1)
            loss_train_ch2 = tf.losses.mean_squared_error(labels=self.target_node_train[:,2,:,:,:], predictions=net_out_train_ch2)
            loss_train_ch3 = tf.losses.mean_squared_error(labels=self.target_node_train[:,3,:,:,:], predictions=net_out_train_ch3)
            loss_train_ch4 = tf.losses.mean_squared_error(labels=self.target_node_train[:,4,:,:,:], predictions=net_out_train_ch4)
            loss_train_ch5 = tf.losses.mean_squared_error(labels=self.target_node_train[:,5,:,:,:], predictions=net_out_train_ch5)
            loss_train_ch6 = tf.losses.mean_squared_error(labels=self.target_node_train[:,6,:,:,:], predictions=net_out_train_ch6)
            loss_train_ch7 = tf.losses.mean_squared_error(labels=self.target_node_train[:,7,:,:,:], predictions=net_out_train_ch7)
            
#            self.net_out_train = tf.concat([net_out_train_ch0[:,tf.newaxis,:3,:,:],net_out_train_ch1[:,tf.newaxis,:3,:,:],
#                net_out_train_ch2[:,tf.newaxis,:3,:,:],net_out_train_ch3[:,tf.newaxis,:3,:,:],
#                net_out_train_ch4[:,tf.newaxis,:3,:,:],net_out_train_ch5[:,tf.newaxis,:3,:,:],
#                net_out_train_ch6[:,tf.newaxis,:3,:,:],net_out_train_ch7[:,tf.newaxis,:3,:,:],
#                net_out_train_ch0[:,tf.newaxis,3:,:,:],net_out_train_ch1[:,tf.newaxis,3:,:,:],
#                net_out_train_ch2[:,tf.newaxis,3:,:,:],net_out_train_ch3[:,tf.newaxis,3:,:,:],
#                net_out_train_ch4[:,tf.newaxis,3:,:,:],net_out_train_ch5[:,tf.newaxis,3:,:,:],
#                net_out_train_ch6[:,tf.newaxis,3:,:,:],net_out_train_ch7[:,tf.newaxis,3:,:,:]], axis=1)
            self.net_out_train = tf.concat([net_out_train_ch0[:,tf.newaxis,:,:,:],net_out_train_ch1[:,tf.newaxis,:,:,:],
                net_out_train_ch2[:,tf.newaxis,:,:,:],net_out_train_ch3[:,tf.newaxis,:,:,:],
                net_out_train_ch4[:,tf.newaxis,:,:,:],net_out_train_ch5[:,tf.newaxis,:,:,:],
                net_out_train_ch6[:,tf.newaxis,:,:,:],net_out_train_ch7[:,tf.newaxis,:,:,:]], axis=1)
            self.loss_train    = loss_train_ch0 + loss_train_ch1 + loss_train_ch2 + loss_train_ch3 + loss_train_ch4 + loss_train_ch5 + loss_train_ch6 + loss_train_ch7

            net_out_test_ch0 = gnet(self.input_node_test, 2*(opt.DSrate-1), self.is_Training, nCh=opt.ngf, scope='gnet/0', reg_=self.reg, reuse=True)
            net_out_test_ch1 = gnet(self.input_node_test, 2*(opt.DSrate-1), self.is_Training, nCh=opt.ngf, scope='gnet/1', reg_=self.reg, reuse=True)
            net_out_test_ch2 = gnet(self.input_node_test, 2*(opt.DSrate-1), self.is_Training, nCh=opt.ngf, scope='gnet/2', reg_=self.reg, reuse=True)
            net_out_test_ch3 = gnet(self.input_node_test, 2*(opt.DSrate-1), self.is_Training, nCh=opt.ngf, scope='gnet/3', reg_=self.reg, reuse=True)
            net_out_test_ch4 = gnet(self.input_node_test, 2*(opt.DSrate-1), self.is_Training, nCh=opt.ngf, scope='gnet/4', reg_=self.reg, reuse=True)
            net_out_test_ch5 = gnet(self.input_node_test, 2*(opt.DSrate-1), self.is_Training, nCh=opt.ngf, scope='gnet/5', reg_=self.reg, reuse=True)
            net_out_test_ch6 = gnet(self.input_node_test, 2*(opt.DSrate-1), self.is_Training, nCh=opt.ngf, scope='gnet/6', reg_=self.reg, reuse=True)
            net_out_test_ch7 = gnet(self.input_node_test, 2*(opt.DSrate-1), self.is_Training, nCh=opt.ngf, scope='gnet/7', reg_=self.reg, reuse=True)
            
            loss_test_ch0 = tf.losses.mean_squared_error(labels=self.target_node_test[:,0,:,:,:], predictions=net_out_test_ch0)
            loss_test_ch1 = tf.losses.mean_squared_error(labels=self.target_node_test[:,1,:,:,:], predictions=net_out_test_ch1)
            loss_test_ch2 = tf.losses.mean_squared_error(labels=self.target_node_test[:,2,:,:,:], predictions=net_out_test_ch2)
            loss_test_ch3 = tf.losses.mean_squared_error(labels=self.target_node_test[:,3,:,:,:], predictions=net_out_test_ch3)
            loss_test_ch4 = tf.losses.mean_squared_error(labels=self.target_node_test[:,4,:,:,:], predictions=net_out_test_ch4)
            loss_test_ch5 = tf.losses.mean_squared_error(labels=self.target_node_test[:,5,:,:,:], predictions=net_out_test_ch5)
            loss_test_ch6 = tf.losses.mean_squared_error(labels=self.target_node_test[:,6,:,:,:], predictions=net_out_test_ch6)
            loss_test_ch7 = tf.losses.mean_squared_error(labels=self.target_node_test[:,7,:,:,:], predictions=net_out_test_ch7)
#            self.net_out_test = tf.concat([net_out_test_ch0[:,tf.newaxis,:3,:,:],net_out_test_ch1[:,tf.newaxis,:3,:,:],
#                net_out_test_ch2[:,tf.newaxis,:3,:,:],net_out_test_ch3[:,tf.newaxis,:3,:,:],
#                net_out_test_ch4[:,tf.newaxis,:3,:,:],net_out_test_ch5[:,tf.newaxis,:3,:,:],
#                net_out_test_ch6[:,tf.newaxis,:3,:,:],net_out_test_ch7[:,tf.newaxis,:3,:,:],
#                net_out_test_ch0[:,tf.newaxis,3:,:,:],net_out_test_ch1[:,tf.newaxis,3:,:,:],
#                net_out_test_ch2[:,tf.newaxis,3:,:,:],net_out_test_ch3[:,tf.newaxis,3:,:,:],
#                net_out_test_ch4[:,tf.newaxis,3:,:,:],net_out_test_ch5[:,tf.newaxis,3:,:,:],
#                net_out_test_ch6[:,tf.newaxis,3:,:,:],net_out_test_ch7[:,tf.newaxis,3:,:,:]], axis=1)
#            
            self.net_out_test = tf.concat([net_out_test_ch0[:,tf.newaxis,:,:,:],net_out_test_ch1[:,tf.newaxis,:,:,:],
                net_out_test_ch2[:,tf.newaxis,:,:,:],net_out_test_ch3[:,tf.newaxis,:,:,:],
                net_out_test_ch4[:,tf.newaxis,:,:,:],net_out_test_ch5[:,tf.newaxis,:,:,:],
                net_out_test_ch6[:,tf.newaxis,:,:,:],net_out_test_ch7[:,tf.newaxis,:,:,:]], axis=1)
            
            self.loss_test    = loss_test_ch0 + loss_test_ch1 + loss_test_ch2 + loss_test_ch3 + loss_test_ch4 + loss_test_ch5 + loss_test_ch6 + loss_test_ch7 

            self.optimizer_ = tf.train.AdamOptimizer(learning_rate = self.lr_)
            self.gvs        = self.optimizer_.compute_gradients(self.loss_train, var_list=self.theta)
            self.optimizer  = self.optimizer_.apply_gradients(self.gvs,global_step=self.global_step)
            
            ##
            target_train_k  = tf_pad0(self.myD2C_train.CH2D_RAKI(self.input_node_train,self.target_node_train),padX=opt.hPad)
            netoutL_train_k = tf_pad0(self.myD2C_train.CH2D_RAKI(self.input_node_train,self.net_out_train),padX=opt.hPad)
            target_test_k   = self.myD2C_test.CH2D_RAKI(self.input_node_test,self.target_node_test)
            netout_test_k   = self.myD2C_test.CH2D_RAKI(self.input_node_test,self.net_out_test)
            if opt.use_kproj:
                netout_test_k = netout_test_k*(1-self.ACS_mask) + target_train_k*self.ACS_mask
            # display-tensorboard
            target_train_ssos   = tf_imgri2ssos(tf_kri2imgri(target_train_k))
            net_out_train_ssos  = tf_imgri2ssos(tf_kri2imgri(netoutL_train_k))
            target_test_ssos    = tf_imgri2ssos(tf_kri2imgri(target_test_k))
            net_out_test_ssos   = tf_imgri2ssos(tf_kri2imgri(netout_test_k)) 

            tf.summary.scalar("obj.func/train_last ",  self.loss_train)
            tf.summary.scalar("obj.func/test ", self.loss_test)
            scale_ = 255.0/tf.reduce_max(target_test_ssos)
            tf.summary.image("train_phase/target__ssos", tf.cast(target_train_ssos*scale_,dtype=tf.uint8))
            tf.summary.image("train_phase/netout__ssos", tf.cast(net_out_train_ssos*scale_,dtype=tf.uint8))
            tf.summary.image("train_phase/netout_errorx10", tf.cast(tf.abs(net_out_train_ssos-target_train_ssos)*10*scale_,dtype=tf.uint8))
            tf.summary.image("test_phase/target__ssos", tf.cast(target_test_ssos*scale_,dtype=tf.uint8))
            tf.summary.image("test_phase/netout__ssos", tf.cast(net_out_test_ssos*scale_,dtype=tf.uint8))
            tf.summary.image("test_phase/netout_errorx10", tf.cast(tf.abs(net_out_test_ssos-target_test_ssos)*10*scale_,dtype=tf.uint8))
            self.merged_all = tf.summary.merge_all() 

       

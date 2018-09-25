import tensorflow as tf
#import tensorlayer as tl
#from tensorlayer.layers import InputLayer, Conv2d, MaxPool2d, DeConv2d, ConcatLayer
import tensorflow.contrib.layers as li
from ipdb import set_trace as st
from util.netUtil import unet, gnet,gnet2, Conv2dw, Pool2d, Conv2dT, Conv2dTw, CR, Conv1x1
from util.util import myTFfftshift2, tf_imgri2kri, tf_kri2imgri, slice2ker__, slice2ker__b
dtype = tf.float32
d_form  = 'channels_first'
d_form_ = 'NCHW'
ch_dim  = 1
 
def Unet_(inputs, n_out, weights=[], info_theta=[],nCh=32):
    nB=1
    shape_lv0 = [nB,nCh, int(inputs.shape[2]),int(inputs.shape[3])]
    shape_lv1 = [nB,nCh*2, int(int(inputs.shape[2])/2),int(int(inputs.shape[3])/2)]
    ch_convt21= [2,2,nCh*2,nCh*4]
    ch_convt10= [2,2,  nCh,nCh*2]
    if weights==[]:
        init = li.xavier_initializer()
        with tf.variable_scope('dummy'):
            c11 =  tf.layers.conv2d(inputs,filters=nCh, kernel_size=(3,3),strides=(1,1), padding="SAME", use_bias=True, data_format=d_form, kernel_initializer=init,name='c11')
            r11 =  tf.nn.relu(c11,name='r11')
            c12 =  tf.layers.conv2d(r11,filters=nCh, kernel_size=(3,3),strides=(1,1), padding="SAME", use_bias=True, data_format=d_form, kernel_initializer=init,name='c12')
            r12 =  tf.nn.relu(c12,name='r12')
            
            if True:
                p1 =  tf.layers.average_pooling2d(r12,pool_size=2,strides=2,padding='valid',data_format=d_form,name='p1')
                c21 =  tf.layers.conv2d(p1,filters=nCh*2, kernel_size=(3,3),strides=(1,1), padding="SAME", use_bias=True, data_format=d_form, kernel_initializer=init,name='c21')
                r21 =  tf.nn.relu(c21,name='r21')
                c22 =  tf.layers.conv2d(r21,filters=nCh*2, kernel_size=(3,3),strides=(1,1), padding="SAME", use_bias=True, data_format=d_form, kernel_initializer=init,name='c22')
                r22 =  tf.nn.relu(c22,name='r22')
                
                if True:
                    p2 = tf.layers.average_pooling2d(r22,pool_size=2,strides=2,padding='valid',data_format=d_form,name='p2')
                    c31 =  tf.layers.conv2d(p2,filters=nCh*4, kernel_size=(3,3),strides=(1,1), padding="SAME", use_bias=True, data_format=d_form, kernel_initializer=init,name='c31')
                    r31 =  tf.nn.relu(c31,name='r31')
                    c32 =  tf.layers.conv2d(r31,filters=nCh*4, kernel_size=(3,3),strides=(1,1), padding="SAME", use_bias=True, data_format=d_form, kernel_initializer=init,name='c32')
                    r32 =  tf.nn.relu(c32,name='r32')
                    t2 =  tf.nn.conv2d_transpose(r32, tf.ones(ch_convt21), shape_lv1, strides=[1,1,2,2],padding='VALID',data_format=d_form_,name='t2')
                
                c_21 =  tf.layers.conv2d(r22+t2,filters=nCh*2, kernel_size=(3,3),strides=(1,1), padding="SAME", use_bias=True, data_format=d_form, kernel_initializer=init,name='c_21')
                r_21 =  tf.nn.relu(c_21,name='r_21')
                c_22 =  tf.layers.conv2d(r_21,filters=nCh*2, kernel_size=(3,3),strides=(1,1), padding="SAME", use_bias=True, data_format=d_form, kernel_initializer=init,name='c_22')
                r_22 =  tf.nn.relu(c_22,name='r_22')
                t1 =  tf.nn.conv2d_transpose(r_22, tf.ones(ch_convt10), shape_lv0,strides=[1,1,2,2],padding='VALID',data_format=d_form_,name='t1')
            
            c_2 =  tf.layers.conv2d(r12+t1,filters=nCh, kernel_size=(3,3),strides=(1,1), padding="SAME", use_bias=True, data_format=d_form, kernel_initializer=init)
            r_2 = tf.nn.relu(c_2,name='r_2')
            c_1 =  tf.layers.conv2d(r_2, filters=nCh, kernel_size=(3,3),strides=(1,1), padding="SAME", use_bias=True, data_format=d_form, kernel_initializer=init)
            r_1 = tf.nn.relu(c_1,name='r_1')

            out_k_ri =  tf.layers.conv2d(r_1+r11,filters=n_out, kernel_size=(1,1),strides=(1,1), padding="SAME", use_bias=False, data_format=d_form, kernel_initializer=init, name='cout')

        return out_k_ri 
    else:
        idw=0; idx=0
        weights_  = []
        idx,w11   = slice2ker__(weights,   0, info_theta[idw]); idw+=1; weights_.append(w11)
        idx,b11   = slice2ker__b(weights,idx, info_theta[idw]); idw+=1; weights_.append(b11)
        idx,w12   = slice2ker__(weights, idx, info_theta[idw]); idw+=1; weights_.append(w12)
        idx,b12   = slice2ker__b(weights,idx, info_theta[idw]); idw+=1; weights_.append(b12)
        idx,w21   = slice2ker__(weights, idx, info_theta[idw]); idw+=1; weights_.append(w21)
        idx,b21   = slice2ker__b(weights,idx, info_theta[idw]); idw+=1; weights_.append(b21)
        idx,w22   = slice2ker__(weights, idx, info_theta[idw]); idw+=1; weights_.append(w22)
        idx,b22   = slice2ker__b(weights,idx, info_theta[idw]); idw+=1; weights_.append(b22)

        idx,w31   = slice2ker__(weights, idx, info_theta[idw]); idw+=1; weights_.append(w31)
        idx,b31   = slice2ker__b(weights,idx, info_theta[idw]); idw+=1; weights_.append(b31)
        idx,w32   = slice2ker__(weights, idx, info_theta[idw]); idw+=1; weights_.append(w32)
        idx,b32   = slice2ker__b(weights,idx, info_theta[idw]); idw+=1; weights_.append(b32)
        
        idx,w_21  = slice2ker__(weights, idx, info_theta[idw]); idw+=1; weights_.append(w_21)
        idx,b_21  = slice2ker__b(weights,idx, info_theta[idw]); idw+=1; weights_.append(b_21)
        idx,w_22  = slice2ker__(weights, idx, info_theta[idw]); idw+=1; weights_.append(w_22)
        idx,b_22  = slice2ker__b(weights,idx, info_theta[idw]); idw+=1; weights_.append(b_22)
        
        idx,w_2   = slice2ker__(weights, idx, info_theta[idw]); idw+=1; weights_.append(w_2)
        idx,b_2   = slice2ker__b(weights,idx, info_theta[idw]); idw+=1; weights_.append(b_2)
        idx,w_1   = slice2ker__(weights, idx, info_theta[idw]); idw+=1; weights_.append(w_1)
        idx,b_1   = slice2ker__b(weights,idx, info_theta[idw]); idw+=1; weights_.append(b_1)

        idx,w1x1 = slice2ker__(weights,idx,info_theta[idw]); weights_.append(w1x1)

        c11       =    Conv2dw(inputs, w11,  b=b11, name_='c11')
        r11       = tf.nn.relu(        c11,          name='r11')
        c12       =    Conv2dw(   r11, w12,  b=b12, name_='c12')
        r12       = tf.nn.relu(        c12,          name='r12')

        p1       = tf.layers.average_pooling2d(r12,pool_size=2,strides=2,padding='valid',data_format=d_form,name='p1')
        c21      =    Conv2dw(    p1, w21,  b=b21, name_='c21')
        r21      = tf.nn.relu(        c21,          name='r21')
        c22      =    Conv2dw(   r21, w22,  b=b22, name_='c22')
        r22      = tf.nn.relu(        c22,          name='r22')

        p2       = tf.layers.average_pooling2d(r22,pool_size=2,strides=2,padding='valid',data_format=d_form,name='p2')
        c31      =    Conv2dw(    p2, w31,  b=b31, name_='c31')
        r31      = tf.nn.relu(        c31,          name='r31')
        c32      =    Conv2dw(   r31, w32,  b=b32, name_='c32')
        r32      = tf.nn.relu(        c32,          name='r32')
        t2       = tf.nn.conv2d_transpose(r32, tf.ones(ch_convt21),shape_lv1, strides=[1,1,2,2], padding='VALID',data_format=d_form_,name='t2')

        c_21      =    Conv2dw(r22+t2, w_21,  b=b_21, name_='c_21')
        r_21      = tf.nn.relu(       c_21,           name='r_21')
        c_22      =    Conv2dw( r_21, w_22,  b=b_22, name_='c_22')
        r_22      = tf.nn.relu(       c_22,           name='r_22')

        t1       = tf.nn.conv2d_transpose(r_22, tf.ones(ch_convt10),shape_lv0, strides=[1,1,2,2], padding='VALID',data_format=d_form_,name='t1')

        c_2      =    Conv2dw(  r12+t1, w_2,  b=b_2, name_='c_2')
        r_2      = tf.nn.relu(          c_2,          name='r_2')
        c_1      =    Conv2dw(     r_2, w_1,  b=b_1, name_='c_1')
        r_1      = tf.nn.relu(          c_1,          name='r_1')


        out_k_ri = tf.nn.tanh( Conv2dw(r_1+r11,w1x1, name_='cout'),name='tahn')
        return out_k_ri, weights_
####################################################################################

def tmp_net(inputs, n_out, weights=[],info_theta=[]):
    if weights==[]:
        with tf.variable_scope('dummy'):
            out_k_ri =  tf.layers.conv2d(inputs,filters=n_out, kernel_size=(3,3),strides=(1,1), padding="SAME", use_bias=False, data_format='channels_first', kernel_initializer=li.xavier_initializer(), name='degug_conv' )
        return out_k_ri
    else:
        idx,w1 = slice2ker__(weights,0, info_theta[0])
        weights_ = [w1]

        out_k_ri = Conv2dw(inputs,w1,name_='G1')
        return out_k_ri, weights_
####################################################################################


#def Unet_(inputs, n_out, weights=[], info_theta=[], nCh=64):
#    """
#    inputs : tensor or placeholder input [batch, channel, row, col]
#    """
#    if weights==[]:
#        #out_k_ri = tf.tanh( unet_wo_BN(inputs, n_out, scope='dummy'))
#        out_k_ri = unet_wo_BN(inputs, n_out, scope='dummy')
#        return out_k_ri
#    else:
#        weights_ = []
#        #
#        idx=0
#        cur, w_ =  slice2ker__(weights, 0, info_theta[idx]); idx=idx+1
#        cur, b_ = slice2ker__b(weights, cur, info_theta[idx]); idx=idx+1 
#        down0_1     =  tf.nn.relu( Conv2dw( inputs, w_, b=b_, name_='lv0_1'));  weights_.append(w_); weights_.append(b_)
#        cur, w_ =  slice2ker__(weights, cur, info_theta[idx]); idx=idx+1
#        cur, b_ = slice2ker__b(weights, cur, info_theta[idx]); idx=idx+1
#        down0_2     =  tf.nn.relu( Conv2dw(down0_1, w_, b=b_, name_='lv0_2'));  weights_.append(w_); weights_.append(b_)
#
#        
##        pool1       = Pool2d(  down0_2, name='lv1_p')
##        cur, w_ =  slice2ker__(weights, cur, info_theta[idx]); idx=idx+1
##        cur, b_ = slice2ker__b(weights, cur, info_theta[idx]); idx=idx+1
##        down1_1     =  tf.nn.relu( Conv2dw(  pool1, w_, b=b_, name_='lv1_1'));  weights_.append(w_); weights_.append(b_)
##        cur, w_ =  slice2ker__(weights, cur, info_theta[idx]); idx=idx+1
##        cur, b_ = slice2ker__b(weights, cur, info_theta[idx]); idx=idx+1
##        down1_2     =  tf.nn.relu( Conv2dw(down1_1, w_, b=b_, name_='lv1_2'));  weights_.append(w_); weights_.append(b_)
##
##        pool2       = Pool2d(  down1_2, name='lv2_p')
##        cur, w_ =  slice2ker__(weights, cur, info_theta[idx]); idx=idx+1
##        cur, b_ = slice2ker__b(weights, cur, info_theta[idx]); idx=idx+1
##        down2_1     =  tf.nn.relu( Conv2dw(  pool2, w_, b=b_, name_='lv2_1'));  weights_.append(w_); weights_.append(b_)
##        cur, w_ =  slice2ker__(weights, cur, info_theta[idx]); idx=idx+1
##        cur, b_ = slice2ker__b(weights, cur, info_theta[idx]); idx=idx+1
##        down2_2     =  tf.nn.relu( Conv2dw(down2_1, w_, b=b_, name_='lv2_2'));  weights_.append(w_); weights_.append(b_)
# 
##        pool3       = Pool2d(  down2_2, name='lv3_p')
##        cur, w_ =  slice2ker__(weights, cur, info_theta[idx]); idx=idx+1
##        cur, b_ = slice2ker__b(weights, cur, info_theta[idx]); idx=idx+1
##        down3_1     =  tf.nn.relu( Conv2dw(  pool3, w_, b=b_, name_='lv3_1'));  weights_.append(w_); weights_.append(b_)
##        cur, w_ =  slice2ker__(weights, cur, info_theta[idx]); idx=idx+1
##        cur, b_ = slice2ker__b(weights, cur, info_theta[idx]); idx=idx+1
##        down3_2     =  tf.nn.relu( Conv2dw(down3_1, w_, b=b_, name_='lv3_2'));  weights_.append(w_); weights_.append(b_)
##        
##        pool4       = Pool2d(  down3_2, name='lv4_p')
##        cur, w_ =  slice2ker__(weights, cur, info_theta[idx]); idx=idx+1
##        cur, b_ = slice2ker__b(weights, cur, info_theta[idx]); idx=idx+1
##        down4_1     =  tf.nn.relu( Conv2dw(  pool4, w_, b=b_, name_='lv4_1'));  weights_.append(w_); weights_.append(b_)
##        cur, w_ =  slice2ker__(weights, cur, info_theta[idx]); idx=idx+1
##        cur, b_ = slice2ker__b(weights, cur, info_theta[idx]); idx=idx+1
##        down4_2     =  tf.nn.relu( Conv2dw(down4_1, w_, b=b_, name_='lv4_2'));  weights_.append(w_); weights_.append(b_)
##        #
##        cur, w_     =  slice2ker__(weights, cur, info_theta[idx]); idx=idx+1
##        up4         = Conv2dTw( down4_2, w_, nCh*16, nCh*8, name_='lv4__up');  weights_.append(w_);
##        CC3         = tf.concat([down3_2, up4], axis=ch_dim, name='CC3')
##        cur, w_     =  slice2ker__(weights, cur, info_theta[idx]); idx=idx+1
##        cur, b_     = slice2ker__b(weights, cur, info_theta[idx]); idx=idx+1
##        up3_1       = tf.nn.relu( Conv2dw(  CC3, w_, b=b_, name_='lv3__1'));  weights_.append(w_); weights_.append(b_)
##        cur, w_     =  slice2ker__(weights, cur, info_theta[idx]); idx=idx+1
##        cur, b_     = slice2ker__b(weights, cur, info_theta[idx]); idx=idx+1
##        up3_2       = tf.nn.relu( Conv2dw(up3_1, w_, b=b_, name_='lv3__2'));  weights_.append(w_); weights_.append(b_)
#        
##        cur, w_     =  slice2ker__(weights, cur, info_theta[idx]); idx=idx+1
##        up3         = Conv2dTw(  up3_2, w_, nCh*8, nCh*4, name_='lv3__up');  weights_.append(w_)
##        CC2         = tf.concat([down2_2, up3], axis=ch_dim, name='CC2')
##        cur, w_     =  slice2ker__(weights, cur, info_theta[idx]); idx=idx+1
##        cur, b_     = slice2ker__b(weights, cur, info_theta[idx]); idx=idx+1
##        up2_1       = tf.nn.relu( Conv2dw(  CC2, w_, b=b_, name_='lv2__1'));  weights_.append(w_); weights_.append(b_)
##        cur, w_     =  slice2ker__(weights, cur, info_theta[idx]); idx=idx+1
##        cur, b_     = slice2ker__b(weights, cur, info_theta[idx]); idx=idx+1
##        up2_2       = tf.nn.relu( Conv2dw(up2_1, w_, b=b_, name_='lv2__2'));  weights_.append(w_); weights_.append(b_)
#        
##        cur, w_     =  slice2ker__(weights, cur, info_theta[idx]); idx=idx+1
##        #up2         = Conv2dTw(   up2_2, w_, nCh*4, nCh*2, name_='lv2__up');  weights_.append(w_);      
##        up2         = Conv2dTw(   down2_2, w_, nCh*4, nCh*2, name_='lv2__up');  weights_.append(w_);      
##        CC1         = tf.concat([down1_2, up2], axis=ch_dim, name='CC1')
##        cur, w_     =  slice2ker__(weights, cur, info_theta[idx]); idx=idx+1
##        cur, b_     = slice2ker__b(weights, cur, info_theta[idx]); idx=idx+1
##        up1_1       = tf.nn.relu( Conv2dw(  CC1, w_, b=b_, name_='lv1__1'));  weights_.append(w_); weights_.append(b_)
##        cur, w_     =  slice2ker__(weights, cur, info_theta[idx]); idx=idx+1
##        cur, b_     = slice2ker__b(weights, cur, info_theta[idx]); idx=idx+1
##        up1_2       = tf.nn.relu( Conv2dw(up1_1, w_, b=b_, name_='lv1__2'));  weights_.append(w_); weights_.append(b_)
##
##        cur, w_     =  slice2ker__(weights, cur, info_theta[idx]); idx=idx+1
##        up1         = Conv2dTw(   up1_2, w_, nCh*2, nCh, name_='lv1__up');  weights_.append(w_)
##        CC0         = tf.concat([down0_2, up1], axis=ch_dim, name='CC0')
#        cur, w_     =  slice2ker__(weights, cur, info_theta[idx]); idx=idx+1
#        cur, b_     = slice2ker__b(weights, cur, info_theta[idx]); idx=idx+1
#        #up0_1       = tf.nn.relu( Conv2dw(  CC0, w_, b=b_, name_='lv0__1'));  weights_.append(w_); weights_.append(b_)
#        up0_1       = tf.nn.relu( Conv2dw(  down0_2, w_, b=b_, name_='lv0__1'));  weights_.append(w_); weights_.append(b_)
#        cur, w_     =  slice2ker__(weights, cur, info_theta[idx]); idx=idx+1
#        cur, b_     = slice2ker__b(weights, cur, info_theta[idx]); idx=idx+1
#        up0_2       = tf.nn.relu( Conv2dw(up0_1, w_, b=b_, name_='lv0__2'));  weights_.append(w_); weights_.append(b_)
#        cur, w_     =  slice2ker__(weights, cur, info_theta[idx]);
#        # out_k_ri    = tf.tanh( Conv2dw(  up0_2, w_, b=[], name_='conv1x1') )
#        out_k_ri    = Conv2dw(  up0_2, w_, b=[], name_='conv1x1') ;  weights_.append(w_);
#
#        return out_k_ri, weights_
def Gnet_(inputs, n_out, weights=[], info_theta=[],nCh=32):
    """ x : tensor or placeholder input [batch, row, col, channel]
    n_out : numbet of output channel
    """
    if weights==[]:
        #out_k_ri = tf.tanh( gnet(inputs, n_out, True, nCh=8, scope='dummy'))
        out_k_ri =  gnet(inputs, n_out, True, nCh=nCh, scope='dummy')
        return out_k_ri
    else:
        idx, w1 = slice2ker__(weights,  0, info_theta[0])
        idx, w2 = slice2ker__(weights,idx, info_theta[1])
        idx, w3 = slice2ker__(weights,idx, info_theta[2])
        
        weights_ = [w1,w2,w3]
        ##
        F1_1 = Conv2dw(inputs, w1, name_='G1')
        F1_2 = tf.nn.relu(F1_1, name='G1R')
        F2_1 = Conv2dw(  F1_2, w2, name_='G2')
        F2_2 = tf.nn.relu(F2_1, name='G2R')
        out_k_ri = Conv2dw(  F2_2, w3, name_='G3')
        return out_k_ri, weights_

def c2ws(weights, info_theta):
    idx = 0
    w_outs = []
    for a_info_theta in info_theta:
        idx, a_w = slice2ker__(weights, idx, a_info_theta)
        w_outs.append(a_w)
    return w_outs

def Unet_wo_BN(inputs, n_out, weights=[], info_theta=[],nCh=32):
    """ x : tensor or placeholder input [batch, row, col, channel]
    n_out : numbet of output channel
    """
    nB=1
    shape_lv0 = [nB,nCh, int(inputs.shape[2]),int(inputs.shape[3])]
    shape_lv1 = [nB,nCh*2, int(int(inputs.shape[2])/2),int(int(inputs.shape[3])/2)]
    ch_convt21= [2,2,nCh*2,nCh*4]
    ch_convt10= [2,2,  nCh,nCh*2]
    if weights==[]: 
        reg_=None
        with tf.variable_scope('dummy', reuse=False): 
            down0_1     =    CR(   inputs,  nCh, name='lv0_1', reg=reg_)
            down0_2     =    CR(  down0_1,  nCh, name='lv0_2', reg=reg_)
        
            pool1       = Pool2d(  down0_2, name='lv1_p')
            down1_1     =    CR(    pool1,nCh*2, name='lv1_1', reg=reg_) 
            down1_2     =    CR(  down1_1,nCh*2, name='lv1_2', reg=reg_)
            
            pool2       = Pool2d(  down1_2, name='lv2_p')
            down2_1     =    CR(    pool2,nCh*4,  name='lv2_1', reg=reg_) 
            down2_2     =    CR(  down2_1,nCh*4,  name='lv2_2', reg=reg_)
            up2     = tf.nn.conv2d_transpose(down2_2, tf.ones(ch_convt21), shape_lv1,strides=[1,1,2,2],padding='VALID',data_format=d_form_,name='lv2__up')
            
            CC1         = tf.concat([down1_2, up2], axis=ch_dim, name='CC2')
            up1_1       =    CR(      CC1, nCh*2,  name='lv1__1', reg=reg_)
            up1_2       =    CR(    up1_1, nCh*2,  name='lv1__2', reg=reg_)
            up1     = tf.nn.conv2d_transpose(up1_2, tf.ones(ch_convt10), shape_lv0,strides=[1,1,2,2],padding='VALID',data_format=d_form_,name='lv1__up')
            
            CC0         = tf.concat([down0_2, up1], axis=ch_dim, name='CC1')
            up0_1       =    CR(      CC0,   nCh,  name='lv0__1', reg=reg_)
            up0_2       =    CR(    up0_1,   nCh,  name='lv0__2', reg=reg_)
            
            return Conv1x1(   up0_2, n_out,name='conv1x1')
    else:
        _w = c2ws(weights,info_theta)
        idx = 0
        down0_1 =  tf.nn.relu( Conv2dw(  inputs, _w[idx], name_='lv0_1') ); idx+=1
        down0_2 =  tf.nn.relu( Conv2dw( down0_1, _w[idx], name_='lv0_2') ); idx+=1

        pool1   =  Pool2d(  down0_2, name='lv1_p')
        down1_1 =  tf.nn.relu( Conv2dw(   pool1, _w[idx], name_='lv1_1') ); idx+=1
        down1_2 =  tf.nn.relu( Conv2dw( down1_1, _w[idx], name_='lv1_2') ); idx+=1
        
        pool2   =  Pool2d(  down1_2, name='lv2_p')
        down2_1 =  tf.nn.relu( Conv2dw(   pool2, _w[idx], name_='lv2_1') ); idx+=1
        down2_2 =  tf.nn.relu( Conv2dw( down2_1, _w[idx], name_='lv2_2') ); idx+=1
        up2     = tf.nn.conv2d_transpose(down2_2, tf.ones(ch_convt21), shape_lv1,strides=[1,1,2,2],padding='VALID',data_format=d_form_,name='lv2__up')
       
        CC1     =  tf.concat([down1_2, up2], axis=ch_dim, name='CC2')
        up1_1   =  tf.nn.relu( Conv2dw(   CC1, _w[idx], name_='lv1__1') ); idx+=1
        up1_2   =  tf.nn.relu( Conv2dw( up1_1, _w[idx], name_='lv1__2') ); idx+=1
        up1     = tf.nn.conv2d_transpose(up1_2, tf.ones(ch_convt10), shape_lv0,strides=[1,1,2,2],padding='VALID',data_format=d_form_,name='lv1__up')     
        
        CC0     =  tf.concat([down0_2, up1], axis=ch_dim, name='CC1')
        up0_1   =  tf.nn.relu( Conv2dw(   CC0, _w[idx], name_='lv0__1') ); idx+=1
        up0_2   =  tf.nn.relu( Conv2dw( up0_1, _w[idx], name_='lv0__2') ); idx+=1
        
        out_k_ri = Conv2dw(  up0_2, _w[idx], name_='conv1x1')
        return out_k_ri, _w


#
#def Gnet2_(inputs, n_out, weights=[], info_theta=[],nCh=32):
#    """ x : tensor or placeholder input [batch, row, col, channel]
#    n_out : numbet of output channel
#    """
#    if weights==[]:
#        out_k_ri =  gnet2(inputs, n_out, True, nCh=nCh, scope='dummy')
#        return out_k_ri
#    else:
#        idx, w1 = slice2ker__(weights,  0, info_theta[0])
#        idx, w2 = slice2ker__(weights,idx, info_theta[1])
#        idx, w3 = slice2ker__(weights,idx, info_theta[2])
#        idx, w4 = slice2ker__(weights,idx, info_theta[3])
#        idx, w5 = slice2ker__(weights,idx, info_theta[4])
#     
#        weights_ = [w1,w2,w3,w4,w5]
#        ##
#        F1_1 = Conv2dw(inputs, w1, name_='G1')
#        F1_2 = tf.nn.relu(F1_1, name='G1R')
#        F2_1 = Conv2dw(  F1_2, w2, name_='G2')
#        F2_2 = tf.nn.relu(F2_1, name='G2R')
#        F3_1 = Conv2dw(  F2_2, w3, name_='G3')
#        F3_2 = tf.nn.relu(F3_1, name='G3R')      
#        F4_1 = Conv2dw(  F3_2, w4, name_='G4')
#        F4_2 = tf.nn.relu(F4_1, name='G4R')
#        out_k_ri = Conv2dw(  F4_2, w5, name_='G5')
#        return out_k_ri, weights_
#

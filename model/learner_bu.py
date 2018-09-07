import tensorflow as tf
#import tensorlayer as tl
#from tensorlayer.layers import InputLayer, Conv2d, MaxPool2d, DeConv2d, ConcatLayer
import tensorflow.contrib.layers as li
from ipdb import set_trace as st
from util.netUtil import unet, gnet, Conv2dw, unet_wo_BN, Pool2d, Conv2dT, Conv2dTw
from util.util import myTFfftshift2, tf_imgri2kri, tf_kri2imgri, slice2ker__, slice2ker__b
dtype = tf.float32
d_form  = 'channels_first'
d_form_ = 'NCHW'
ch_dim  = 1
 
def Unet_(inputs, n_out, weights=[],info_theta=[]):
    nCh=64
    nB=1
    shape_lv0 = [nB,nCh, int(inputs.shape[2]),int(inputs.shape[3])]
    if weights==[]:
        init = li.xavier_initializer()
        with tf.variable_scope('dummy'):
            c11 =  tf.layers.conv2d(inputs,filters=nCh, kernel_size=(3,3),strides=(1,1), padding="SAME", use_bias=True, data_format=d_form, kernel_initializer=init,name='c11')
            r11 =  tf.nn.relu(c11,name='r11')
            c12 =  tf.layers.conv2d(r11,filters=nCh, kernel_size=(3,3),strides=(1,1), padding="SAME", use_bias=True, data_format=d_form, kernel_initializer=init,name='c12')
            r12 =  tf.nn.relu(c12,name='r12')
            
            if True:
                p1 =  tf.layers.average_pooling2d(r12,pool_size=2,strides=2,padding='valid',data_format=d_form,name='p1')
                c2 =  tf.layers.conv2d(p1,filters=nCh, kernel_size=(3,3),strides=(1,1), padding="SAME", use_bias=True, data_format=d_form, kernel_initializer=init,name='c2')
                r2 = tf.nn.relu(c2,name='r2')
                t1 =  tf.nn.conv2d_transpose(r2, tf.ones([2,2,nCh,nCh]), shape_lv0,strides=[1,1,2,2],padding='VALID',data_format=d_form_,name='t1')
            
            #c_2 =  tf.layers.conv2d(r2,filters=nCh, kernel_size=(3,3),strides=(1,1), padding="SAME", use_bias=True, data_format=d_form, kernel_initializer=init)
            #r_2 = tf.nn.relu(c_2,name='r_2')
            c_1 =  tf.layers.conv2d(r12+t1, filters=nCh, kernel_size=(3,3),strides=(1,1), padding="SAME", use_bias=True, data_format=d_form, kernel_initializer=init)
            r_1 = tf.nn.relu(c_1,name='r_1')

            out_k_ri =  tf.layers.conv2d(r_1,filters=n_out, kernel_size=(1,1),strides=(1,1), padding="SAME", use_bias=False, data_format=d_form, kernel_initializer=init, name='cout')

        return out_k_ri 
    else:
        idw=0; idx=0
        idx,w11   = slice2ker__(weights,   0, info_theta[idw]); idw+=1
        idx,b11   = slice2ker__b(weights,idx, info_theta[idw]); idw+=1
        idx,w12   = slice2ker__(weights, idx, info_theta[idw]); idw+=1
        idx,b12   = slice2ker__b(weights,idx, info_theta[idw]); idw+=1
        idx,w2   = slice2ker__(weights, idx, info_theta[idw]); idw+=1
        idx,b2   = slice2ker__b(weights,idx, info_theta[idw]); idw+=1
        
        #idx,w_2   = slice2ker__(weights, idx, info_theta[idw]); idw+=1
        #idx,b_2   = slice2ker__b(weights,idx, info_theta[idw]); idw+=1
        idx,w_1   = slice2ker__(weights, idx, info_theta[idw]); idw+=1
        idx,b_1   = slice2ker__b(weights,idx, info_theta[idw]); idw+=1
        

        idx,w1x1 = slice2ker__(weights,idx,info_theta[idw])
        weights_ = [w11,b11,w12,b12,w2,b2,w_1,b_1,  w1x1]

        c11       =    Conv2dw(inputs, w11,  b=b11, name_='c11')
        r11       = tf.nn.relu(        c11,          name='r11')
        c12       =    Conv2dw(   r11, w12,  b=b12, name_='c12')
        r12       = tf.nn.relu(        c12,          name='r12')

        p1       = tf.layers.average_pooling2d(r12,pool_size=2,strides=2,padding='valid',data_format=d_form,name='p1')
        c2       =    Conv2dw(    p1,  w2,   b=b2, name_='c2')
        r2       = tf.nn.relu(         c2,          name='r2')
        t1       = tf.nn.conv2d_transpose(r2, tf.ones([2,2,nCh,nCh]),shape_lv0, strides=[1,1,2,2], padding='VALID',data_format=d_form_,name='t1')
#        c_2      =    Conv2dw(    r2, w_2,  b=b_2, name_='c_2')
#        r_2      = tf.nn.relu(        c_2,          name='r_2')
        c_1      =    Conv2dw(  r12+t1, w_1,  b=b_1, name_='c_1')
        r_1      = tf.nn.relu(        c_1,          name='r_1')


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
def Gnet_(inputs, n_out, weights=[], info_theta=[]):
    """ x : tensor or placeholder input [batch, row, col, channel]
    n_out : numbet of output channel
    """
    if weights==[]:
        #out_k_ri = tf.tanh( gnet(inputs, n_out, True, nCh=8, scope='dummy'))
        out_k_ri =  gnet(inputs, n_out, True, nCh=8, scope='dummy')
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
        #out_k_ri = tf.tanh(out_k_ri)
        return out_k_ri, weights_


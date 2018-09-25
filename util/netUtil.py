import tensorflow as tf
#import tensorlayer as tl
#from tensorlayer.layers import InputLayer, Conv2d, MaxPool2d, DeConv2d, ConcatLayer
import tensorflow.contrib.layers as li
from ipdb import set_trace as st
from util.util import myTFfftshift2, slice2ker, DIM2CH2, DIM2CH4, tf_imgri2kri, tf_kri2imgri
dtype = tf.float32
d_form  = 'channels_first'
d_form_ = 'NCHW'
ch_dim  = 1

def Conv2dw(x, w, b=[], name_='', reg=None):
    wx =  tf.nn.conv2d(x,w,[1,1,1,1], "SAME", data_format=d_form_, name=name_)
    if b==[]:
        return wx
    else:
        return tf.nn.bias_add(wx,b,data_format=d_form_)

def Conv2d(x, ch_out, name, reg=None, use_bias=False):
    return tf.layers.conv2d(x, filters=ch_out, kernel_size=(3,3), strides=(1,1), padding="SAME", use_bias=use_bias,data_format=d_form, kernel_initializer=li.xavier_initializer(), kernel_regularizer=reg, name="".join((name,"_Conv")))

def BN(x, is_Training, name):
    scope=name+'_bn'
    return tf.cond(is_Training, lambda: li.batch_norm(x, is_training=True, epsilon=0.000001, center=True, data_format=d_form_, updates_collections=None, scope=scope),
            lambda: li.batch_norm(x, is_training=False, updates_collections=None, epsilon=0.000001, center=True, data_format=d_form_,scope=scope, reuse=True) )

def BN_(x, name):
    scope=name+'_bn'
    return li.batch_norm(x, is_training=True, epsilon=0.000001, center=True, data_format=d_form_, updates_collections=None, scope=scope)

def Pool2d(x, name):#ch_out, name):
    return tf.layers.average_pooling2d(x, pool_size=2, strides=2, padding='valid', data_format=d_form, name=name)
    #return tf.layers.max_pooling2d(x, pool_size=2, strides=2, padding='valid', name=name)


def Conv2dT(x, ch_out, name, kernel_sz=[2,2], strides=[2,2],padding='valid'):
    return tf.layers.conv2d_transpose(x, filters=ch_out, kernel_size=kernel_sz, strides=strides, data_format=d_form,use_bias=False, kernel_initializer=li.xavier_initializer(), name=name)

def Conv2dTw(x, w2x2, nB=1, name_=''):
    sz        = x.shape
    out_shape = [nB,int(int(sz[1])/2), int(sz[2])*2, int(sz[3])*2]
    #w2x2 : = tf.ones([2,2,ch_out, ch_in ])
    return tf.nn.conv2d_transpose(x, w2x2, out_shape, strides=[1,1,2,2],padding='VALID', data_format=d_form_,name=name_)


def ReLU(x,name, BN_tag=True):
    return tf.nn.relu(x, name="".join((name,"_R")))
    #return tf.nn.leaky_relu(BN_(x,name=name), name="".join((name,"_R")),alpha=0.2)

def Conv1x1(x, ch_out, name, reg=None):
    return tf.layers.conv2d(x, filters=ch_out, kernel_size=(1,1), strides=(1,1), padding="SAME", use_bias=False, data_format=d_form, kernel_initializer=li.xavier_initializer(), kernel_regularizer=reg, name="".join((name,"_Conv1x1")))

def CBR(x, ch_out, is_Training, name, reg=None, r_DR=0, no_BN=False):
    return ReLU( BN( Conv2d(x, ch_out, name, reg, use_bias=no_BN), is_Training, name), name)
    #if r_DR==0:
    #    if no_BN:
    #        return ReLU( Conv2d(x, ch_out, name, reg, use_bias=no_BN), name)
    #    else:
    #        return ReLU( BN( Conv2d(x, ch_out, name, reg), name, BN_valid ), name   )
    #else:
    #    if no_BN:
    #        return tf.nn.dropout( ReLU( Conv2d( x, ch_out, name, reg, use_bias=no_BN), name), r_DR )
    #    else:
    #        return tf.nn.dropout( ReLU( BN( Conv2d( x, ch_out, name, reg), name, BN_valid), name), r_DR )
def CR(x, ch_out, name, reg=None):
    return ReLU( Conv2d(x, ch_out, name, reg, use_bias=False), name)
    #if r_DR==0:
    #    if no_BN:
    #        return ReLU( Conv2d(x, ch_out, name, reg, use_bias=no_BN), name)
    #    else:
    #        return ReLU( BN( Conv2d(x, ch_out, name, reg), name, BN_valid ), name   )
    #else:
    #    if no_BN:
    #        return tf.nn.dropout( ReLU( Conv2d( x, ch_out, name, reg, use_bias=no_BN), name), r_DR )
    #    else:
    #        return tf.nn.dropout( ReLU( BN( Conv2d( x, ch_out, name, reg), name, BN_valid), name), r_DR )


def unet(inp, n_out, is_Training, nCh=64, name_='', reg_=None, reuse=False, scope='net'):
    ## Unet def goes here
    with tf.variable_scope(scope, reuse=reuse): 
        down0_1     =    CBR(      inp,  nCh, is_Training, name=name_+'lv0_1', reg=reg_)
        down0_2     =    CBR(  down0_1,  nCh, is_Training, name=name_+'lv0_2', reg=reg_)
    
        pool1       = Pool2d(  down0_2,nCh*2, name=name_+'lv1_p')
        down1_1     =    CBR(    pool1,nCh*2, is_Training, name=name_+'lv1_1', reg=reg_) 
        down1_2     =    CBR(  down1_1,nCh*2, is_Training, name=name_+'lv1_2', reg=reg_)
        
        pool2       = Pool2d(  down1_2,nCh*4, name=name_+'lv2_p')
        down2_1     =    CBR(    pool2,nCh*4, is_Training, name=name_+'lv2_1', reg=reg_) 
        down2_2     =    CBR(  down2_1,nCh*4, is_Training, name=name_+'lv2_2', reg=reg_)
        
        pool3       = Pool2d(  down2_2,nCh*8, name=name_+'lv3_p')
        down3_1     =    CBR(    pool3,nCh*8, is_Training, name=name_+'lv3_1', reg=reg_)
        down3_2     =    CBR(  down3_1,nCh*8, is_Training, name=name_+'lv3_2', reg=reg_)
        
        pool4       = Pool2d(  down3_2,nCh*16, name=name_+'lv4_p')
        down4_1     =    CBR(    pool4,nCh*16, is_Training, name=name_+'lv4_1', reg=reg_) 
        down4_2     =    CBR(  down4_1,nCh*16, is_Training, name=name_+'lv4_2', reg=reg_)
        up4         = Conv2dT( down4_2,nCh*8, name=name_+'lv4__up')
        
        CC3         = tf.concat([down3_2, up4], axis=ch_dim,name=name_+'CC4')
        up3_1       =    CBR(      CC3, nCh*8, is_Training, name=name_+'lv3__1',reg=reg_)
        up3_2       =    CBR(    up3_1, nCh*8, is_Training, name=name_+'lv3__2',reg=reg_)
        up3         = Conv2dT(   up3_2, nCh*4, name=name_+'lv3__up')
        
        CC2         = tf.concat([down2_2, up3], axis=ch_dim, name=name_+'CC3')
        up2_1       =    CBR(      CC2, nCh*4, is_Training, name=name_+'lv2__1', reg=reg_)
        up2_2       =    CBR(    up2_1, nCh*4, is_Training, name=name_+'lv2__2', reg=reg_)
        up2         = Conv2dT(   up2_2, nCh*2, name=name_+'lv2__up')
        
        CC1         = tf.concat([down1_2, up2], axis=ch_dim, name=name_+'CC2')
        up1_1       =    CBR(      CC1, nCh*2, is_Training, name=name_+'lv1__1', reg=reg_)
        up1_2       =    CBR(    up1_1, nCh*2, is_Training, name=name_+'lv1__2', reg=reg_)
        up1         = Conv2dT(   up1_2,   nCh, name=name_+'lv1__up')
        
        CC0         = tf.concat([down0_2, up1], axis=ch_dim, name=name_+'CC1')
        up0_1       =    CBR(      CC0,   nCh, is_Training, name=name_+'lv0__1', reg=reg_)
        up0_2       =    CBR(    up0_1,   nCh, is_Training, name=name_+'lv0__2', reg=reg_)
        
        return Conv1x1(   up0_2, n_out,name=name_+'conv1x1')
def gnet(inp, n_out, is_Training, nCh=32, name_='', reg_=None, reuse=False,scope='net'):
    F1h = 5
    F1w = 2
    F2h = 1
    F2w = 1
    F3h = 3
    F3w = 2
    nwF1 = nCh
    nwF2 = int(nCh/4)
    use_bias = False
    with tf.variable_scope(scope, reuse=reuse):
       F1_1 = tf.layers.conv2d(inp, filters=nwF1, kernel_size=(F1h,F1w), strides=(1,1), padding="SAME", use_bias=use_bias,data_format=d_form, kernel_initializer=li.xavier_initializer(), kernel_regularizer=reg_, name="".join((name_,"_Conv")))
       #F1_1B= tf.layers.batch_normalization(F1_1,axis=ch_dim,training=is_Training)
       F1_2 = tf.nn.relu(F1_1, name="".join((name_,"_ReLU")))
       #
       F2_1 =  tf.layers.conv2d(F1_2, filters=nwF2, kernel_size=(F2h,F2w), strides=(1,1), padding="SAME", use_bias=use_bias,data_format=d_form, kernel_initializer=li.xavier_initializer(), kernel_regularizer=reg_, name="".join((name_,"_Conv2")))
       #F2_1B= tf.layers.batch_normalization(F2_1,axis=ch_dim,training=is_Training)
       F2_2 = tf.nn.relu(F2_1, name="".join((name_,"_ReLU2")))
       #
       F3   =  tf.layers.conv2d(F2_2, filters=n_out, kernel_size=(F3h,F3w), strides=(1,1), padding="SAME", use_bias=use_bias,data_format=d_form, kernel_initializer=li.xavier_initializer(), kernel_regularizer=reg_, name="".join((name_,"_Conv3")))

       return F3


def gnet2(inp, n_out, is_Training, nCh=32, name_='', reg_=None, reuse=False,scope='net'):
    F1h = 5
    F1w = 2
    F2h = 1
    F2w = 1
    F3h = 3
    F3w = 2
    nwF1 = nCh
    nwF2 = nCh
    use_bias = False
    with tf.variable_scope(scope, reuse=reuse):
       F1_1 = tf.layers.conv2d(inp, filters=nwF1, kernel_size=(F1h,F1w), strides=(1,1), padding="SAME", use_bias=use_bias,data_format=d_form, kernel_initializer=li.xavier_initializer(), kernel_regularizer=reg_, name="".join((name_,"_Conv")))
       F1_2 = tf.nn.relu(F1_1, name="".join((name_,"_ReLU")))
       
       F2_1 =  tf.layers.conv2d(F1_2, filters=nwF2, kernel_size=(F2h,F2w), strides=(1,1), padding="SAME", use_bias=use_bias,data_format=d_form, kernel_initializer=li.xavier_initializer(), kernel_regularizer=reg_, name="".join((name_,"_Conv2")))
       F2_2 = tf.nn.relu(F2_1, name="".join((name_,"_ReLU2")))
       
       F3_1 =  tf.layers.conv2d(F2_2, filters=nwF2, kernel_size=(F3h,F3w), strides=(1,1), padding="SAME", use_bias=use_bias,data_format=d_form, kernel_initializer=li.xavier_initializer(), kernel_regularizer=reg_, name="".join((name_,"_Conv3")))
       F3_2 = tf.nn.relu(F3_1, name="".join((name_,"_ReLU3")))
       
       F4_1 =  tf.layers.conv2d(F3_2, filters=nwF2, kernel_size=(F2h,F2w), strides=(1,1), padding="SAME", use_bias=use_bias,data_format=d_form, kernel_initializer=li.xavier_initializer(), kernel_regularizer=reg_, name="".join((name_,"_Conv4")))
       F4_2 = tf.nn.relu(F4_1, name="".join((name_,"_ReLU4")))      #
       Fout   =  tf.layers.conv2d(F4_2, filters=n_out, kernel_size=(F3h,F3w), strides=(1,1), padding="SAME", use_bias=use_bias,data_format=d_form, kernel_initializer=li.xavier_initializer(), kernel_regularizer=reg_, name="".join((name_,"_Conv5")))

       return Fout



























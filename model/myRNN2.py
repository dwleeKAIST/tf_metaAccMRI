import tensorflow as tf
import tensorflow.contrib.layers as li
from ipdb import set_trace as st
from tensorflow.contrib import rnn#
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import array_ops, math_ops
from util.util import tf_kri2imgri
from util.MyWeight import MyWeight
dtype  = tf.float32
d_form = 'channels_first'
d_form_= 'NCHW'
ch_dim = 1


P=10.0
expP = math_ops.exp(P)
expNegP = math_ops.exp(-P)
eps = 0.00000000001
def preProc(x):
    absX = math_ops.abs(x)
    ge = math_ops.cast(math_ops.greater_equal(absX,expNegP),tf.float32)  # if absX >=e-p
    x1_ge = math_ops.divide(math_ops.log(absX+eps),P)
    x2_ge = math_ops.sign(x)
    lt = math_ops.cast(math_ops.less(absX,expNegP),tf.float32) # otherwise
    x1_lt = array_ops.ones_like(x)*-1
    x2_lt = expP*x
    #
    z1 = x1_ge*ge + x1_lt*lt
    z2 = x2_ge*ge + x2_lt*lt
    return z1,z2
 

class myLSTM:
    def __init__(self, dummy_theta_shapes, Learner, opt):
        self.nHidden1 = opt.nHidden
        self.Learner  = Learner
        self.lambda_loss = opt.lambda_loss
        self.ngf      = opt.ngf
        #
        self.nInput2  = 8#4  #opt.nHidden+1
        self.nHidden2 = int(opt.ntheta)
        self.name     = 'myDoubleLSTM'
        self.nBtheta  = int(opt.ntheta)
        self.moment   = 0.9
        with tf.variable_scope(self.name):
            self.WF2 = tf.get_variable('WF2',[self.nInput2, 1],initializer=tf.contrib.layers.xavier_initializer())
            self.WI2 = tf.get_variable('WI2',[self.nInput2, 1],initializer=tf.contrib.layers.xavier_initializer())
            self.WO2 = tf.get_variable('WO2',[self.nInput2, 1],initializer=tf.contrib.layers.xavier_initializer())
            self.bF2 = tf.Variable(5.,name='bF2')
            self.bI2 = tf.Variable(-5.,name='bI2')
            self.bO2 = tf.Variable(5.,name='bO2')
            #self.bO2 = tf.Variable(-5.,name='bO2')
        self.THETA = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        ## Learner parameter
        self.nCh_out = opt.nCh_out#2# 16
        self.info_theta = dummy_theta_shapes
        #self.w = MyWeight( 

    def LSTM2_f(self, x_grad, x_loss, p_grad, p_loss, c_prev, h_prev, f_prev, i_prev, o_prev):
        ''' f_prev and i_prev = h_prev, later they are concatenated'''
        f_cur = tf.sigmoid( tf.matmul(  tf.concat([ h_prev,x_loss, x_grad,p_grad,p_loss, f_prev], axis=1), self.WF2 ) + self.bF2 )
        i_cur = tf.sigmoid( tf.matmul(  tf.concat([ h_prev,x_loss, x_grad,p_grad,p_loss, i_prev], axis=1), self.WI2 ) + self.bI2 )
        o_cur = tf.sigmoid( tf.matmul(  tf.concat([ h_prev,x_loss, x_grad,p_grad,p_loss, o_prev], axis=1), self.WO2 ) + self.bO2 )
        c_cur = f_cur*c_prev - i_cur*(x_grad+h_prev)
        h_cur = o_cur*(x_grad+h_prev)
        return c_cur, h_cur, f_cur, i_cur, o_cur

    ''' main forward '''
    def f(self, k_shot, x, y, c, h, f, i,o,is_Training): 
        
        h1_prev = []
        losses = []
        for ik in range(k_shot):
            loss, grad_thetas = self.learner_df(x,y,c) # loss : 1x1, grad_thetas : 2112 x 1 x1        
            losses.append(loss)
            
            _ploss        = tf.stack( preProc(loss))
            ploss         = tf.tile(_ploss[tf.newaxis,:], [grad_thetas.shape[0],1])  #    2112 x 2
            pGrad         = tf.stack(preProc(grad_thetas),axis=1)        # 2112 x 2
            rep_loss = tf.tile(loss[tf.newaxis,tf.newaxis],[grad_thetas.shape[0],1])
            c, h, f, i, o = self.LSTM2_f(grad_thetas,rep_loss, pGrad[:,:,0], ploss, c, h, f, i, o)
        total_train_loss = tf.add_n(losses)
        return c, h, f, i, o, total_train_loss, grad_thetas
             
    def learner_f(self, x, y,theta):
        net_out, theta_ = self.Learner(x, self.nCh_out, weights=theta, info_theta=self.info_theta, nCh=self.ngf)
        loss = tf.losses.mean_squared_error(labels=y, predictions=net_out)
        return net_out, theta_, loss

    def learner_df(self, x, y, theta):
        net_out, theta_, loss = self.learner_f(x,y,theta)
        loss = loss*self.lambda_loss
        grad_theta      = tf.gradients(loss,theta_)
        grad_thetas     = [[[]]]
        grad_theta_ = [ tf.reshape(aGrad,[-1,1]) for aGrad in grad_theta]
        grad_thetas = tf.concat(grad_theta_,axis=0)
        return loss, grad_thetas
   

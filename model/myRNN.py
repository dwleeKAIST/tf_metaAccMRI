import tensorflow as tf
import tensorflow.contrib.layers as li
from ipdb import set_trace as st
from tensorflow.contrib import rnn#
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import array_ops, math_ops
from util.util import tf_kri2imgri
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
    #z =  array_ops.concat([z1,z2],axis=0)
    #z = array_ops.where( math_ops.is_nan(z), tf.zeros_like(z),z)
    return z1,z2
 

class myDoubleLSTM:
    def __init__(self, dummy_theta_shapes, Learner, nHidden=100, ntheta=2112, kloss=False):
        self.nInput1  =   4
        self.nHidden1 = nHidden
        self.Learner  = Learner
        self.kloss    = kloss
        #
        self.droprate = 0.5
        self.nInput2  = nHidden 
        self.nHidden2 = ntheta 
        self.name     = 'myDoubleLSTM'
        self.nBtheta  = ntheta
        self.moment   = 0.9
        with tf.variable_scope(self.name):
            self.W1 = tf.Variable(tf.truncated_normal([self.nInput1+self.nHidden1,4*self.nHidden1],-0.01,0.01),name='W1')
            self.b1 = tf.Variable(tf.zeros([1,4*self.nHidden1]),name='b1')
            #
            self.WF2 = tf.Variable(tf.truncated_normal([self.nInput2+2, 1],-0.01,0.01),name='WF2')
            self.WI2 = tf.Variable(tf.truncated_normal([self.nInput2+2, 1],-0.01,0.01),name='WI2')
            self.bF2 = tf.Variable(5.,name='bF2')
            self.bI2 = tf.Variable(-5.,name='bI2')


        self.THETA = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        ## Learner parameter
        self.nCh_out = 2# 16
        self.info_theta = dummy_theta_shapes

    def LSTM1_f(self,x,c_prev,h_prev=[]):
        if h_prev==[]:
            h_prev = tf.zeros([self.nBtheta,self.nHidden1])
        # x : 2112 x steps x 4
        # h : 2112 x 100
        # W1: (4+100) x 4*100
        # steps = x.shape[1] #0:nBatch 1:steps

        #for istep in range(steps):
        if True:
            dots  = tf.matmul( tf.concat([x[:,0,:],h_prev],axis=1), self.W1 ) + self.b1
            dots_ = tf.reshape( dots, [self.nBtheta, 4, self.nHidden1])
            I     = tf.sigmoid( dots_[:,0,:])
            F     = tf.sigmoid( dots_[:,1,:])
            O     = tf.sigmoid( dots_[:,2,:])
            c_hat = tf.tanh( dots_[:,3,:] )
            c_cur = F*c_prev + I*c_hat
            h_cur = O * tf.tanh(c_cur)
            # for step iteration
            #c_prev = c_cur
            #tf.assign(c_prev, c_cur)
            #h_prev =  h_cur
        return c_cur, h_cur

    def LSTM2_f(self, x2_h, x2_grad, c_prev, f_prev, i_prev, delta_prev):
        ''' f_prev and i_prev = h_prev, later they are concatenated'''
        ## no steps now.
        # x2_h: 2112 x steps x 100
        # x2_grad: 2112 x steps x 1
        # WF2: (100+2) x 1
        # bF2: 1 x 1
        # c_prev : 2112 x 1
        # f_prev : 2112 x 1
        # i_prev : 2112 x 1
        # delta_prev : 2112 x 1

        if delta_prev==[]:
            delta_prev = tf.zeros([self.nBtheta,1])
        steps = 1#x2_h.shape[1]
        #for istep in range(steps):
        if True:
            #f_cur_ = tf.matmul(  tf.concat([ x2_h, c_prev, f_prev], axis=1), self.WF2 ) + self.bF2
            #i_cur_ = tf.matmul(  tf.concat([ x2_h, c_prev, i_prev], axis=1), self.WI2 ) + self.bI2
            f_cur_ = tf.matmul(  tf.concat([ x2_h, c_prev, f_prev], axis=1), self.WF2 ) + self.bF2
            i_cur_ = tf.matmul(  tf.concat([ x2_h, c_prev, i_prev], axis=1), self.WI2 ) + self.bI2


            i_cur     = tf.sigmoid(i_cur_)
            f_cur     = tf.sigmoid(f_cur_)
            #
            delta_cur = self.moment*delta_prev - i_cur*x2_grad
            c_cur = f_cur * c_prev + delta_cur
        return c_cur, f_cur, i_cur, delta_cur

    def f(self, k_shot, x, y, c1_prev, c2_prev, f2_prev,i2_prev,is_Training): 
        
        h1_prev = []
        delta_prev = []
        losses = [] 
        for ik in range(k_shot):
            # batch for here
            # for iB in nB:
            # later check
            loss, grad_thetas = self.learner_df(x,y,c2_prev) # loss : 1x1, grad_thetas : 2112 x 1 x1        
            losses.append(loss)
            pre_loss          = tf.stack( preProc(loss))
            #pre_loss          = tf.transpose(tmp_loss[:,tf.newaxis,tf.newaxis],[1,0])
            preLoss_          = tf.tile(pre_loss[tf.newaxis,tf.newaxis,:], [grad_thetas.shape[0],1,1])  #    2112 x 1 x 2
            preGrad           = tf.stack(preProc(grad_thetas),axis=2)        # 2112 x 1 x 2
            x1_inp            = tf.concat([preLoss_,preGrad],axis=2) # 2112 x1 x4
            #
            c1_cur, h1_cur    = self.LSTM1_f(x1_inp,c1_prev,h_prev=h1_prev)
            c2_cur, f2_cur, i2_cur,delta_cur = self.LSTM2_f(h1_cur,grad_thetas, c2_prev, f2_prev, i2_prev, delta_prev)
            # for next step iteration
            c1_prev= c1_cur
            c2_prev= c2_cur
            f2_prev= f2_cur
            i2_prev= i2_cur
            h1_prev= h1_cur
            #tf.assign(c1_prev, c1_cur)
            #tf.assign(c2_prev, c2_cur)
            #tf.assign(f2_prev, f2_cur)
            #tf.assign(i2_prev, i2_cur)
            delta_prev =  delta_cur
            #
        loss = tf.add_n(losses)
        return c1_cur, c2_cur, f2_cur, i2_cur, loss, grad_thetas
             
    def learner_f(self, x, y,theta):
        
        net_out, theta_ = self.Learner(x, self.nCh_out, weights=theta, info_theta=self.info_theta)
        if self.kloss:
            loss = tf.losses.mean_squared_error(labels=y, predictions=net_out)
        else:
            loss = tf.losses.mean_squared_error(labels=tf_kri2imgri(y),predictions=tf_kri2imgri(net_out))
        return net_out, theta_, loss

    def learner_df(self, x, y, theta):
        net_out, theta_, loss = self.learner_f(x,y,theta)
        grad_theta      = tf.gradients(loss,theta_)
        grad_thetas     = [[[]]]
        grad_theta_ = [ tf.reshape(aGrad,[-1,1]) for aGrad in grad_theta]
        grad_thetas = tf.concat(grad_theta_,axis=0)
        return loss, grad_thetas

 
#class myMultiRNNCell(rnn.MultiRNNCell):
#    def __init__(self, cells, state_is_tuple=True):
#        super(myMultiRNNCell, self).__init__(cells)
#        #self._cells
#
#    def call(self, inputs, state):
#        '''
#        inputs : 1st : processed Loss, processed Grad
#                 2nd : h from 1st, grad
#        '''
#        cur_state_pos = 0
#        cur_inp = inputs # should be devided.
#        #first_inp    = preProc(inputs)
#        first_inp    = inputs
#        second_inp   = array_ops.slice(inputs,[0,0],[-1,int(inputs.shape[1])-1])# for 2nd, w/o loss
#        new_states = []
#
#        # for i, cell in enumerate(self._cells):
#        ## first LSTM goes here
#        with vs.variable_scope("cell_%d" % 0):
#            if self._state_is_tuple:
#                cur_state = state[0]
#            else:
#                cur_state = array_ops.slice(state,[0,cur_state_pos],[-1,self._cells[0].state_size])
#                cur_state_pos+= self._cells[0].state_size
#            ## apply cell Here
#            first_out, new_state = self._cells[0](first_inp, cur_state)
#            new_states.append(new_state)
#
#        ## second LSTM goes here
#        cur_inp = array_ops.concat([first_out, second_inp],1)
#        with vs.variable_scope("cell_%d" % 1):
#            if self._state_is_tuple:
#                cur_state = state[1]
#            else:
#                cur_state = array_ops.slice(state,[0,cur_state_pos],[-1,self._cells[1].state_size])
#                cur_state_pos+= self._cells[1].state_size
#            ## apply cell Here
#            cur_inp, new_state = self._cells[1](cur_inp, cur_state)
#            new_states.append(new_state)
#
#        ##
#        new_states = (tuple(new_states) if self._state_is_tuple else array_ops.concat(new_states,1))
#        return cur_inp, new_states       

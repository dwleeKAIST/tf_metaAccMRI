import tensorflow as tf
import numpy as np
from ipdb import set_trace as st
import matplotlib.pyplot as plt

ch_dim=1
dtype = tf.float32

def tf_pad0(k_ri, padX,name_=""):
    return  tf.pad(k_ri, [[0,0,],[0,0,],[0,0,],[padX,padX]], 'CONSTANT')

def tf_imgri2kri(img_ri, name_=""):
    hnOut  = int(int(img_ri.shape[ch_dim])/2)
    i_real = tf.slice(img_ri, [0,    0,0,0], [-1,hnOut,-1,-1])#, name=name_.join("_ri2real_i"))
    i_imag = tf.slice(img_ri, [0,hnOut,0,0], [-1,hnOut,-1,-1])#, name=name_.join("_ri2imag_i"))
    i_comp = tf.complex(i_real, i_imag, name="ri2complex_i")
    k_comp = myTFfftshift2(tf.ifft2d(i_comp, name='img2k')) # temp edit by DW for scale issue
    return   tf.concat([tf.cast(tf.real(k_comp),dtype), tf.cast(tf.imag(k_comp),dtype)],axis=1)

def tf_kri2imgri(k_ri, name_=""):
    hnOut  = int(int(k_ri.shape[ch_dim])/2)
    k_real = tf.slice(k_ri, [0,    0,0,0], [-1,hnOut,-1,-1])#, name=name_.join("_ri2real_k"))
    k_imag = tf.slice(k_ri, [0,hnOut,0,0], [-1,hnOut,-1,-1])#, name=name_.join("_ri2imag_k"))
    k_comp = tf.complex(k_real, k_imag, name="ri2complex_k")
    i_comp = tf.fft2d(myTFfftshift2(k_comp), name='k2img') # temporally edited by DW for scale issue
    return   tf. concat([tf.cast(tf.real(i_comp),dtype), tf.cast(tf.imag(i_comp),dtype)],axis=1)

class DIM2CH2():
    def __init__(self, shape4D=[1,16,588,288], inp_1st=True):
        self.inp_1st = inp_1st
        self.sz = shape4D
        self.R = 2
        self.R_cat = []
        self.nQ = int(self.sz[3]/self.R)

        for x in range(self.nQ):
            for iR in range(self.R):
                self.R_cat.append(x+iR*self.nQ)
    
    def D2CH(self, k_space ):
        output  = tf.strided_slice(k_space,[0,0,0,0], self.sz, strides=[1,1,1,self.R])
        input   = tf.strided_slice(k_space,[0,0,0,1], self.sz, strides=[1,1,1,self.R])
        return input, output

    def CH2D(self, k_inp, k_rec):
        k_space = tf.concat([k_rec,k_inp], axis=3)
        return tf.gather(k_space,self.R_cat,axis=3) 

    ## util for LapNet
    def D2CH_(self, k_space):
        if self.inp_1st:
            k1  = tf.strided_slice(k_space,[0,0,0,0], self.sz, strides=[1,1,1,self.R])
            k2   = tf.strided_slice(k_space,[0,0,0,1], self.sz, strides=[1,1,1,self.R])
        else:
            st()
        return k1, k2

    def CH2D_(self, k1, k2):
        if self.inp_1st:
            k_space = tf.concat([k1,k2], axis=3)
        else:
            k_space = tf.concat([k2,k1],axis=3)
        return tf.gather(k_space,self.R_cat,axis=3) 



class DIM2CH4():
    def __init__(self, shape4D=[1,16,588,288]):
        self.sz = shape4D
        self.R  = 4 
        self.R_cat = []
        self.nQ = int(self.sz[3]/self.R)

        for x in range(self.nQ):
            for iR in range(self.R):
                self.R_cat.append(x+iR*self.nQ)
   
    def D2CH(self, k_space, sz=[] ):
        if sz==[]:
            sz = self.sz
        out1   = tf.strided_slice(k_space,[0,0,0,0], sz, strides=[1,1,1,4])
        input  = tf.strided_slice(k_space,[0,0,0,1], sz, strides=[1,1,1,4])
        out2   = tf.strided_slice(k_space,[0,0,0,2], sz, strides=[1,1,1,4])
        out3   = tf.strided_slice(k_space,[0,0,0,3], sz, strides=[1,1,1,4])
        output = tf.concat([out1,out2,out3],axis=ch_dim)
        return input, output

    def CH2D(self, k_inp, k_rec ):   
        nCh= int(self.sz[1] )
        k1 = tf.slice(k_rec, [0,    0,0,0], [-1,nCh,-1,self.nQ])
        k3 = tf.slice(k_rec, [0,  nCh,0,0], [-1,nCh,-1,self.nQ])
        k4 = tf.slice(k_rec, [0,nCh*2,0,0], [-1,nCh,-1,self.nQ])
        k_space = tf.concat([k1,k_inp,k3,k4],axis=3)
        return tf.gather(k_space, self.R_cat, axis=3)

    ## util for LapNet
    def D2CH_(self, k_space, sz=[] ):
        if sz==[]:
            sz = self.sz
        out1   = tf.strided_slice(k_space,[0,0,0,0], sz, strides=[1,1,1,4])
        out2  = tf.strided_slice(k_space,[0,0,0,1], sz, strides=[1,1,1,4])
        out3   = tf.strided_slice(k_space,[0,0,0,2], sz, strides=[1,1,1,4])
        out4   = tf.strided_slice(k_space,[0,0,0,3], sz, strides=[1,1,1,4])
        return out1,out2,out3,out4
   
    def CH2D_(self, k1,k2,k3,k4):   
        nCh= int(self.sz[1] )
        k_space = tf.concat([k1,k2,k3,k4],axis=3)
        return tf.gather(k_space, self.R_cat, axis=3)
def slice2ker__b( full_tensor, start_idx, szs):
    len2slice = szs[0]
    cur_idx   = start_idx+int(len2slice)
    return cur_idx, tf.reshape(tf.slice(full_tensor, [start_idx,0], [len2slice,-1]),szs)


def slice2ker__( full_tensor, start_idx, szs):
    len2slice = szs[0]*szs[1]*szs[2]*szs[3]
    cur_idx   = start_idx+int(len2slice)
    return cur_idx, tf.reshape(tf.slice(full_tensor, [start_idx,0], [len2slice,-1]),szs)


def slice2ker_( full_tensor, start_idx, szs):
    len2slice = szs[0]*szs[1]*szs[2]*szs[3]
    cur_idx   = start_idx+int(len2slice)
    return cur_idx, tf.reshape(tf.slice(full_tensor, [0,start_idx], [-1,len2slice]),szs)


def slice2ker( full_tensor, start_idx, kH=3, kW=3, chin=2, chout=2, name_=''):
    len2slice = kH*kW*chin*chout
    cur_idx   = start_idx+len2slice
    return cur_idx, tf.reshape(tf.slice(full_tensor, [0,start_idx], [-1,len2slice], name=name_),[kH,kW,chin,chout])
    #return cur_idx, tf.reshape(tf.slice(full_tensor, [0,start_idx], [-1,len2slice], name="slice_".join(name_)),[3,3,kerOut_chin,kerOut_chout])
    #Batch num should be 0

def tf_ri2sum(img_ri):
    hnout  = int(int(img_ri.shape[ch_dim])/2)
    real   = tf.slice(img_ri,[0,0,0,0],[-1,hnout,-1,-1])
    imag   = tf.slice(img_ri,[0,hnout,0,0],[-1,hnout,-1,-1])
    #i_ssos = tf.sqrt( tf.reduce_sum(tf.square(real) + tf.square(imag),axis=1, keep_dims=True) )
    real_ =  tf.reduce_sum(real,axis=1, keep_dims=True) 
    imag_ =  tf.reduce_sum(imag,axis=1, keep_dims=True)
    i_sum =  tf.concat([real_,imag_],axis=1)

    return i_sum 

def tf_imgri2ssos(img_ri):
    hnout  = int(int(img_ri.shape[ch_dim])/2)
    real   = tf.slice(img_ri,[0,0,0,0],[-1,hnout,-1,-1])
    imag   = tf.slice(img_ri,[0,hnout,0,0],[-1,hnout,-1,-1])
    i_ssos = tf.sqrt( tf.reduce_sum(tf.square(real) + tf.square(imag),axis=1, keep_dims=True) )
    i_ssos = tf.transpose(i_ssos, [0,2,3,1])
    return i_ssos

def tf_ri2ssos(img_ri, io_kspace=False):
    hnout  = int(int(img_ri.shape[ch_dim])/2)
    real   = tf.slice(img_ri,[0,0,0,0],[-1,hnout,-1,-1])
    imag   = tf.slice(img_ri,[0,hnout,0,0],[-1,hnout,-1,-1])

    if io_kspace:
        k_ssos = tf.sqrt( tf.reduce_sum(tf.square(real) + tf.square(imag),axis=1, keep_dims=True) )
        k_ssos = tf.transpose(k_ssos, [0,2,3,1])
        img = tf.fft2d(myTFfftshift2(tf.complex(real,imag))) #temp edit
        i_ssos = tf.sqrt( tf.reduce_sum(tf.abs(img), axis=1, keep_dims=True) )
        i_ssos = tf.transpose(i_ssos,[0,2,3,1])
        return i_ssos, k_ssos
    else:
        i_ssos = tf.sqrt( tf.reduce_sum(tf.square(real) + tf.square(imag),axis=1, keep_dims=True) )
        i_ssos = tf.transpose(i_ssos, [0,2,3,1])
        k = myTFfftshift2(tf.ifft2d(tf.complex(real,imag))) #temp_edit
        k_ssos = tf.sqrt( tf.reduce_sum(tf.abs(k),axis=1, keep_dims=True) )
        k_ssos = tf.transpose(k_ssos, [0,2,3,1])
        return i_ssos, k_ssos

def myNumExt(s):
    head = s.rstrip('0123456789')
    tail = s[len(head):]
    return int(tail)

def ri2ssos(inp):
    st()
    sz   = inp.shape
    nCh  = int(int(sz[3])/2)
    if nCh == 1:
        out  = tf.sqrt(tf.square(inp[:,:,:,0:nCh])+tf.square(inp[:,:,:,nCh:]))
        return out
    else:
        st()

class MyWeight():
    def __init__(self, nY,nX,nCh,wtype,scale=1):
        self.hny = int(nY/2)
        self.hnx = int(nX/2)
        ax_x, ax_y = np.mgrid[-self.hny:self.hny,-self.hnx:self.hnx]
        z_max    = np.sqrt( np.square(self.hny)+np.square(self.hnx))
        ax_x = ax_x/z_max
        ax_y = ax_y/z_max
        z          = np.sqrt( np.square(ax_x*scale) + np.square(ax_y*scale) ) * np.pi
        
        if wtype == '1white':
            w = np.abs(z)
        elif wtype == '2white':
            w = np.square(np.abs(z))
        elif wtype == '1counter':
            w = np.abs(1-np.exp(-1j*z))
        elif wtype == '2counter':
            w = np.abs( np.square( 1-np.exp(-1j*z) ) )
        else:
            st()
        eps = 0.00001
        w[self.hny, self.hnx]  = eps
        uw = np.divide(1,w)
        w[self.hny,self.hnx]   = 0
        uw[self.hny, self.hnx] = 0

        DC_r = 1 
        self.DC_mask  = np.zeros([nCh, nY, nX], dtype=float)
        self.DC_mask[:,self.hny-DC_r:self.hny+DC_r, self.hnx-DC_r:self.hnx+DC_r] = 1.0 
        self.DC_maskT = np.ones([nCh, nY,nX],dtype=float)-self.DC_mask
        #
        if nCh==1:
            self.weight = np.expand_dims(w,0).astype(np.float64)
            self.unweight = np.expand_dims(uw,0).astype(np.float64)
        else:
            self.weight   = np.tile( np.expand_dims(w,0), (nCh,1,1))
            self.unweight = np.tile( np.expand_dims(uw,0), (nCh,1,1))

    def get_w(self):
        return self.weight

    def get_uw(self):
        return self.unweight

    def get_DC_mask(self):
        return self.DC_mask
    def get_DC_maskT(self):
        return self.DC_maskT


def play(img, XY_dims=[2,3]):
    if len(img.shape)==2:
        img_z = np.abs(img)
    else:
        if XY_dims==[2,3]:
            img_z = img[0,0,:,:]
        else:
            st()
    plt.imshow(img_z)
    plt.show()


def myTFfftshift2(inp, axes=(1,2)):
    if axes==(1,2):
        hnY = int(int(inp.shape[2])/2)
        hnX = int(int(inp.shape[3])/2)
        out = tf.concat( [tf.slice(inp, [0,0,hnY,0],[-1,-1,hnY,-1] ), tf.slice(inp,[0,0,0,0], [-1,-1,hnY,-1])], axis=2)
        out = tf.concat( [tf.slice(out, [0,0,0,hnX],[-1,-1,-1,hnX] ), tf.slice(out,[0,0,0,0], [-1,-1,-1,hnX])], axis=3)
    else:
        st()
    return out

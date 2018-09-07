from os import listdir
from os.path import join, isfile
import random
from scipy import io as sio
import numpy as np
import copy
from ipdb import set_trace as st

class DB7T():
    def __init__(self,opt,phase,dataroot='./../../data/MRI/T1w_pad_8ch_x2_halfnY',DSrate=2):
        super(DB7T, self).__init__()
        random.seed(0)
        self.dataroot = dataroot
        self.root   = join(self.dataroot,phase)
        self.flist  = []

        list_fname = join(self.dataroot, 'DBset_'+phase+'.npy')
        if isfile(list_fname):
            print(list_fname + ' exists. Now loading...')
            self.flist = np.load(list_fname)
        else:
            print('Now generating.... '+list_fname)
            flist=[]
            for aDir in sorted(listdir(self.root)):
                for aImg in sorted(listdir(join(self.root,aDir))):
                    flist.append(join(aDir,aImg))
            np.save(list_fname,flist)
            self.flist = flist
        if opt.smallDB: 
            self.flist = self.flist[0:int(len(self.flist)/10)]
 
        self.nCh  = 8*2
        self.nY   = 288#576
        self.nX   = 288
        self.len  = len(self.flist) 
        self.nACS = 32#28
        self.DSrate = DSrate
        self.dsnX   = int(self.nX/self.DSrate)
        self.len  = len(self.flist) 
        self.dsnACS = int(self.nACS/self.DSrate)

        self.ACS_s= int((self.nX-self.nACS)/2)
        self.ACS_e= self.ACS_s+self.nACS
        self.mask = np.zeros([self.nCh, self.nY, self.nX])

        self.nACSACS=16
        self.ACSACS_s=int((self.nX-self.nACSACS)/2)
        self.ACSACS_e=self.ACSACS_s+self.nACSACS

        if self.DSrate==2:
            self.bias=0
        elif self.DSrate==4:
            self.bias=0
        else:
            st()
        vec = [i*self.DSrate+self.bias for i in range(int(self.nX/self.DSrate))]
        self.mask[:,:,vec]=1
        self.mask2 = copy.deepcopy(self.mask)
        self.mask[:,:,self.ACS_s:self.ACS_e]=1
        self.mask2[:,:,self.ACSACS_s:self.ACSACS_e]=1

    def getInfo(self,opt):
        opt.nCh_in = self.nCh
        opt.nCh_out = self.nCh
        opt.nY   = self.nY
        opt.nX   = self.nX
        opt.nACS = self.nACS
        opt.savepath  = './../../data/MRI/result'
        sdir = opt.savepath+'/'+opt.name+'/'
        opt.log_dir   = sdir+'log_dir/train'
        opt.log_dir_v = sdir+'log_dir/valid'
        opt.ckpt_dir  = sdir+'ckpt_dir'
        opt.hPad = int((self.nX-self.nACS)  /2)
        return opt
    
    def getBatch(self, start, end):
        end   = min([end,self.len])
        batch = self.flist[start:end]

        sz_       = [end-start, self.nCh, self.nY, self.nX]
        sz_ACS    = [end-start, self.nCh, self.nY, self.nACS]
        target_ACSk_ri  = np.empty(sz_ACS, dtype=np.float32)
        input_ACSk_ri   = np.empty(sz_ACS, dtype=np.float32)
        target_k_ri     = np.empty(sz_, dtype=np.float32)
        input_k_ri      = np.empty(sz_, dtype=np.float32)

        for iB, aBatch in enumerate(batch):
            aTarget_k = self.read_mat(join(self.root,aBatch),'orig_k')
            DS_k      = aTarget_k*self.mask
            DS_ACS_k  = aTarget_k*self.mask2
            input_ACSk_ri[iB,:,:,:]  =  DS_ACS_k[:,:,self.ACS_s:self.ACS_e]
            target_ACSk_ri[iB,:,:,:] = aTarget_k[:,:,self.ACS_s:self.ACS_e]
            input_k_ri[iB,:,:,:]  = DS_k
            target_k_ri[iB,:,:,:] = aTarget_k
        return input_ACSk_ri, target_ACSk_ri, input_k_ri, target_k_ri    
    
    def getBatch_G_train(self, start, end):
        end   = min([end,self.len])
        batch = self.flist[start:end]
        #if np.random.randint(0,high=1):
        batch2= self.flist[ np.random.randint(0,high=self.len,size=end-start) ]
        #else:
        #    batch2 = batch

        sz_       = [end-start, self.nCh, self.nY, self.dsnX]
        sz_ACS    = [end-start, self.nCh, self.nY, self.dsnACS]
        sz_out    = [end-start, self.nCh*(self.DSrate-1), self.nY, self.dsnX]
        sz_ACS_out= [end-start, self.nCh*(self.DSrate-1), self.nY, self.dsnACS]
        target_ACSk_ri  = np.empty(sz_ACS_out, dtype=np.float32)
        input_ACSk_ri   = np.empty(sz_ACS, dtype=np.float32)
        target_k_ri     = np.empty(sz_out, dtype=np.float32)
        input_k_ri      = np.empty(sz_, dtype=np.float32)

        for iB, aBatch in enumerate(batch):
            aTarget_k   = self.read_mat(join(self.root,aBatch),'orig_k')
            bTarget_k   = self.read_mat(join(self.root,batch2[iB]),'orig_k')
            aACS_k      = aTarget_k[:,:,self.ACS_s:self.ACS_e]
            input_ACSk_ri[iB,:,:,:]  = aACS_k[:,:,self.bias::self.DSrate]
            target_ACSk_ri[iB,:,:,:] = aACS_k[:,:,self.bias+1::self.DSrate]
            input_k_ri[iB,:,:,:]  = bTarget_k[:,:,self.bias::self.DSrate]
            target_k_ri[iB,:,:,:] = bTarget_k[:,:,self.bias+1::self.DSrate]
        
        return input_ACSk_ri, target_ACSk_ri, input_k_ri, target_k_ri    

    def getBatch_G(self, start, end):
        end   = min([end,self.len])
        batch = self.flist[start:end]
         
        sz_       = [end-start, self.nCh, self.nY, self.dsnX]
        sz_ACS    = [end-start, self.nCh, self.nY, self.dsnACS]
        sz_out    = [end-start, self.nCh*(self.DSrate-1), self.nY, self.dsnX]
        sz_ACS_out= [end-start, self.nCh*(self.DSrate-1), self.nY, self.dsnACS]
        target_ACSk_ri  = np.empty(sz_ACS_out, dtype=np.float32)
        input_ACSk_ri   = np.empty(sz_ACS, dtype=np.float32)
        target_k_ri     = np.empty(sz_out, dtype=np.float32)
        input_k_ri      = np.empty(sz_, dtype=np.float32)

        for iB, aBatch in enumerate(batch):
            aTarget_k   = self.read_mat(join(self.root,aBatch),'orig_k')
            aACS_k      = aTarget_k[:,:,self.ACS_s:self.ACS_e]
            input_ACSk_ri[iB,:,:,:]  = aACS_k[:,:,self.bias::self.DSrate]
            target_ACSk_ri[iB,:,:,:] = aACS_k[:,:,self.bias+1::self.DSrate]
            input_k_ri[iB,:,:,:]  = aTarget_k[:,:,self.bias::self.DSrate]
            target_k_ri[iB,:,:,:] = aTarget_k[:,:,self.bias+1::self.DSrate]
        
        return input_ACSk_ri, target_ACSk_ri, input_k_ri, target_k_ri    

    def getBatch_G_(self, start, end):
        end   = min([end,self.len])
        batch = self.flist[start:end]

        sz_       = [end-start, self.nCh, self.nY, self.dsnX, self.nCh]
        sz_ACS    = [end-start, self.nCh, self.nY, self.dsnACS, self.nCh]
        sz_out    = [end-start,  self.nY, self.dsnX, self.nCh*(self.DSrate-1)]
        sz_ACS_out= [end-start,  self.nY, self.dsnACS, self.nCh*(self.DSrate-1)]
        target_ACSk_ri  = np.empty(sz_ACS_out, dtype=np.float32)
        input_ACSk_ri   = np.empty(sz_ACS, dtype=np.float32)
        target_k_ri     = np.empty(sz_out, dtype=np.float32)
        input_k_ri      = np.empty(sz_, dtype=np.float32)

        for iB, aBatch in enumerate(batch):
            aTarget_k   = self.read_mat(join(self.root,aBatch),'orig_k')
            aACS_k      = aTarget_k[:,:,self.ACS_s:self.ACS_e]
            input_ACSk_ri[iB,:,:,:]  = np.moveaxis(aACS_k[:,:,self.bias::self.DSrate],0,-1)
            target_ACSk_ri[iB,:,:,:] = np.moveaxis(aACS_k[:,:,self.bias+1::self.DSrate],0,-1)
            input_k_ri[iB,:,:,:]  = np.moveaxis(aTarget_k[:,:,self.bias::self.DSrate],0,-1)
            target_k_ri[iB,:,:,:] = np.moveaxis(aTarget_k[:,:,self.bias+1::self.DSrate],0,-1)
        return input_ACSk_ri, target_ACSk_ri, input_k_ri, target_k_ri    


    def getBatch_meta_bu(self, start, end):
        end   = min([end,self.len])
        batch = self.flist[start:end]

        sz_   = [end-start, self.nCh*2, self.nY, self.hX]
        sz_ACS= [end-start, self.nCh*2, self.nY, self.hACS]
        
        target_ACSk_ri = np.empty(sz_ACS, dtype=np.float32)
        input_ACSk_ri  = np.empty(sz_ACS, dtype=np.float32)
        target_k_ri = np.empty(sz_, dtype=np.float32)
        input_k_ri  = np.empty(sz_, dtype=np.float32)

        for iB, aBatch in enumerate(batch):
            #aTarget = np.moveaxis(self.read_mat(join(self.root,aBatch)),2,0)
            ACS_inp, ACS_out, full_inp, full_out = self.read_mat_meta(join(self.root,aBatch))
            input_ACSk_ri[iB,:,:,:]  = ACS_inp#[self.sel_ch_ri]
            target_ACSk_ri[iB,:,:,:] = ACS_out#[self.sel_ch_ri]
            input_k_ri[iB,:,:,:]  = full_inp#[self.sel_ch_ri]
            target_k_ri[iB,:,:,:] = full_out#[self.sel_ch_ri]
        return input_ACSk_ri, target_ACSk_ri, input_k_ri, target_k_ri    



    def getBatch_k(self, start, end):
        end   = min([end,self.len])
        batch = self.flist[start:end]
        sz    = [end-start, self.nCh  , self.nY, self.nX] 
        sz_ri = [end-start, self.nCh*2, self.nY, self.nX]
        
        target_k_ri = np.empty(sz_ri, dtype=np.float32)
        input_k_ri  = np.empty(sz_ri, dtype=np.float32)

        for iB, aBatch in enumerate(batch):
            aTarget = np.moveaxis(self.read_mat(join(self.root,aBatch)),2,0)
            
            if self.use_flip:
                R  = np.random.rand(2)
                #if R[0]>0.5:
                #    orig_img= np.flip(orig_img,1)
                if R[1]>0.5:
                    orig_img= np.flip(orig_img,2)
            orig_k      = np.fft.fftshift( np.fft.fft2(aTarget, axes=(1,2)), axes=(1,2))
            down_k      = np.multiply(orig_k, self.mask)
            if self.normalize:
                scale_ = 1.0/np.amax(np.abs(down_k))
                orig_k = orig_k*scale_
                down_k = down_k*scale_
            #down_img    = np.fft.ifft2(np.fft.ifftshift(down_k,axes=(1,2)),axes=(1,2))
            target_k_ri[iB,:,:,:] = np.concatenate( (np.real(orig_k), np.imag(orig_k)), axis=0)
            input_k_ri[iB,:,:,:] = np.concatenate( (np.real(down_k), np.imag(down_k)), axis=0) 
        return input_k_ri, target_k_ri   

    def shuffle(self, seed=0):
        random.seed(seed)
        random.shuffle(self.flist)

    def name(self):
        return '7T 8ch dataset'


    def __len__(self):
        return self.len
    
    @staticmethod
    def read_mat(filename, var_name="img"):
        mat = sio.loadmat(filename)
        return mat[var_name]

    @staticmethod
    def read_mat_meta(filename):
        mat = sio.loadmat(filename)
        return mat['ACS_inp'], mat['ACS_out'], mat['orig_inp'], mat['orig_out']
if __name__ == "__main__":
    tmp = DB7T_8ch('../../data/MRI')

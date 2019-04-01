from os import listdir
from os.path import join, isfile
import random
from scipy import io as sio
import numpy as np
import copy
from util.MyWeight import MyWeight
from ipdb import set_trace as st

DB_path1 = './../../mrdata/T1w_pad_8ch_x2_halfnY'

class DB7T():
    def __init__(self,opt,phase):#,dataroot='./../../mrdata/T1w_pad_8ch_x2_halfnY'):
        super(DB7T, self).__init__()
        random.seed(0)
        self.dataroot = opt.dataroot
        self.root   = join(self.dataroot,phase)
        self.flist  = []

        self.sel_8ch = [11,12,15,17,19,22,25,28,
                43,44,47,49,51,54,57,60]

        self.norm_ch = False#True
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
        self.nC   = 8 if self.dataroot == DB_path1 else 32
        self.nCh  = 8*2 if (self.dataroot == DB_path1 or opt.model=='Gnet_DS4_8ch' or opt.model=='Gnet2_DS4_8ch') else 64 # nCh*Real/imag
        self.nY   = 288#576
        self.nX   = 288
        self.len  = len(self.flist) 
        self.DSrate = opt.DSrate
        self.nACS = 32 if self.DSrate==2 or self.DSrate==4 else 36
        self.dsnX   = int(self.nX/self.DSrate)
        self.len  = len(self.flist) 
        self.dsnACS = int(self.nACS/self.DSrate)

        self.ACS_s= int((self.nX-self.nACS)/2)
        self.ACS_e= self.ACS_s+self.nACS
        self.mask_woACS = np.zeros([self.nCh, self.nY, self.nX])

        #self.nACSACS=16
        #self.ACSACS_s=int((self.nX-self.nACSACS)/2)
        #self.ACSACS_e=self.ACSACS_s+self.nACSACS
        self.Aug = opt.Aug and phase=='train'
        self.bias=0
        ##
        vec = [i*self.DSrate+self.bias for i in range(int(self.nX/self.DSrate))]
        self.mask_woACS[:,:,vec]=1
        self.mask = copy.deepcopy(self.mask_woACS)
        #self.mask2 = copy.deepcopy(self.mask)
        self.mask[:,:,self.ACS_s:self.ACS_e]=1
        #self.mask2[:,:,self.ACSACS_s:self.ACSACS_e]=1
        
        self.ACS_mask =np.zeros([self.nCh,self.nY,self.nX])
        self.ACS_mask[:,:,self.ACS_s:self.ACS_e]=1

        myW = MyWeight(self.nY,self.nX,self.nCh,'1white')
        self.w = myW.get_w()
        self.uw= myW.get_uw()

    def getInfo(self,opt):
        opt.nCh_in = self.nCh
        opt.nCh_out = self.nCh*(self.DSrate-1)
        opt.nY   = self.nY
        opt.nX   = self.nX
        opt.dsnX = self.dsnX
        opt.nACS = self.nACS
        opt.dsnACS=self.dsnACS
        opt.savepath  = './../../data/MRI/result'
        sdir = opt.savepath+'/'+opt.name+'/'
        opt.log_dir   = sdir+'log_dir/train'
        opt.log_dir_v = sdir+'log_dir/valid'
        opt.ckpt_dir  = sdir+'ckpt_dir'
        opt.hPad = int((self.nX-self.nACS)  /2)
        opt.ACS_mask = self.ACS_mask
        return opt
    
#    def getInfo_RAKI(self,opt):
#        opt.nCh_in = int(self.nCh/2)
#        opt.nCh_out = 2*(self.DSrate-1)
#        opt.nY   = self.nY
#        opt.nX   = self.nX
#        opt.dsnX = self.dsnX
#        opt.nACS = self.nACS
#        opt.dsnACS=self.dsnACS
#        opt.savepath  = './../../data/MRI/result'
#        sdir = opt.savepath+'/'+opt.name+'/'
#        opt.log_dir   = sdir+'log_dir/train'
#        opt.log_dir_v = sdir+'log_dir/valid'
#        opt.ckpt_dir  = sdir+'ckpt_dir'
#        opt.hPad = int((self.nX-self.nACS)  /2)
#        opt.ACS_mask = self.ACS_mask
#        return opt
 
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
    
    def getBatch_RAKI(self, start, end):
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
            DS_k_woACS = aTarget_k*self.mask_woACS
            DS_k_wACS = aTarget_k*self.mask
            input_ACSk_ri[iB,:,:,:]  =  DS_k_woACS[:,:,self.ACS_s:self.ACS_e]
            target_ACSk_ri[iB,:,:,:] = DS_k_wACS[:,:,self.ACS_s:self.ACS_e]
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
            target_k_ri[iB,:,:,:] = aTarget_k[:,:,self.bias+1::self.DSrate]
        
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
     
    def getBatch_G4_train(self, start, end):
        end   = min([end,self.len])
        batch = self.flist[start:end]
        batch2= self.flist[ np.random.randint(0,high=self.len,size=end-start) ]
         
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
            ACS1 = aACS_k[:,:,self.bias+1::self.DSrate]
            ACS2 = aACS_k[:,:,self.bias+2::self.DSrate]
            ACS3 = aACS_k[:,:,self.bias+3::self.DSrate]
            target_ACSk_ri[iB,:,:,:] = np.concatenate((ACS1,ACS2,ACS3),axis=0)
            
            input_k_ri[iB,:,:,:]  = bTarget_k[:,:,self.bias::self.DSrate]
            k1 = bTarget_k[:,:,self.bias+1::self.DSrate]
            k2 = bTarget_k[:,:,self.bias+2::self.DSrate]
            k3 = bTarget_k[:,:,self.bias+3::self.DSrate]
            target_k_ri[iB,:,:,:] = np.concatenate((k1,k2,k3),axis=0) 
        return input_ACSk_ri, target_ACSk_ri, input_k_ri, target_k_ri      
    def getBatch_G4(self, start, end):
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
            if self.Aug:
                aTarget_k = aTarget_k*np.random.normal(loc=1.0,scale=0.05)

            aACS_k      = aTarget_k[:,:,self.ACS_s:self.ACS_e]
            input_ACSk_ri[iB,:,:,:]  = aACS_k[:,:,self.bias::self.DSrate]
            ACS1 = aACS_k[:,:,self.bias+1::self.DSrate]
            ACS2 = aACS_k[:,:,self.bias+2::self.DSrate]
            ACS3 = aACS_k[:,:,self.bias+3::self.DSrate]
            
            target_ACSk_ri[iB,:,:,:] = np.concatenate((ACS1,ACS2,ACS3),axis=0)
            input_k_ri[iB,:,:,:]  = aTarget_k[:,:,self.bias::self.DSrate]
            k1 = aTarget_k[:,:,self.bias+1::self.DSrate]
            k2 = aTarget_k[:,:,self.bias+2::self.DSrate]
            k3 = aTarget_k[:,:,self.bias+3::self.DSrate]
            target_k_ri[iB,:,:,:] = np.concatenate((k1,k2,k3),axis=0) 
        return input_ACSk_ri, target_ACSk_ri, input_k_ri, target_k_ri    

    def getBatch_G4_sel8(self, start, end):
        end   = min([end,self.len])
        batch = self.flist[start:end]
         
        sz_       = [end-start, 16, self.nY, self.dsnX]
        sz_ACS    = [end-start, 16, self.nY, self.dsnACS]
        sz_out    = [end-start, 16*(self.DSrate-1), self.nY, self.dsnX]
        sz_ACS_out= [end-start, 16*(self.DSrate-1), self.nY, self.dsnACS]
        target_ACSk_ri  = np.empty(sz_ACS_out, dtype=np.float32)
        input_ACSk_ri   = np.empty(sz_ACS, dtype=np.float32)
        target_k_ri     = np.empty(sz_out, dtype=np.float32)
        input_k_ri      = np.empty(sz_, dtype=np.float32)

        for iB, aBatch in enumerate(batch):
            aTmp_k   = self.read_mat(join(self.root,aBatch),'orig_k')
            aTarget_k   = aTmp_k[self.sel_8ch,:,:]
            if self.Aug:
                aTarget_k = aTarget_k*np.random.normal(loc=1.0,scale=0.01)

            aACS_k      = aTarget_k[:,:,self.ACS_s:self.ACS_e]
            input_ACSk_ri[iB,:,:,:]  = aACS_k[:,:,self.bias::self.DSrate]
            ACS1 = aACS_k[:,:,self.bias+1::self.DSrate]
            ACS2 = aACS_k[:,:,self.bias+2::self.DSrate]
            ACS3 = aACS_k[:,:,self.bias+3::self.DSrate]
            
            target_ACSk_ri[iB,:,:,:] = np.concatenate((ACS1,ACS2,ACS3),axis=0)
            input_k_ri[iB,:,:,:]  = aTarget_k[:,:,self.bias::self.DSrate]
            k1 = aTarget_k[:,:,self.bias+1::self.DSrate]
            k2 = aTarget_k[:,:,self.bias+2::self.DSrate]
            k3 = aTarget_k[:,:,self.bias+3::self.DSrate]
            target_k_ri[iB,:,:,:] = np.concatenate((k1,k2,k3),axis=0) 
        return input_ACSk_ri, target_ACSk_ri, input_k_ri, target_k_ri    


    def getBatch_G4_std_sel8(self, start, end):
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
        scale_std = np.empty([end-start, self.nCh],dtype=np.float32)
        
        for iB, aBatch in enumerate(batch):
            aTmp_k,vec_std   = self.read_mat_std(join(self.root,aBatch),'orig_k')
            aTarget_k   = aTmp_k[self.sel_8ch,:,:]
            scale_std[iB,0:int(self.nCh/2)] = vec_std[0,self.sel_8ch[0:8]]
            scale_std[iB,int(self.nCh/2):] = vec_std[0,self.sel_8ch[0:8]]

            if self.Aug:
                aTarget_k = aTarget_k*np.random.normal(loc=1.0,scale=0.01)

            aACS_k      = aTarget_k[:,:,self.ACS_s:self.ACS_e]
            input_ACSk_ri[iB,:,:,:]  = aACS_k[:,:,self.bias::self.DSrate]
            ACS1 = aACS_k[:,:,self.bias+1::self.DSrate]
            ACS2 = aACS_k[:,:,self.bias+2::self.DSrate]
            ACS3 = aACS_k[:,:,self.bias+3::self.DSrate]
            
            target_ACSk_ri[iB,:,:,:] = np.concatenate((ACS1,ACS2,ACS3),axis=0)
            input_k_ri[iB,:,:,:]  = aTarget_k[:,:,self.bias::self.DSrate]
            k1 = aTarget_k[:,:,self.bias+1::self.DSrate]
            k2 = aTarget_k[:,:,self.bias+2::self.DSrate]
            k3 = aTarget_k[:,:,self.bias+3::self.DSrate]
            target_k_ri[iB,:,:,:] = np.concatenate((k1,k2,k3),axis=0) 
        return input_ACSk_ri, target_ACSk_ri, input_k_ri, target_k_ri, scale_std 


    def getBatch_G4_std(self, start, end):
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
        scale_std = np.empty([end-start, self.nCh],dtype=np.float32)
        
        for iB, aBatch in enumerate(batch):
            aTarget_k,vec_std   = self.read_mat_std(join(self.root,aBatch),'orig_k')
            scale_std[iB,0:int(self.nCh/2)] = vec_std[0,:]
            scale_std[iB,int(self.nCh/2):] = vec_std[0,:]

            if self.Aug:
                aTarget_k = aTarget_k*np.random.normal(loc=1.0,scale=0.01)

            aACS_k      = aTarget_k[:,:,self.ACS_s:self.ACS_e]
            input_ACSk_ri[iB,:,:,:]  = aACS_k[:,:,self.bias::self.DSrate]
            ACS1 = aACS_k[:,:,self.bias+1::self.DSrate]
            ACS2 = aACS_k[:,:,self.bias+2::self.DSrate]
            ACS3 = aACS_k[:,:,self.bias+3::self.DSrate]
            
            target_ACSk_ri[iB,:,:,:] = np.concatenate((ACS1,ACS2,ACS3),axis=0)
            input_k_ri[iB,:,:,:]  = aTarget_k[:,:,self.bias::self.DSrate]
            k1 = aTarget_k[:,:,self.bias+1::self.DSrate]
            k2 = aTarget_k[:,:,self.bias+2::self.DSrate]
            k3 = aTarget_k[:,:,self.bias+3::self.DSrate]
            target_k_ri[iB,:,:,:] = np.concatenate((k1,k2,k3),axis=0) 
        return input_ACSk_ri, target_ACSk_ri, input_k_ri, target_k_ri, scale_std 



    def getBatch_G4_withW(self, start, end):
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
            if self.Aug:
                aTarget_k = aTarget_k*np.random.normal(loc=1.0,scale=0.01)
            ''' weighting here '''
            aTarget_k = aTarget_k*self.w
            aACS_k      = aTarget_k[:,:,self.ACS_s:self.ACS_e]
            input_ACSk_ri[iB,:,:,:]  = aACS_k[:,:,self.bias::self.DSrate]
            ACS1 = aACS_k[:,:,self.bias+1::self.DSrate]
            ACS2 = aACS_k[:,:,self.bias+2::self.DSrate]
            ACS3 = aACS_k[:,:,self.bias+3::self.DSrate]
            
            target_ACSk_ri[iB,:,:,:] = np.concatenate((ACS1,ACS2,ACS3),axis=0)
            input_k_ri[iB,:,:,:]  = aTarget_k[:,:,self.bias::self.DSrate]
            k1 = aTarget_k[:,:,self.bias+1::self.DSrate]
            k2 = aTarget_k[:,:,self.bias+2::self.DSrate]
            k3 = aTarget_k[:,:,self.bias+3::self.DSrate]
            target_k_ri[iB,:,:,:] = np.concatenate((k1,k2,k3),axis=0) 
        return input_ACSk_ri, target_ACSk_ri, input_k_ri, target_k_ri, self.w[np.newaxis,:,:,:], self.uw[np.newaxis,:,:,:]

    def getBatch_G6(self, start, end):
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
            if self.Aug:
                aTarget_k = aTarget_k*np.random.normal(loc=1.0,scale=0.01)

            aACS_k      = aTarget_k[:,:,self.ACS_s:self.ACS_e]
            input_ACSk_ri[iB,:,:,:]  = aACS_k[:,:,self.bias::self.DSrate]
            ACS1 = aACS_k[:,:,self.bias+1::self.DSrate]
            ACS2 = aACS_k[:,:,self.bias+2::self.DSrate]
            ACS3 = aACS_k[:,:,self.bias+3::self.DSrate]
            ACS4 = aACS_k[:,:,self.bias+4::self.DSrate]
            ACS5 = aACS_k[:,:,self.bias+5::self.DSrate]
            target_ACSk_ri[iB,:,:,:] = np.concatenate((ACS1,ACS2,ACS3,ACS4,ACS5),axis=0)
            ''' full k'''
            input_k_ri[iB,:,:,:]  = aTarget_k[:,:,self.bias::self.DSrate]
            k1 = aTarget_k[:,:,self.bias+1::self.DSrate]
            k2 = aTarget_k[:,:,self.bias+2::self.DSrate]
            k3 = aTarget_k[:,:,self.bias+3::self.DSrate]
            k4 = aTarget_k[:,:,self.bias+4::self.DSrate]
            k5 = aTarget_k[:,:,self.bias+5::self.DSrate]
            target_k_ri[iB,:,:,:] = np.concatenate((k1,k2,k3,k4,k5),axis=0) 
        return input_ACSk_ri, target_ACSk_ri, input_k_ri, target_k_ri    


    def getBatch_G4_RAKI(self, start, end):
        end   = min([end,self.len])
        batch = self.flist[start:end]
         
        sz_       = [end-start, self.nCh, self.nY, self.dsnX]
        sz_ACS    = [end-start, self.nCh, self.nY, self.dsnACS]
        sz_out    = [end-start, self.nC, 2*(self.DSrate-1), self.nY, self.dsnX]
        sz_ACS_out= [end-start, self.nC, 2*(self.DSrate-1), self.nY, self.dsnACS]
        target_ACSk_ri  = np.empty(sz_ACS_out, dtype=np.float32)
        input_ACSk_ri   = np.empty(sz_ACS, dtype=np.float32)
        target_k_ri     = np.empty(sz_out, dtype=np.float32)
        input_k_ri      = np.empty(sz_, dtype=np.float32)

        for iB, aBatch in enumerate(batch):
            aTarget_k   = self.read_mat(join(self.root,aBatch),'orig_k')
            aACS_k      = aTarget_k[:,:,self.ACS_s:self.ACS_e]
            input_ACSk_ri[iB,:,:,:]  = aACS_k[:,:,self.bias::self.DSrate]
            target_ACSk_ri[iB,:,0,:,:] = aACS_k[:self.nC,:,self.bias+1::self.DSrate]
            target_ACSk_ri[iB,:,1,:,:] = aACS_k[self.nC:,:,self.bias+1::self.DSrate]
            target_ACSk_ri[iB,:,2,:,:] = aACS_k[:self.nC,:,self.bias+2::self.DSrate]
            target_ACSk_ri[iB,:,3,:,:] = aACS_k[self.nC:,:,self.bias+2::self.DSrate]
            target_ACSk_ri[iB,:,4,:,:] = aACS_k[:self.nC,:,self.bias+3::self.DSrate]
            target_ACSk_ri[iB,:,5,:,:] = aACS_k[self.nC:,:,self.bias+3::self.DSrate]

            input_k_ri[iB,:,:,:]  = aTarget_k[:,:,self.bias::self.DSrate]
            target_k_ri[iB,:,0,:,:] = aTarget_k[:self.nC,:,self.bias+1::self.DSrate]
            target_k_ri[iB,:,1,:,:] = aTarget_k[self.nC:,:,self.bias+1::self.DSrate]
            target_k_ri[iB,:,2,:,:] = aTarget_k[:self.nC,:,self.bias+2::self.DSrate]
            target_k_ri[iB,:,3,:,:] = aTarget_k[self.nC:,:,self.bias+2::self.DSrate]
            target_k_ri[iB,:,4,:,:] = aTarget_k[:self.nC,:,self.bias+3::self.DSrate]
            target_k_ri[iB,:,5,:,:] = aTarget_k[self.nC:,:,self.bias+3::self.DSrate]
            
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
    def read_mat_std(filename, var_name="img"):
        mat = sio.loadmat(filename)
        return mat[var_name], mat['vec_std']

    @staticmethod
    def read_mat_meta(filename):
        mat = sio.loadmat(filename)
        return mat['ACS_inp'], mat['ACS_out'], mat['orig_inp'], mat['orig_out']
if __name__ == "__main__":
    tmp = DB7T_8ch('../../data/MRI')

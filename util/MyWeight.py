import numpy as np

class MyWeight():
    def __init__(self, nY,nX,nCh,wtype,scale=0.1):
        self.hny = int(nY/2)
        self.hnx = int(nX/2)
        ax_x, ax_y = np.mgrid[-self.hny:self.hny,-self.hnx:self.hnx]
        z          = np.sqrt( np.square(ax_x*scale) + np.square(ax_y*scale) )
        
        if wtype == '1white':
            w = np.abs(z)
        elif wtype == '2white':
            w = np.square(np.abs(z))
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

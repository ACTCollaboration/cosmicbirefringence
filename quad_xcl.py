# Reconstruction using quadratic estimator
import numpy as np
import healpy as hp
import cPickle as pickle
import curvedsky
import basic
import prjlib
import quad_func


p, f, r = prjlib.init(loadw=False)
p0, f0 = prjlib.filename_init(PSA='s14&15_deep56')
p1, f1 = prjlib.filename_init(PSA='s14&15_boss')
__, __, w1, __ = prjlib.window(f0)
__, __, w2, __ = prjlib.window(f1)
r.w4 = np.sqrt(w1*w2)

#//// Power spectrum ////#
oLmax  = p.quad.oLmax
bn     = p.quad.bn
binspc = p.quad.binspc

for q in p.quad.qlist:

    cl = np.zeros((p.snmax,1,oLmax+1))
    cb = np.zeros((p.snmax,1,bn))
  
    for i in range(p.snmax):

        print(i)
        glm1, clm = pickle.load(open(p0.quad.f[q].alm[i],"rb"))
        mfg1, mfc = pickle.load(open(p0.quad.f[q].mfb[i],"rb"))
        glm2, clm = pickle.load(open(p1.quad.f[q].alm[i],"rb"))
        mfg2, mfc = pickle.load(open(p1.quad.f[q].mfb[i],"rb"))
        glm1 -= mfg1
        glm2 -= mfg2

        # correct bias terms and MC noise due to mean-field bias
        cl[i,0,:] = curvedsky.utils.alm2cl(oLmax,glm1,glm2)/r.w4
        for j in range(1):
            cb[i,j,:] = basic.aps.cl2bcl(bn,oLmax,cl[i,j,:],spc=binspc)
            np.savetxt(f.nul[q].xl[i],np.concatenate((p.quad.bc[None,:],cb[i,:,:])).T)

    # save to file
    if p.snmax>=2:
        print('save sim') 
        np.savetxt(f.nul[q].mxls,np.concatenate((p.quad.eL[None,:],np.mean(cl[1:,:,:],axis=0),np.std(cl[1:,:,:],axis=0))).T)
        np.savetxt(f.nul[q].mxbs,np.concatenate((p.quad.bc[None,:],np.mean(cb[1:,:,:],axis=0),np.std(cb[1:,:,:],axis=0))).T)

    if p.snmin==0:
        print('save real')
        np.savetxt(f.nul[q].oxls,np.concatenate((p.quad.eL[None,:],cl[0,:,:])).T)
        np.savetxt(f.nul[q].oxbs,np.concatenate((p.quad.bc[None,:],cb[0,:,:])).T)


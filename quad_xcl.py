# Reconstruction using quadratic estimator
import numpy as np
import healpy as hp
import pickle
import curvedsky
import basic
import prjlib
import quad_func


p, f, r = prjlib.init(loadw=False)
p0, f0 = prjlib.filename_init(PSA='s14&15_deep56',doreal='True',dearot='True')
p1, f1 = prjlib.filename_init(PSA='s14&15_boss',doreal='True')
px, fx = prjlib.filename_init(PSA='s14&15_cross',doreal='True')
__, __, w1, __ = prjlib.window(f0)
__, __, w2, __ = prjlib.window(f1)
r.w4 = np.sqrt(w1*w2)

#//// bias terms ////#
oc0 = prjlib.loadocl(f0.scl)
oc1 = prjlib.loadocl(f1.scl)
quad_func.quad.diagcinv(p0.quad,oc0)
quad_func.quad.diagcinv(p1.quad,oc1)
#quad_func.n0x(px.quad,p0.quad,p1.quad,f0.alm,f1.alm,r.w4,r.lcl)
#quad_func.rdn0x(px.quad,p0.quad,p1.quad,0,1,f0.alm,f1.alm,r.w4,r.lcl)

#//// Power spectrum ////#
oLmax  = p.quad.oLmax
bn     = p.quad.bn
binspc = p.quad.binspc

for q in p.quad.qlist:

    cl = np.zeros((p.snmax,1,oLmax+1))
  
    for i in range(p.snmax):

        print(i)
        glm1, clm = pickle.load(open(p0.quad.f[q].alm[i],"rb"))
        mfg1, mfc = pickle.load(open(p0.quad.f[q].mfb[i],"rb"))
        glm2, clm = pickle.load(open(p1.quad.f[q].alm[i],"rb"))
        mfg2, mfc = pickle.load(open(p1.quad.f[q].mfb[i],"rb"))
        glm1 -= mfg1
        glm2 -= mfg2

        #if i==0:
        #    rdn0 = np.loadtxt(px.quad.f[q].rdn0[i],unpack=True)[1]
        #else:
        #    rdn0 = 0.

        # correct bias terms and MC noise due to mean-field bias
        cl[i,0,:] = curvedsky.utils.alm2cl(oLmax,glm1,glm2)/r.w4 #- rdn0
        if i>0:  np.savetxt(px.quad.f[q].cl[i],np.concatenate((p.quad.eL[None,:],cl[i,:,:])).T)

    # save to file
    mb = prjlib.multipole_binning(p.quad.bn,spc=p.quad.binspc)
    cb = prjlib.binning(cl,mb)
    if p.snmax>=2:
        print('save sim')
        np.savetxt(px.quad.f[q].mcls,np.concatenate((p.quad.eL[None,:],np.mean(cl[1:,:,:],axis=0),np.std(cl[1:,:,:],axis=0))).T)
        np.savetxt(px.quad.f[q].mcbs,np.concatenate((p.quad.bc[None,:],np.mean(cb[1:,:,:],axis=0),np.std(cb[1:,:,:],axis=0))).T)

    if p.snmin==0:
        print('save real')
        np.savetxt(px.quad.f[q].ocls,np.concatenate((p.quad.eL[None,:],cl[0,:,:])).T)
        np.savetxt(px.quad.f[q].ocbs,np.concatenate((p.quad.bc[None,:],cb[0,:,:])).T)


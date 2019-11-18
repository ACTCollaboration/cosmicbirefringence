# Reconstruction using quadratic estimator
import numpy as np
import healpy as hp
import cPickle as pickle
import curvedsky
import basic
import prjlib
import quad_func


p, f, r = prjlib.init()

#//// Reconstruction ////#
ocl = prjlib.loadocl(f.scl)
quad_func.quad.diagcinv(p.quad,ocl)
quad_func.quad.al(p.quad,r.lcl,ocl)
quad_func.quad.qrec(p.quad,p.snmin,p.snmax,f.alm,r.lcl)
quad_func.quad.n0(p.quad,f.alm,r.w4,r.lcl)
#quad_func.quad.rdn0(p.quad,p.snmin,p.snmax,f.alm,r.w4,r.lcl)
quad_func.quad.mean(p.quad,p.snmin,p.snmax,r.w4)

#//// Power spectrum ////#
oLmax = p.quad.oLmax
bn    = p.quad.bn

for q in p.quad.qlist:

    cl = np.zeros((p.snmax,4,oLmax+1))
    cb = np.zeros((p.snmax,4,bn))
    n0 = np.loadtxt(p.quad.f[q].n0bl,unpack=True,usecols=(1,2))
  
    for i in range(p.snmax):

        print(i)
        glm, clm = pickle.load(open(p.quad.f[q].alm[i],"rb"))
        mfg, mfc = pickle.load(open(p.quad.f[q].mfb[i],"rb"))
        glm -= mfg
        clm -= mfc
        if p.quad.qtype == 'lens':
            klm = np.complex128(hp.fitsfunc.read_alm(f.palm[i]))
            klm = curvedsky.utils.lm_healpy2healpix(len(klm),klm,5100)[:oLmax+1,:oLmax+1]
            klm *= r.kL[:,None]
        if p.quad.qtype == 'rot':
            klm = pickle.load(open(f.aalm[i],"rb"))[:oLmax+1,:oLmax+1]

        if i==0:
            #rdn0 = np.loadtxt(f.quad[q].rdn0[i],unpack=True,usecols=(1,2))
            rdn0 = n0 + n0/p.quad.snmf 
        else:
            rdn0 = n0 + n0/(p.quad.snmf-1.)

        # correct bias terms and MC noise due to mean-field bias
        cl[i,0,:] = curvedsky.utils.alm2cl(oLmax,glm)/r.w4 - rdn0[0,:]
        cl[i,1,:] = curvedsky.utils.alm2cl(oLmax,clm)/r.w4 - rdn0[1,:]
        cl[i,2,:] = curvedsky.utils.alm2cl(oLmax,glm,klm)/r.w2
        cl[i,3,:] = curvedsky.utils.alm2cl(oLmax,klm)
        for j in range(4):
            cb[i,j,:] = basic.aps.cl2bcl(bn,oLmax,cl[i,j,:],spc=p.quad.binspc)
            np.savetxt(p.quad.f[q].cl[i],np.concatenate((p.quad.bc[None,:],cb[i,:,:])).T)

    # save to file
    if p.snmax>=2:
        print('save sim') 
        np.savetxt(p.quad.f[q].mcls,np.concatenate((p.quad.eL[None,:],np.mean(cl[1:,:,:],axis=0),np.std(cl[1:,:,:],axis=0))).T)
        np.savetxt(p.quad.f[q].mcbs,np.concatenate((p.quad.bc[None,:],np.mean(cb[1:,:,:],axis=0),np.std(cb[1:,:,:],axis=0))).T)

    if p.snmin==0:
        print('save real')
        np.savetxt(p.quad.f[q].ocls,np.concatenate((p.quad.eL[None,:],cl[0,:,:])).T)
        np.savetxt(p.quad.f[q].ocbs,np.concatenate((p.quad.bc[None,:],cb[0,:,:])).T)


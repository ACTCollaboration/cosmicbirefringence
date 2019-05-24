# Reconstruction using quadratic estimator
import numpy as np
import healpy as hp
import curvedsky
import basic
import prjlib
import cPickle as pickle
import quad_func


p, f, r = prjlib.init()
prjlib.make_qrec_filter(p,f,r)


#//// Reconstruction ////#
quad_func.al(p,f,r)
quad_func.qrec(p,f.alm,f.quad,r)
quad_func.n0(p,f.alm,f.quad,r)
#quad_func.rdn0(p,f.alm,f.quad,r)
quad_func.mean(p,f.quad,r)


#//// Power spectrum ////#
for q in p.qlist:

    cl = np.zeros((p.snmax,4,p.lmax+1))
    cb = np.zeros((p.snmax,4,p.bn))
    n0 = np.loadtxt(f.quad[q].n0bl,unpack=True,usecols=(1,2))
  
    for i in range(p.snmax):

        print(i)
        glm, clm = pickle.load(open(f.quad[q].alm[i],"rb"))
        mfg, mfc = pickle.load(open(f.quad[q].mfb[i],"rb"))
        if p.qtype == 'lens':
            klm = np.complex128(hp.fitsfunc.read_alm(f.palm[i]))
            klm = curvedsky.utils.lm_healpy2healpix(len(klm),klm,5100)[:p.lmax+1,:p.lmax+1]
            klm *= r.kL[:,None]
        if p.qtype == 'rot':
            klm = pickle.load(open(f.aalm[i],"rb"))

        if i==0:
            try:
                rdn0 = np.loadtxt(f.quad[q].rdn0[i],unpack=True,usecols=(1,2))
            except:
                print('no file for RDN0, use N0')
                rdn0 = n0
                rdn0 = n0 + n0/p.snmf 
        else:
            rdn0 = n0 + n0/(p.snmf-1.)

        # correct bias terms and MC noise due to mean-field bias
        #////#
        if i==0 and p.doreal:
            cl[i,0,:] = curvedsky.utils.alm2cl(p.lmax,glm)/r.W4 - 2*curvedsky.utils.alm2cl(p.lmax,glm,mfg)/r.Wx + curvedsky.utils.alm2cl(p.lmax,mfg)/r.w4 - rdn0[0,:]
            cl[i,1,:] = curvedsky.utils.alm2cl(p.lmax,clm)/r.W4 - 2*curvedsky.utils.alm2cl(p.lmax,clm,mfc)/r.Wx + curvedsky.utils.alm2cl(p.lmax,mfc)/r.w4 - rdn0[1,:]
        else:
            cl[i,0,:] = curvedsky.utils.alm2cl(p.lmax,glm-mfg)/r.w4 - rdn0[0,:]
            cl[i,1,:] = curvedsky.utils.alm2cl(p.lmax,clm-mfc)/r.w4 - rdn0[1,:]
            cl[i,2,:] = curvedsky.utils.alm2cl(p.lmax,glm-mfg,klm)/r.w2
            cl[i,3,:] = curvedsky.utils.alm2cl(p.lmax,klm)
        #////#
        for j in range(4):
            cb[i,j,:] = basic.aps.cl2bcl(p.bn,p.lmax,cl[i,j,:],spc=p.binspc)
            np.savetxt(f.quad[q].cl[i],np.concatenate((r.bc[None,:],cb[i,:,:])).T)

    # save to file
    if p.snmax>=2:
        print('save sim') 
        np.savetxt(f.quad[q].mcls,np.concatenate((r.eL[None,:],np.mean(cl[1:,:,:],axis=0),np.std(cl[1:,:,:],axis=0))).T)
        np.savetxt(f.quad[q].mcbs,np.concatenate((r.bc[None,:],np.mean(cb[1:,:,:],axis=0),np.std(cb[1:,:,:],axis=0))).T)

    if p.snmin==0:
        if p.doreal:
            print('save real')
            np.savetxt(f.quad[q].rcls,np.concatenate((r.eL[None,:],cl[0,:,:])).T)
            np.savetxt(f.quad[q].rcbs,np.concatenate((r.bc[None,:],cb[0,:,:])).T)
        else:
            print('save mock obs')
            np.savetxt(f.quad[q].ocls,np.concatenate((r.eL[None,:],cl[0,:,:])).T)
            np.savetxt(f.quad[q].ocbs,np.concatenate((r.bc[None,:],cb[0,:,:])).T)


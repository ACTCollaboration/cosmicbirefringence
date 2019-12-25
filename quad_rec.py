# Reconstruction using quadratic estimator
import numpy as np
import healpy as hp
import os
import pickle
import curvedsky
import basic
import prjlib
import quad_func


def qrec_aps(pquad,q,snmin,snmax,f,r,psquad,stype):

    oLmax = pquad.oLmax
    cl = np.zeros((snmax,4,oLmax+1))
    n0 = np.loadtxt(psquad.f[q].n0bl,unpack=True,usecols=(1,2))
  
    for i in range(snmax):

        if os.path.exists(pquad.f[q].cl[i]): 
            continue

        print(i)
        glm, clm = pickle.load(open(pquad.f[q].alm[i],"rb"))
        mfg, mfc = pickle.load(open(psquad.f[q].mfb[i],"rb"))
        glm -= mfg
        clm -= mfc
        if pquad.qtype == 'lens':
            klm = np.complex128(hp.fitsfunc.read_alm(f.palm[i]))
            klm = curvedsky.utils.lm_healpy2healpix(len(klm),klm,5100)[:oLmax+1,:oLmax+1]
            klm *= r.kL[:,None]
        if pquad.qtype == 'rot':
            if stype !='lcmb':
                klm = pickle.load(open(f.aalm[i],"rb"))[:oLmax+1,:oLmax+1]
            else:
                klm = 0.*glm

        if i==0: #and p.doreal:
            rdn0 = np.loadtxt(p.quad.f[q].rdn0[i],unpack=True,usecols=(1,2))
            #rdn0 = n0 + n0/pquad.snmf 
        else:
            rdn0 = n0 + n0/(pquad.snmf-1.)

        # correct bias terms and MC noise due to mean-field bias
        cl[i,0,:] = curvedsky.utils.alm2cl(oLmax,glm)/r.w4 - rdn0[0,:]
        cl[i,1,:] = curvedsky.utils.alm2cl(oLmax,clm)/r.w4 - rdn0[1,:]
        cl[i,2,:] = curvedsky.utils.alm2cl(oLmax,glm,klm)/r.w2
        cl[i,3,:] = curvedsky.utils.alm2cl(oLmax,klm)
        if i > 0:  np.savetxt(pquad.f[q].cl[i],np.concatenate((pquad.eL[None,:],cl[i,:,:])).T)

    # save to file
    mb = prjlib.multipole_binning(pquad.bn,spc=pquad.binspc)
    cb = prjlib.binning(cl,mb)
    if snmax>=2:
        print('save sim') 
        np.savetxt(pquad.f[q].mcls,np.concatenate((pquad.eL[None,:],np.mean(cl[1:,:,:],axis=0),np.std(cl[1:,:,:],axis=0))).T)
        np.savetxt(pquad.f[q].mcbs,np.concatenate((pquad.bc[None,:],np.mean(cb[1:,:,:],axis=0),np.std(cb[1:,:,:],axis=0))).T)

    if snmin==0:
        print('save real')
        np.savetxt(pquad.f[q].ocls,np.concatenate((pquad.eL[None,:],cl[0,:,:])).T)
        np.savetxt(pquad.f[q].ocbs,np.concatenate((pquad.bc[None,:],cb[0,:,:])).T)


p, f, r = prjlib.init()
ps, fs, _ = prjlib.init(stype='lcmb',dodust='False')

#//// Reconstruction ////#
ow = False
#ow = True
snmax = p.snmax
if p.stype!='lcmb': snmax = 101

if p.stype in ['absrot','relrot'] or p.dodust:
    ow = True
    ocl = prjlib.loadocl(fs.scl)
    quad_func.quad.diagcinv(ps.quad,ocl)
    quad_func.quad.diagcinv(p.quad,ocl)
    quad_func.quad.qrec(ps.quad,p.snmin,snmax,f.alm,r.lcl,qout=p.quad,overwrite=ow)
else:
    ps = p
    ocl = prjlib.loadocl(f.scl)
    quad_func.quad.diagcinv(p.quad,ocl)
    quad_func.quad.al(p.quad,r.lcl,ocl)
    quad_func.quad.qrec(p.quad,p.snmin,snmax,f.alm,r.lcl,overwrite=ow)
    quad_func.quad.n0(p.quad,f.alm,r.w4,r.lcl,overwrite=ow)
    #quad_func.quad.diagrdn0(p.quad,p.snmax,r.lcl,ocl,f.cli)
    #quad_func.quad.mean(p.quad,p.snmin,snmax,r.w4,overwrite=ow)

#RDN0
quad_func.quad.rdn0(p.quad,0,p.snmax,f.alm,r.w4,r.lcl)

#//// Power spectrum ////#
for q in p.quad.qlist:
    qrec_aps(p.quad,q,p.snmin,snmax,f,r,ps.quad,p.stype)


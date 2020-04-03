# Reconstruction using quadratic estimator
import numpy as np
import healpy as hp
import os
import pickle
import curvedsky
import basic
import prjlib
import quad_func


def qrec_aps(pquad,q,snmin,snmax,f,r,psquad,stype,rdn0=True,overwrite=False):

    oLmax = pquad.oLmax
    cl = np.zeros((snmax+1,4,oLmax+1))
    n0 = np.loadtxt(psquad.f[q].n0bs,unpack=True,usecols=(1,2))
  
    #mfg, mfc = pickle.load(open(psquad.f[q].mf,"rb"))

    for i in range(snmin,snmax+1):

        if os.path.exists(pquad.f[q].cl[i]) and not overwrite:
            continue

        print(i)
        glm, clm = pickle.load(open(pquad.f[q].alm[i],"rb"))
        #mfg, mfc = pickle.load(open(psquad.f[q].mfb[i],"rb"))
        #glm -= mfg
        #clm -= mfc
        if pquad.qtype == 'lens':
            klm = np.complex128(hp.fitsfunc.read_alm(f.palm[i]))
            klm = curvedsky.utils.lm_healpy2healpix(len(klm),klm,5100)[:oLmax+1,:oLmax+1]
            klm *= r.kL[:,None]
        if pquad.qtype == 'rot':
            if stype !='lcmb':
                klm = pickle.load(open(f.aalm[i],"rb"))[:oLmax+1,:oLmax+1]
            else:
                klm = 0.*glm

        if i==0 and rdn0:
            rdn0 = np.loadtxt(p.quad.f[q].rdn0[i],unpack=True,usecols=(1,2))
        else:
            #rdn0 = n0 + n0/(pquad.mfsim-1.)
            rdn0 = n0 #+ n0/pquad.mfsim

        # correct bias terms and MC noise due to mean-field bias
        cl[i,0,:] = curvedsky.utils.alm2cl(oLmax,glm)/r.w4 - rdn0[0,:]
        cl[i,1,:] = curvedsky.utils.alm2cl(oLmax,clm)/r.w4 - rdn0[1,:]
        cl[i,2,:] = curvedsky.utils.alm2cl(oLmax,glm,klm)/r.w2
        cl[i,3,:] = curvedsky.utils.alm2cl(oLmax,klm)
        np.savetxt(pquad.f[q].cl[i],np.concatenate((pquad.eL[None,:],cl[i,:,:])).T)

    # save to file
    if snmax>=1:
        print('save sim average') 
        np.savetxt(pquad.f[q].mcls,np.concatenate((pquad.eL[None,:],np.mean(cl[1:,:,:],axis=0),np.std(cl[1:,:,:],axis=0))).T)


p, f, r = prjlib.init()
ps, fs, _ = prjlib.init(stype='lcmb',dodust='False')

#//// Reconstruction ////#
ow = False
ow = True
snmax = p.snmax
if p.stype not in ['lcmb','dust']: snmax = 100

if p.stype in ['absrot','relrot','dust']:
    rdn0 = False
    ocl = prjlib.loadocl(fs.scl)
    quad_func.quad.cinvfilter(ps.quad,ocl)
    quad_func.quad.cinvfilter(p.quad,ocl)
    quad_func.quad.qrec(ps.quad,p.snmin,snmax,f.alm,r.lcl,qout=p.quad,overwrite=ow)
    if p.stype=='dust':
        rdn0 = True
        quad_func.quad.rdn0(ps.quad,p.snmin,p.snmax,f.alm,r.w4,r.lcl,qout=p.quad,overwrite=ow,falms=fs.alm)
else:
    rdn0 = True
    ps = p
    ocl = prjlib.loadocl(f.scl)
    quad_func.quad.cinvfilter(p.quad,ocl)
    quad_func.quad.al(p.quad,r.lcl,ocl)
    quad_func.quad.qrec(p.quad,p.snmin,snmax,f.alm,r.lcl,overwrite=ow)
    quad_func.quad.n0(p.quad,f.alm,r.w4,r.lcl,overwrite=ow)
    #quad_func.quad.qrec(p.quad,p.quad.mfmin,p.quad.mfmax,f.alm,r.lcl,overwrite=ow)
    #quad_func.quad.mean(p.quad,r.w4,overwrite=ow)
    #if p.PSA == 's14&15_deep56':
    #quad_func.quad.rdn0(p.quad,p.snmin,p.snmax,f.alm,r.w4,r.lcl,overwrite=ow)
    quad_func.quad.rdn0(p.quad,0,0,f.alm,r.w4,r.lcl,overwrite=ow)

#//// Power spectrum ////#
#if p.PSA == 's14&15_deep56':
if p.PSA == 's14&15_boss':
    ow = True
    for q in p.quad.qlist:
        qrec_aps(p.quad,q,p.snmin,p.snmax,f,r,ps.quad,p.stype,rdn0=rdn0,overwrite=ow)


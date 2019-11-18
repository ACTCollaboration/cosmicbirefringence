# quadratic lensing reconstruction
import numpy as np
import healpy as hp
import curvedsky
import basic
import prjlib
import cPickle as pickle

p = prjlib.params('s14&15_comb')
f = prjlib.filename(p)
r = prjlib.recfunc(p,f)

p0, f0, r0 = prjlib.init(PSA='s14&15_deep56')
p1, f1, r1 = prjlib.init(PSA='s14&15_boss')

for q in p.qlist:

    # load averaged power spectrum and its variance from the fields
    cb = np.zeros((p.snmax,2,p.Bn))
    mcb0, kx0, ik0, vcb0 = np.loadtxt(f0.quad[q].mcbs,unpack=True,usecols=(1,3,4,5))
    mcb1, kx1, ik1, vcb1 = np.loadtxt(f1.quad[q].mcbs,unpack=True,usecols=(1,3,4,5))

    # correct normalization
    s0 = ik0/kx0
    s1 = ik1/kx1
    vcb0 *= s0**2
    vcb1 *= s1**2

    for i in range(p.snmax):
        # linear combination
        cb0 = s0**2 * np.loadtxt(f0.quad[q].cl[i],unpack=True)[1]
        cb1 = s1**2 * np.loadtxt(f1.quad[q].cl[i],unpack=True)[1]
        cb[i,0,:] = ( cb0/vcb0**2 + cb1/vcb1**2 ) / (1./vcb0**2+1./vcb1**2)
        cb[i,1,:] = ik0
        np.savetxt(f.quad[q].cl[i],np.concatenate((r.Bc[None,:],cb[i,:,:])).T)

    # save to file
    if p.snmax>=2:
        print('save sim')
        v = np.std(cb[1:,:,:],axis=0)
        print(v[0,:]/vcb0)
        print(v[0,:]/vcb1)
        np.savetxt(f.quad[q].mcbs,np.concatenate((r.Bc[None,:],np.mean(cb[1:,:,:],axis=0),np.std(cb[1:,:,:],axis=0))).T)

    if p.snmin==0:
        if p.doreal:
            print('save real')
            np.savetxt(f.quad[q].rcbs,np.concatenate((r.Bc[None,:],cb[0,:,:],np.std(cb[1:,:,:],axis=0))).T)
        else:
            print('save mock obs')
            np.savetxt(f.quad[q].ocbs,np.concatenate((r.Bc[None,:],cb[0,:,:],np.std(cb[1:,:,:],axis=0))).T)


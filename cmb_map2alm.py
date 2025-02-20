# Compute harmonic coefficients of CMB maps and power spectrum
import numpy as np
import healpy as hp
import sys
import pickle

# ACT library
from pixell import enmap

# cmblensplus
import basic
import curvedsky
import binning as bins

# local
import prjlib


def mask_kspace(shape,wcs, lxcut = None, lycut = None, lmin = None, lmax = None):
    # copied from orphics
    output = np.ones(shape[-2:], dtype = int)
    if (lmin is not None) or (lmax is not None): modlmap = enmap.modlmap(shape, wcs)
    if (lxcut is not None) or (lycut is not None): ly, lx = enmap.laxes(shape, wcs, oversample=1)
    if lmin is not None:
        output[np.where(modlmap <= lmin)] = 0
    if lmax is not None:
        output[np.where(modlmap >= lmax)] = 0
    if lxcut is not None:
        output[:,np.where(np.abs(lx) < lxcut)] = 0
    if lycut is not None:
        output[np.where(np.abs(ly) < lycut),:] = 0
    return output


def cmbmap2alm(i,mtype,p,f,r):  

    fmap = enmap.read_map(f.imap[mtype][i])  # load flatsky K-space combined map

    # FT
    print('2D ell filtering in F-space and assign to healpix map')
    alm = enmap.fft(fmap)

    # remove some Fourier modes   
    shape, wcs = fmap.shape, fmap.wcs
    kmask = mask_kspace(shape,wcs,lmin=p.lcut,lmax=4000,lxcut=90,lycut=50)
    alm[kmask<0.5] = 0

    # alm -> map
    fmap = enmap.ifft(alm).real

    # transform cmb map to healpix
    hpmap = enmap.to_healpix(fmap,nside=p.nside)

    # from map to alm
    hpmap = r.w * hpmap  # masking
    #hp.fitsfunc.write_map(f.omap[mtype][i],hpmap,overwrite=True)
    alm = curvedsky.utils.hp_map2alm(p.nside,p.lmax,p.lmax,hpmap)

    print("save to file")
    pickle.dump((alm),open(f.alm[mtype][i],"wb"),protocol=pickle.HIGHEST_PROTOCOL)


def cmbalm2cl(f,w2,snmin,snmax,lmax,verbose=True,bn=50,spc='',mlist=['T','E','B']):
    #//// compute cls from alm ////#
    # output order is ell, TT, EE, BB, TE, TB, EB

    L   = np.linspace(0,lmax,lmax+1)
    mb  = bins.multipole_binning(bn,spc=spc,lmax=lmax)
    cbs = np.zeros((snmax+1,6,bn))
    cls = np.zeros((snmax+1,6,lmax+1))

    for i in range(snmin,snmax+1):

        if varbose: print('load alm', i)

        #load cmb alms
        Ealm = pickle.load(open(f.alm['E'][i],"rb"))
        Balm = pickle.load(open(f.alm['B'][i],"rb"))
        if 'T' in mlist:
            Talm = pickle.load(open(f.alm['T'][i],"rb"))
        else:
            Talm = 0*Ealm

        #compute cls
        cls[i,0,:] = curvedsky.utils.alm2cl(lmax,Talm)
        cls[i,1,:] = curvedsky.utils.alm2cl(lmax,Ealm)
        cls[i,2,:] = curvedsky.utils.alm2cl(lmax,Balm)
        cls[i,3,:] = curvedsky.utils.alm2cl(lmax,Talm,Ealm)
        cls[i,4,:] = curvedsky.utils.alm2cl(lmax,Talm,Balm)
        cls[i,5,:] = curvedsky.utils.alm2cl(lmax,Ealm,Balm)
        cls[i,:,:] *= 1./w2

        #compute binned cls
        for j in range(6):
            cbs[i,j,:] = basic.aps.cl2bcl(bn,lmax,cls[i,j,:],spc=spc)

        #save cl at each rlz
        np.savetxt(f.cli[i],np.concatenate((L[None,:],cls[i,:,:])).T)

    # save to files
    if snmax>=1 and not os.path.exists(f.scl): 
        if verbose: print('save sim')
        i0 = max(1,snmin)
        np.savetxt(f.scl,np.concatenate((L[None,:],np.mean(cls[i0:,:,:],axis=0),np.std(cls[i0:,:,:],axis=0))).T)


if __name__ == '__main__':
    import os

    #define parameters, filenames and functions
    p, f, r = prjlib.init()

    #loop for T/E/B at each realization
    for mtype in p.mlist:
        for i in range(p.snmin,p.snmax+1):
            if not os.path.exists(f.alm[mtype][i]):
                print("map to alm", i, end=" ")
                cmbmap2alm(i,mtype,p,f,r)

    # compute cl
    cmbalm2cl(f,r.w2,p.snmin,p.snmax,p.lmax,bn=p.bn,spc=p.binspc,mlist=p.mlist)



# Compute harmonic coefficients of CMB maps and power spectrum
import numpy as np
import healpy as hp
import sys
import cPickle as pickle
from memory_profiler import profile
from orphics import maps, stats
from pixell import enmap

#from cmblensplus
import basic
import curvedsky

#local module
import prjlib


def cmbmap2alm(i,mtype,p,f,r):  

    print 'map to alm', i

    fmap = enmap.read_map(f.imap[mtype][i])  # load flatsky K-space combined map
    print 'map amplitude', np.average(fmap**2)

    # FT
    print 'compute Fourier modes'
    alm = enmap.fft(fmap)

    # remove some Fourier modes   
    print 'define lmask'
    ellmin = 300
    ellmax = 4000
    lxcut  = 90
    lycut  = 50
    shape, wcs = fmap.shape, fmap.wcs
    kmask = maps.mask_kspace(shape,wcs,lmin=ellmin,lmax=ellmax,lxcut=lxcut,lycut=lycut)
    alm[kmask<0.5] = 0

    # alm -> map
    print 'compute filtered map'
    fmap = enmap.ifft(alm).real

    # transform cmb map to healpix
    print('transform to healpix')
    hpmap = enmap.to_healpix(fmap)

    # from map to alm
    hpmap = r.w * hpmap  # masking
    #hp.fitsfunc.write_map(f.omap[mtype][i],hpmap,overwrite=True)
    alm = curvedsky.utils.hp_map2alm(p.nside,p.lmax,p.lmax,hpmap)

    print("save to file")
    pickle.dump((alm),open(f.alm[mtype][i],"wb"),protocol=pickle.HIGHEST_PROTOCOL)


@profile
def cmbalm2cl(p,f,r):
    #//// compute aps ////#
    # output is ell, TT, EE, BB, TE, TB, EB

    cbs = np.zeros((p.snmax,6,p.bn))
    cls = np.zeros((p.snmax,6,p.lmax+1))

    for i in range(p.snmin,p.snmax):

        print('load alm', i)

        #load cmb alms
        Ealm = pickle.load(open(f.alm['E'][i],"rb"))
        Balm = pickle.load(open(f.alm['B'][i],"rb"))
        if 'T' in p.mlist:
            Talm = pickle.load(open(f.alm['T'][i],"rb"))
        else:
            Talm = 0*Ealm

        #compute cls
        cls[i,0,:] = curvedsky.utils.alm2cl(p.lmax,Talm)
        cls[i,1,:] = curvedsky.utils.alm2cl(p.lmax,Ealm)
        cls[i,2,:] = curvedsky.utils.alm2cl(p.lmax,Balm)
        cls[i,3,:] = curvedsky.utils.alm2cl(p.lmax,Talm,Ealm)
        cls[i,4,:] = curvedsky.utils.alm2cl(p.lmax,Talm,Balm)
        cls[i,5,:] = curvedsky.utils.alm2cl(p.lmax,Ealm,Balm)
        cls[i,:,:] *= 1./r.w2

        #compute binned cls
        for j in range(6):
            cbs[i,j,:] = basic.aps.cl2bcl(p.bn,p.lmax,cls[i,j,:],spc=p.binspc)

        #save cl at each rlz
        if p.stype=='lcmb':
            np.savetxt(f.cbi[i],np.concatenate((r.bc[None,:],cbs[i,:,:])).T)

    # save to files
    if p.snmax>=2:
        print('save sim')
        i0 = max(1,p.snmin)
        np.savetxt(f.scl,np.concatenate((r.eL[None,:],np.mean(cls[i0:,:,:],axis=0),np.std(cls[i0:,:,:],axis=0))).T)
        np.savetxt(f.scb,np.concatenate((r.bc[None,:],np.mean(cbs[i0:,:,:],axis=0),np.std(cbs[i0:,:,:],axis=0))).T)

    if p.snmin==0:
        print('save real')
        np.savetxt(f.ocl,np.concatenate((r.eL[None,:],cls[0,:,:])).T)
        np.savetxt(f.ocb,np.concatenate((r.bc[None,:],cbs[0,:,:])).T)


if __name__ == '__main__':

    #define parameters, filenames and functions
    p, f, r = prjlib.init()

    #loop for T/E/B at each realization
    for mtype in p.mlist:

        for i in range(p.snmin,p.snmax):

            print("map to alm", i)
            cmbmap2alm(i,mtype,p,f,r)

    # compute cl
    cmbalm2cl(p,f,r)



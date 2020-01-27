import numpy as np
import healpy as hp
import curvedsky
import prjlib
from pixell import enmap

def transmask(nside,ffmask,frmask):  # transform to fullsky mask
    hpmap = enmap.to_healpix(enmap.read_map(ffmask),nside=nside)
    print(np.max(hpmap))
    hpmap = hpmap/np.max(hpmap)
    hpmap[hpmap<0] = 0.
    hp.fitsfunc.write_map(frmask,hpmap,overwrite=True)


def apodize(frmask,famask,ascale):  # compute apodized window
    mask  = hp.fitsfunc.read_map(frmask) #load mask
    npix  = np.shape(mask)
    amask = curvedsky.utils.apodize(npix,mask,ascale) # apodization window
    hp.fitsfunc.write_map(famask,amask,overwrite=True) # save


for psa in ['s14&15_deep56','s14&15_boss']:
    p = prjlib.params(PSA=psa)
    f = prjlib.filename(p)
    transmask(p.nside,f.fmask,f.rmask)
    apodize(f.rmask,f.amask,p.ascale)

# view
#amask = hp.fitsfunc.read_map(f.omap['T'][0])
#amask = enmap.to_healpix(enmap.read_map(f.aalm[0]))
#from matplotlib.pyplot import *
#hp.mollview(amask)
#savefig('fig_amask_a'+str(p.ascale)+'deg_'+p.psa+'.png')
#show()


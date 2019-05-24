import numpy as np
import healpy as hp
from pixell import enmap

import curvedsky
import prjlib


transmask = True
apodize   = True

p = prjlib.params()
f = prjlib.filename(p)

# transform to fullsky mask
if transmask:
    hpmap = enmap.to_healpix(enmap.read_map(f.fmask))
    print(np.max(hpmap))
    hpmap = hpmap/np.max(hpmap)
    hpmap[hpmap<0] = 0.
    hp.fitsfunc.write_map(f.rmask,hpmap,overwrite=True)

# compute apodized window
if apodize:
    mask  = hp.fitsfunc.read_map(f.rmask) #load mask
    npix  = np.shape(mask)
    amask = curvedsky.utils.apodize(npix,mask,p.ascale) # apodization window
    hp.fitsfunc.write_map(f.amask,amask,overwrite=True) # save


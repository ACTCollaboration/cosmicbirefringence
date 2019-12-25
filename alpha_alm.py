import numpy as np
import healpy as hp
import curvedsky
import prjlib
import pickle
from pixell import enmap
from matplotlib.pyplot import *

p, f, r = prjlib.init()

for i in range(p.snmin,p.snmax):
    print(i,p.nside)
    amap = enmap.to_healpix(enmap.read_map(f.amap[i]),nside=p.nside) # fullsky amap, with opposite sign
    alm  = curvedsky.utils.hp_map2alm(p.nside,p.lmax,p.lmax,amap)
    pickle.dump((alm),open(f.aalm[i],"wb"),protocol=pickle.HIGHEST_PROTOCOL)



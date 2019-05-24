import numpy as np
import healpy as hp
import curvedsky
import prjlib
import cPickle as pickle
from pixell import enmap


p, f, r = prjlib.init()

for i in range(p.snmin,p.snmax):
    print(i)
    amap = enmap.to_healpix(enmap.read_map(f.amap[i]))
    alm  = curvedsky.utils.hp_map2alm(4096,p.lmax,p.lmax,amap)
    pickle.dump((alm),open(f.aalm[i],"wb"),protocol=pickle.HIGHEST_PROTOCOL)



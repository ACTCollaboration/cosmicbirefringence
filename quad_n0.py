# Reconstruction using quadratic estimator
import numpy as np
import healpy as hp
import os
import pickle
import curvedsky
import basic
import prjlib
import quad_func

p, f, r = prjlib.init()

ow = False

ocl = prjlib.loadocl(f.scl)
quad_func.quad.diagcinv(p.quad,ocl)
#quad_func.quad.n0(p.quad,f.alm,r.w4,r.lcl,overwrite=ow)
quad_func.quad.rdn0(p.quad,p.snmin,p.snmax,f.alm,r.w4,r.lcl,overwrite=ow)


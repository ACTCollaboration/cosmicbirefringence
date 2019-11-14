# calculate chi statistics
import numpy as np
import analysis as ana
import prjlib

p = prjlib.params()
f = prjlib.filename(p)
b0, b1 = 0, p.bn
#b0, b1 = 1, p.bn

# compute chi statistics
for q in p.qlist:

    # load data
    b, obs = (np.loadtxt(f.nul[q].oxbs)).T[:,b0:b1]
    sim = np.array([np.loadtxt(f.nul[q].xl[i],unpack=True)[1][b0:b1] for i in range(1,p.snmax)])

    # amplitude estimate
    print q
    st = ana.statistics(obs,sim)
    ana.statistics.x1PTE(st)
    ana.statistics.x2PTE(st)
    print np.around(st.ox1,decimals=1), np.around(st.px1,decimals=3), np.around(st.ox2,decimals=1), np.around(st.px2,decimals=3)


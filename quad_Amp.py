# calculate amplitude parameters
import numpy as np
import analysis as ana
import prjlib

p = prjlib.params()
f = prjlib.filename(p)
b0, b1 = 0, p.bn
#b0, b1 = 1, 50
p0 = prjlib.params(stype='a0p3')
f0 = prjlib.filename(p0)

# compute amplitude
for q in p.qlist:

    # load data
    b, obs = (np.loadtxt(f0.quad[q].ocbs,usecols=(0,1))).T[:,b0:b1]
    sim = np.array([np.loadtxt(f.quad[q].cl[i],unpack=True)[1][b0:b1] for i in range(1,p.snmax)])

    if p.qtype=='rot':
        # fiducial binned cl
        b, fcl = (np.loadtxt(f0.quad[q].mcbs,usecols=(0,1))).T[:,b0:b1]
        print(fcl)
        #cl  = 2*np.pi*1e-5/(np.linspace(0,p.lmax,p.lmax+1)+1.)/np.linspace(0,p.lmax,p.lmax+1)
        #fcl = basic.aps.cl2bcl(p.bn,p.lmax,cl,spc=p.binspc)
    else:
        fcl = ''


    # amplitude estimate
    print q
    st = ana.statistics(obs,sim)
    ana.statistics.get_amp(st,fcl)
    print 'obs A', st.oA, 'mean A', st.mA, 'sigma(A)', st.sA, 'S/N', 1./st.sA, 'A>oA', st.p

    # correlation coefficients
    #cov = ana.statistics.get_corrcoef(st)
    #ana.statistics.plot_corrcoef(st,b,cov,xaname='$L_1$',yaname='$L_2$',fname='cc')


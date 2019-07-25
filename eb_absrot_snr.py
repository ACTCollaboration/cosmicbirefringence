import numpy as np
import analysis as ana

#define parameters, filenames and functions
Dir = '../../data/curvedsky/cmb/aps/'

bn  = 50
spc = 'p2'
spc = ''

if bn==100 and spc=='p2': brange = [(15,100),(25,100),(35,100),(25,90),(25,95)]
if bn==100 and spc=='':   brange = [(5,100),(10,100),(15,100),(10,90),(10,95)]
if bn==50:  brange = [(2,50),(3,50),(4,50),(3,45),(3,40)]

for m in [5,6]:
    for b0, b1 in brange:
        # read data
        faps = Dir+'../alm/aps_lcmb_s14+15_deep56_ns4096_a1deg_oL1-3000_b'+str(bn)+spc
        sCX = np.array([np.loadtxt(faps+'_'+str(i).zfill(5)+'.dat',unpack=True)[m][b0:b1] for i in range(1,101)])
        sEE = np.array([np.loadtxt(faps+'_'+str(i).zfill(5)+'.dat',unpack=True)[2][b0:b1] for i in range(1,101)])
        sBB = np.array([np.loadtxt(faps+'_'+str(i).zfill(5)+'.dat',unpack=True)[3][b0:b1] for i in range(1,101)])
        sTE = np.array([np.loadtxt(faps+'_'+str(i).zfill(5)+'.dat',unpack=True)[4][b0:b1] for i in range(1,101)])
        b, mTT, mEE, mBB, mCX = np.loadtxt(Dir+'aps_sim_1d_lcmb_s14+15_deep56_ns4096_a1deg_oL1-3000_b'+str(bn)+spc+'.dat',unpack=True,usecols=(0,1,2,3,m))[:,b0:b1]
        oTT, oEE, oBB, oTE, oCX = np.loadtxt(Dir+'aps_real_1d_lcmb_s14+15_deep56_ns4096_a1deg_oL1-3000_b'+str(bn)+spc+'.dat',unpack=True,usecols=(1,2,3,4,m))[:,b0:b1]
        print('lmin,lmax=',b[0],b[-1],m)
    
        # method for estimating amplitude
        #ocl = oCX
        #ocl = mCX
        if m==5: 
            ocl = oCX/(-oTE*2*np.pi/180.)
            scl = sCX/(-sTE*2*np.pi/180.)
            #fcl = (mTT-mBB)*2*np.pi/180.
        if m==6: 
            ocl = oCX/((oEE-oBB)*2*np.pi/180.)
            scl = sCX/((sEE-sBB)*2*np.pi/180.)
            #fcl = (mEE-mBB)*2*np.pi/180.
        #scl = sCX
        fcl = 1.

        st = ana.statistics(ocl,scl)
        ana.statistics.get_amp(st,fcl)
        print 'obs A', st.oA, 'sigma(A)', st.sA, 'A>oA', st.p

        ana.statistics.x1PTE(st)
        ana.statistics.x2PTE(st)
        print st.px1, st.px2


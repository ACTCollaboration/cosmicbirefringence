import numpy as np
import prjlib
from matplotlib.pyplot import *
from matplotlib import rc
from memory_profiler import profile

@profile
def func(PSA=''):

    rc('text',usetex=True)

    p = prjlib.params(PSA)
    f = prjlib.filename(p)

    l, ckk = np.loadtxt('/global/homes/t/toshiyan/Work/DATAS/cls/fid_P13.dat',unpack=True,usecols=(0,4))
    ckk = ckk*(l+1)**2/l**2/4.

    #for q in p.qlist:
    for q in ['MV']:
        xlabel('$L$',fontsize=16)
        ylabel('$C_L^{\kappa\kappa}$',fontsize=16)
        #xscale('log')
        xlim(np.sqrt(2.),np.sqrt(2000.))
        ylim(-5e-8,3e-7)
        if p.doreal:
            bL, kk, cc = np.loadtxt(f.quad[q].ocbs,unpack=True)
            errorbar(bL,kk,label='Reconstructed ('+q+')',fmt='o')
        else:
            bL, kk, cc, kx, ik, vk, vc = np.loadtxt(f.quad[q].mcbs,unpack=True,usecols=range(7))
            bL = np.sqrt(bL)
            xticks([np.sqrt(10.),np.sqrt(100.),np.sqrt(1000.)],['10','100','1000'])
        if PSA!='':  
            errorbar(bL,kk,yerr=vk,label='Reconstructed ('+q+')',fmt='o-')
        else:
            sk = (ik/kx)**2 
            errorbar(bL,sk*kk,yerr=sk*vk,label='Reconstructed ('+q+')',fmt='o')
            #input kk
            plot(bL,ik,'--k',label='input')
            #N0
            L, n0 = np.loadtxt(f.quad[q].n0bl,unpack=True,usecols=(0,1))
            plot(L,n0,'--',label='Gaussian noise bias ('+q+')')

        legend(loc=0,numpoints=1,frameon=False)
        savefig('fig_ckk_'+p.psa+'_a'+str(p.ascale)+'deg_'+q+'.png') 
        #show()
        clf()


if __name__ == '__main__':
    #func()
    func(PSA='s14&15_comb')


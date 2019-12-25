import numpy as np
import scipy as sp
from scipy.interpolate import CubicSpline
import healpy as hp
import sys
import basic
import configparser
import quad_class
import quad_func
import analysis as ana

#* Define parameters

Tcmb = 2.72e6

def set_config(pfile='',chvals='',PSA='',stype='',doreal='',dodust='',dearot='',rlmin='',rlmax=''):
    # loading config file
    config = configparser.ConfigParser()

    if not '.ini' in sys.argv[1]:
        pfile = 'params.ini'

    if pfile != '':
        #print('ini file specified in a script file, reading '+pfile)
        config.read(pfile)
    else:
        print('reading '+sys.argv[1])
        config.read(sys.argv[1])
    
    # replacing values
    if chvals != '':
        for sec in chvals:
            for cv, val in chvals[sec]:
                print('replacing values',sec,cv,val)
                config.set(sec,cv,val)

    # additional quick replacing
    if PSA !='':     config.set('DEFAULT','PSA',PSA)
    if stype != '':  config.set('DEFAULT','stype',stype)
    if doreal != '': config.set('DEFAULT','doreal',doreal)
    if dodust != '': config.set('DEFAULT','dodust',dodust)
    if dearot != '': config.set('DEFAULT','dearot',dearot)
    if rlmin !='':   config.set('QUADREC','rlmin',rlmin)
    if rlmax !='':   config.set('QUADREC','rlmax',rlmax)

    return config


class params:

    def __init__(self,config):

        #//// get parameters ////#
        conf = config['DEFAULT']
        self.nside  = conf.getint('nside',4096) #Nside for fullsky cmb map
        self.npix   = 12*self.nside**2
        self.lmin   = conf.getint('lmin',0)
        self.lmax   = conf.getint('lmax',3000)
        self.olmin  = 1
        self.olmax  = self.lmax
        self.ol     = [self.olmin,self.olmax]
        self.bn     = conf.getint('bn',30) 
        self.binspc = conf.get('binspc','')

        self.snmin  = conf.getint('snmin',0)
        self.snmax  = conf.getint('snmax',101)
        self.stype = conf.get('stype','lcmb')
        self.PSA  = conf.get('PSA','s14&15_deep56')
        self.ascale = conf.getint('ascale',1)
        self.doreal = conf.getboolean('doreal',False)
        self.dodust = conf.getboolean('dodust',False)
        self.dearot = conf.getboolean('dearot',False)
        self.lcut   = conf.getint('lcut',100)

        # reconstruction
        self.quad  = quad_func.quad(config['QUADREC'])


        #//// Error check ////#
        #dearot
        if self.dearot:
            if not self.doreal:    sys.exit('derot abs angle: doreal should be True')
            if self.dodust:        sys.exit('derot abs angle: dodust should be False')
            if self.stype!='lcmb': sys.exit('derot abs angle: stype should be lcmb')


        #//// derived parameters ////#
        # total number of real + sim
        self.snum = self.snmax - self.snmin
        self.psa  = self.PSA.replace('&','+')

        #mtype
        if '0p' in self.stype or self.dodust:
            self.mlist = ['E','B']
        else:
            self.mlist = ['T','E','B']

        #rlz num
        if '0p' in self.stype:
            self.quad.snn0 = 50
            self.quad.snrd = 100
            self.quad.snmf = 100

        #doreal
        if self.stype in ['absrot','relrot'] or self.dodust:
            self.doreal = False

        # directory
        self.Dir = '/global/homes/t/toshiyan/Work/Ongoing/ACT/data/curvedsky/'
        
        # tag
        self.stag = self.stype+'_'+self.psa+'_ns'+str(self.nside)+'_lc'+str(self.lcut)+'_a'+str(self.ascale)+'deg'
        self.ids = [str(i).zfill(5) for i in range(500)]
        if self.doreal: self.ids[0] = 'real'
        if self.dodust: self.ids[0] = self.ids[0] + '_dust'
        if self.dearot: self.ids[0] = self.ids[0] + '_dearot'
        
        # alpha reconstruction
        quad_func.quad.fname(self.quad,self.Dir,self.ids,self.stag)



# Define class filename
class filename:

    # The code assumes the following directory structure:
    # 
    # - curvedsky/
    #     - cmb/
    #         - alm/
    #         - aps/
    #     - input/
    #         - aalm/
    #     - mask/ : mask defined in curvedsky
    # 
    # - actsim/
    #     - mask/   : mask defined in flatsky grid
    #

    def __init__(self,params):

        #//// root directories ////#
        if params.stype=='lcmb':
            d_act = '/global/homes/o/omard/cmblens/output/K_space_prepared/'
        elif params.stype=='a0p3':
            d_act = '/global/cscratch1/sd/yguan/sims/v0.6/teb_biref/'
        elif 'a0p' in params.stype:
            d_act = '/global/cscratch1/sd/yguan/sims/v0.6/teb_'+params.stype[1:].replace('p','.')+'/'
        else:
            d_act = '/global/cscratch1/sd/yguan/sims/v0.6/teb_'+params.stype+'/'
        Dir = params.Dir
        d_maps = Dir+'cmb/map2d_lcmb/'
        d_alm  = Dir+'cmb/alm/'
        d_aps  = Dir+'cmb/aps/'
        d_msk  = Dir+'mask/'
        d_yln  = '/global/cscratch1/sd/yguan/sims/v0.6/alpha/' # for input fullsky alpha map

        #//// basic tags ////#
        # map
        stag = params.stag
        xtag = params.stype+'_ns'+str(params.nside)+'_lc'+str(params.lcut)+'_a'+str(params.ascale)+'deg'

        # output multipole range
        oltag = '_ol'+str(params.olmin)+'-'+str(params.olmax)+'_b'+str(params.bn)+params.binspc

        #//// CAMB cls ////#
        # aps of best fit cosmology
        self.lcl = Dir+'input/lensed.dat'

        # window function
        d_prp = '/project/projectdirs/act/data/prepmaps/'
        self.fmask = d_prp+'/mask_'+params.PSA+'.fits'
        self.rmask = d_msk+'/'+params.psa+'_n'+str(params.nside)+'.fits'
        self.amask = d_msk+'/'+params.psa+'_n'+str(params.nside)+'_a'+str(params.ascale)+'.fits'

        #//// CMB, noise, input kappa, input alpha, dust, ... ////#
        ids  = params.ids
        ids0 = [str(i).zfill(5) for i in range(500)]
        # change 1st index
        ids0[0] = ids0[1]

        self.palm = [d_act+'/alex/fullskyPhi_alm_'+x+'.fits' for x in ids0]
        self.amap = [d_yln+'/fullskyalpha_set0_id'+str(xi)+'.fits' for xi in range(501)]
        self.aalm = [Dir+'input/aalm/aalm_'+str(x)+'.fits' for x in ids0]

        # K-space combined T/E/B maps
        self.imap = {}
        self.alm  = {}
        for mtype in params.mlist:
            self.imap[mtype] = [d_act+'preparedSimset00_Map'+x+'_'+mtype+'_'+params.PSA+'.fits' for x in ids0]
            self.alm[mtype]  = [d_alm+'/'+mtype+'_'+stag+'_'+x+'.fits' for x in ids] #lensed cmb alm
            # replace sim to real
            if params.doreal: 
                self.imap[mtype][0] = d_act+'/preparedMap_'+mtype+'_'+params.PSA+'.fits'
        
        # dust map
        self.dust = '/project/projectdirs/act/data/curvedsky/dust/thermaldust_353GHz.fits'

        # cmb aps
        self.cli = [d_aps+'/rlz/aps_'+stag+'_'+x+'.dat' for x in ids]
        self.scl = d_aps+'aps_sim_1d_'+stag+'.dat'
        self.scb = d_aps+'aps_sim_1d_'+stag+oltag+'.dat'
        self.ocl = d_aps+'aps_'+ids[0]+'_1d_'+stag+'.dat'
        self.ocb = d_aps+'aps_'+ids[0]+'_1d_'+stag+oltag+'.dat'


class recfunc:

    def __init__(self,params,filename):

        #multipole
        self.el = np.linspace(0,params.lmax,params.lmax+1)
        self.ol = np.linspace(0,params.olmax,params.olmax+1)
        self.kl = self.el*(self.el+1)*.5

        #binned multipole
        self.bp, self.bc = basic.aps.binning(params.bn,params.ol,spc=params.binspc)

        #theoretical cl
        self.lcl = basic.aps.read_cambcls(filename.lcl,params.lmin,params.lmax,4,bb=True)/Tcmb**2



#////////// Initial setup //////////#
def params_init(pfile='',chvals='',PSA='',stype='',doreal='',dodust='',dearot='',rlmin='',rlmax=''):
    config = set_config(pfile,chvals,PSA,stype,doreal,dodust,dearot,rlmin,rlmax)
    p = params(config)
    return p


def filename_init(pfile='',chvals='',PSA='',stype='',doreal='',dodust='',dearot='',rlmin='',rlmax=''):
    p = params_init(pfile,chvals,PSA,stype,doreal,dodust,dearot,rlmin,rlmax)
    f = filename(p)
    return p, f


def init(pfile='',chvals='',PSA='',stype='',doreal='',rlmin='',dodust='',dearot='',rlmax='',loadw=True):
    p, f = filename_init(pfile,chvals,PSA,stype,doreal,dodust,dearot,rlmin,rlmax)
    r = recfunc(p,f)
    if loadw:
        r.w, r.w2, r.w4, tw = window(f)
    return p, f, r


def window(filename):

    wsf = hp.fitsfunc.read_map(filename.rmask)
    #if boass and params.stype=='arot': wsf *= 6e-05

    print(filename.amask)
    w = hp.fitsfunc.read_map(filename.amask)

    totw = wsf*w
    w2 = np.average(totw**2)
    w4 = np.average(totw**4)
    print(w2,w4)

    return w, w2, w4, totw


def loadocl(filename):
    print('loading TT/EE/BB/TE from pre-computed spectrum:',filename)
    return np.loadtxt(filename,unpack=True,usecols=(1,2,3,4))


#////////// Multipole binning //////////

class multipole_binning:

    def __init__(self,n,spc='p2',lmin=1,lmax=2048):
        self.n = n
        self.spc = spc
        self.lmin = lmin
        self.lmax = lmax
        self.bp, self.bc = basic.aps.binning(n,[lmin,lmax],spc=spc)


def binning_all(bn,bn1=10,lmin=10,Lsp=2048):
    if Lsp==2048:
        mb0 = multipole_binning(bn,spc='p2',lmin=lmin,lmax=2048)
        mb1 = None
        mb  = mb0
    else:
        mb  = multipole_binning(bn,spc='p2',lmin=lmin,lmax=Lsp)
        mb0 = multipole_binning(bn,spc='p2',lmin=lmin,lmax=Lsp)
        mb1 = multipole_binning(bn1,spc='',lmin=Lsp+1,lmax=2048)
        mb.n = mb0.n + mb1.n
        mb.bp = np.concatenate((mb0.bp,mb1.bp[1:]))
        mb.bc = np.concatenate((mb0.bc,mb1.bc))
    return mb, mb0, mb1


def binning(cl,b0,b1=None):

    if b1 is None:
        return binning1(cl,b0)
    else:
        return binning2(cl,b0,b1)


def binning1(cl,b):

    if b.lmax > np.shape(cl)[-1] - 1:
        sys.exit('size of b.lmax is wrong')

    if np.ndim(cl) == 1:
        cb = basic.aps.cl2bcl(b.n,b.lmax,cl[:b.lmax+1],lmin=b.lmin,spc=b.spc)

    if np.ndim(cl) == 2:
        snmax = np.shape(cl)[0]
        cb = np.array([basic.aps.cl2bcl(b.n,b.lmax,cl[i,:b.lmax+1],lmin=b.lmin,spc=b.spc) for i in range(snmax)])

    if np.ndim(cl) == 3:
        snmax = np.shape(cl)[0]
        clnum = np.shape(cl)[1]
        cb = np.array([[basic.aps.cl2bcl(b.n,b.lmax,cl[i,c,:b.lmax+1],lmin=b.lmin,spc=b.spc) for c in range(clnum)] for i in range(snmax)])

    return cb


def binning2(cl,b0,b1):

    if b1.lmin != b0.lmax+1:
        sys.exit('wrong split')
    if b1.lmax > np.shape(cl)[-1]-1:
        sys.exit('wrong lmax')

    if np.ndim(cl) == 1:
        cb0 = basic.aps.cl2bcl(b0.n,b0.lmax,cl[:b0.lmax+1],spc=b0.spc,lmin=b0.lmin)
        cb1 = basic.aps.cl2bcl(b1.n,b1.lmax,cl[:b1.lmax+1],spc=b1.spc,lmin=b1.lmin)
        return np.concatenate((cb0,cb1))

    if np.ndim(cl) == 2:
        cb0 = binning(cl,b0)
        cb1 = binning(cl,b1)
        return np.concatenate((cb0,cb1),axis=1)


#////////// Binned Spectrum //////////

def binned_claa(Lmax,mb0,mb1=None):
    L = np.linspace(0,Lmax,Lmax+1)
    fcl = 1e-4*2*np.pi/(L**2+L+1e-30)
    return binning(fcl,mb0,mb1)


def binned_cl(fcl,mb0,mb1=None,cn=1):
    scl = np.loadtxt(fcl,unpack=True)[cn]
    return binning(scl,mb0,mb1)


def binned_cl_rlz(fcl,sn0,sn1,mb0,mb1=None,cn=1):
    scl = np.array([np.loadtxt(fcl[i],unpack=True)[cn] for i in range(sn0+1,sn1+1)])
    return binning(scl,mb0,mb1)


#////////// Absrot estimate //////////

def est_angle(oCX,sCX,oCY,sCY,fcl=1.):
    # method for estimating amplitude
    ocl = oCX/(oCY*2*np.pi/180.)
    scl = sCX/(sCY*2*np.pi/180.)
    st = ana.statistics(ocl,scl)
    ana.statistics.get_amp(st,fcl)
    print('obs A [deg]', np.around(st.oA,decimals=3), 'sigma(A) [deg]', np.around(st.sA,decimals=3), 'A>oA', st.p)
    ana.statistics.x1PTE(st)
    ana.statistics.x2PTE(st)
    print(np.around(st.px1,decimals=3), np.around(st.px2,decimals=3))


def est_angles(patch,spec='EB',bn=50,spc='',lmin=200,lmax=2048,doreal='True',dearot='False',sn=100):
    if spec == 'TB': m=5
    if spec == 'EB': m=6
    __, f = filename_init(doreal=doreal,PSA='s14&15_'+patch,dearot=dearot)
    mb  = multipole_binning(bn,spc=spc,lmin=lmin,lmax=lmax)
    scl = np.array([np.loadtxt(f.cli[i],unpack=True,usecols=(2,3,4,m)) for i in range(1,sn+1)])
    scb = binning(scl,mb)
    ocl = np.loadtxt(f.ocl,unpack=True,usecols=(2,3,4,m))
    ocb = binning(ocl,mb)
    if spec=='TB':
        est_angle(ocb[3,:],scb[:,3,:],ocb[2,:],scb[:,2,:])
    if spec=='EB':
        est_angle(ocb[3,:],scb[:,3,:],ocb[0,:]-ocb[1,:],scb[:,0,:]-scb[:,1,:])


#////////// HL Likelihood //////////

def lnLLH(rx,fcl,icov,bi=None):
    # rx = ocb/scb
    # icov is the covariance of ocb
    bn, bn = np.shape(icov)
    if bi is None:
        gx = np.sign(rx-1.)*np.sqrt(2.*(rx-np.log(rx)-1.))
    else:
        gx = np.zeros(bn)
        gx[bi] = np.sign(rx[bi]-1.)*np.sqrt(2.*(rx[bi]-np.log(rx[bi])-1.))
    return -0.5*np.dot(gx*fcl,np.dot(icov,gx*fcl))


def lnLLHs(rx,fcl,icov,bi=None):
    # rx = ocb/scb
    # icov is the covariance of ocb
    bn, bn = np.shape(icov)
    gx = np.sign(rx-1.)*np.sqrt(2.*(rx-np.log(rx)-1.))
    return -0.5*gx*fcl[bi]*icov[bi,bi]*gx*fcl[bi]

def posterior(A,ocb,mcb,macb,icov,c0,c1):
    Lh = np.zeros(len(A))
    for i, a in enumerate(A):
        scb = c0*(mcb+(a/.1)*(macb-mcb))
        Lh[i] = np.exp(lnLLH(ocb/scb,mcb*c1,icov))
    return Lh

    
def Lgauss(bi,Ab,Afb,icov):
    bn, bn = np.shape(icov)
    dA = np.zeros(bn)
    dA[bi] = Ab-Afb
    return -0.5*np.dot(dA,np.dot(icov,dA))

#////////// Direct Likelihood //////////

def fit_skewnorm(xA,dat):
    mA, vA, sA = sp.stats.skewnorm.fit(dat)
    #xA = np.arange(min(0,np.min(dat)),np.max(dat),0.001)
    return sp.stats.skewnorm.pdf(xA,mA,vA,sA)


def like(dat,odat=0.0,ddat=0.01):
    return (np.abs(dat-odat)<=ddat).sum()/np.float(len(dat)) / (2.*ddat)


def calc_like(Ainp,dat,odat=None,ddat=0.01):
    L = np.zeros(len(Ainp))
    if odat is None: odat = np.zeros(len(Ainp))
    for i, Ai in enumerate(Ainp):
        L[i] = like(dat[i,:],odat=odat[i],ddat=ddat)
    return L


def calc_CDF(L,As,interp='cubic'):
    if interp=='cubic':
        dx = 0.001
        x = np.arange(0.,As[-1],dx)
        f = CubicSpline(As,L,bc_type='natural')
        Like = f(x)
    else:
        dx = As[1:]-As[:-1]
        x = (As[1:]+As[:-1])*.5
        Like = (L[1:]+L[:-1])*.5
    Ltot = np.sum(dx*Like)
    PDF = Like/Ltot
    CDF = np.cumsum(PDF*dx)
    return x, PDF, CDF


def lintrans(Ab,As):
    mAb = np.mean(Ab,axis=1)
    c0 = mAb[0]
    c1 = (mAb[2]-mAb[1])/(As[2]-As[1])
    return c0, c1


def quadstats(patch,As,sn,mb0,mb1=None,rlmin='200',wi='lcmb',doreal='True'):

    fcb = binned_claa(2048,mb0,mb1)

    ps  = params_init(stype='lcmb',PSA='s14&15_'+patch,rlmin=rlmin)
    scb = binned_cl_rlz(ps.quad.f['EB'].cl,0,sn,mb0,mb1)
    n0s = np.loadtxt(ps.quad.f['EB'].n0bl,unpack=True)[1]

    if wi=='LCMB':
        scb1 = binned_cl_rlz(ps.quad.f['EB'].cl,100,200,mb0,mb1)
        wi, __, __ = ana.opt_weight(scb1/fcb)
    else:
        pw = params_init(stype=wi,PSA='s14&15_'+patch,rlmin=rlmin)
        acb = binned_cl_rlz(pw.quad.f['EB'].cl,0,sn,mb0,mb1)
        wi, __, __ = ana.opt_weight(acb/fcb,diag=True)

    if doreal:
        pr  = params_init(stype='lcmb',PSA='s14&15_'+patch,rlmin=rlmin,doreal='True',dearot='True')
        ocb = binned_cl(pr.quad.f['EB'].ocls,mb0,mb1)
    else:
        ocb = np.mean(scb,axis=0)

    Ab = np.zeros((len(As),sn))
    oA = np.zeros(len(As))
    for i, A in enumerate(As):
        if A==0.:
            Ab[i,:] = np.sum(wi*scb/fcb,axis=1)
            #oA[i] = np.mean(np.sum(wi*scb/fcb,axis=1))
            oA[i] = np.sum(wi*ocb/fcb)
        else:
            pa = params_init(stype='a'+str(A).replace('.','p'),PSA='s14&15_'+patch,rlmin=rlmin)
            acb = binned_cl_rlz(pa.quad.f['EB'].cl,0,sn,mb0,mb1)
            n0a = np.loadtxt(pa.quad.f['EB'].n0bl,unpack=True)[1]
            Ocb = ocb + binning(n0s-n0a,mb0,mb1)
            oA[i] = np.sum(wi*Ocb/fcb)
            #dn0 = np.loadtxt(pa.quad.f['EB'].n0bl,unpack=True)[1] - n0
            #acb = prjlib.binning(acl+dn0[None,:],bn,spc,lmin=lmin)[:,b0:b1]
            Ab[i,:] = np.sum(wi*acb/fcb,axis=1)

    c0, c1 = lintrans(Ab,As)
    estAb = (Ab-c0)/c1
    estoA = (oA-c0)/c1
    #plot(As,np.mean(Ab,axis=1),colors[bi]+'-')

    return estAb, estoA



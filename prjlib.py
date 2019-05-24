import numpy as np
import healpy as hp
import sys
import basic
import curvedsky
import configparser
from pixell import enmap

#* Define parameters

Tcmb = 2.72e6

class params:

    def __init__(self,PSA=''):

        #//// load config file ////#
        config = configparser.ConfigParser()
        print('reading '+sys.argv[1])
        config.read(sys.argv[1])
        conf = config['DEFAULT']

        #//// get parameters ////#
        self.nside  = conf.getint('nside',4096) #Nside for fullsky cmb map
        self.npix   = 12*self.nside**2
        self.lmin   = conf.getint('lmin',0)
        self.lmax   = conf.getint('lmax',3000)

        self.olmin  = conf.getint('olmin',2)
        self.olmax  = conf.getint('olmax',3000)
        self.bn     = conf.getint('bn',30) 
        self.binspc = conf.get('binspc','p2')

        self.snmin  = conf.getint('snmin',0)
        self.snmax  = conf.getint('snmax',101)
        self.stype  = conf.get('stype','lcmb')
        if PSA!='':
            self.PSA  = PSA
        else:
            self.PSA  = conf.get('PSA','s14&15_deep56')
        self.ascale = conf.getint('ascale',1)
        #self.doreal = True
        self.doreal = conf.getboolean('doreal',False)

        # reconstruction
        self.qtype  = conf.get('qtype','lens')
        self.nsidet = conf.getint('nsidet',2048)
        self.rlmin  = conf.getint('rlmin',500)
        self.rlmax  = conf.getint('rlmax',3000)
        self.snn0   = conf.getint('snn0',50)
        self.snrd   = conf.getint('snrd',100)
        self.snmf   = conf.getint('snmf',100)

        #//// derived parameters ////#
        # total number of real + sim
        self.snum = self.snmax - self.snmin
        self.psa  = self.PSA.replace('&','+')
        self.oL   = [self.olmin,self.olmax]

        #definition of T+P
        self.qDO = [True,True,True,False,False,False]
        self.qMV = ['TT','TE','EE']

        #definition of qest
        self.qlist = ['TT','TE','EE','TB','EB','MV']
        #self.qlist = ['MV']
        if self.qtype=='rot':
            self.qlist = ['EB']


# * Define quad estimator names
class quadest:

    def __init__(self,qtype,qest,root,stag,ltag,otag,ids):

        # qtype is the type of mode coupling, such as lens, rot, etc
        qalm = root + qtype + '/alm/'
        qrdn = root + qtype + '/rdn0/'
        qmlm = root + qtype + '/mean/'
        qaps = root + qtype + '/aps/'

        # normalization and tau transfer function
        self.al   = qaps+'Al_'+qest+'_'+stag+ltag+'.dat'
        self.wl   = qaps+'Wl_'+qest+'_'+stag+ltag+'.dat'

        # N0/N1 bias
        self.n0bl = qaps+'n0_'+qest+'_'+stag+ltag+'.dat'
        self.n1bs = qaps+'n1_'+qest+'_'+stag+ltag+'.dat'

        # mean field
        self.ml   = [qmlm+'ml_'+qest+'_'+stag+ltag+'_'+x+'.dat' for x in ids]
        self.mfb  = [qmlm+'mfb_'+qest+'_'+stag+ltag+'_'+x+'.fits' for x in ids]

        # reconstructed spectra
        self.mcls = qaps+'cl_'+qest+'_'+stag+ltag+'.dat'
        self.mcbs = qaps+'cl_'+qest+'_'+stag+ltag+otag+'.dat'
        self.ocls = qaps+'cl_obs_'+qest+'_'+stag+ltag+'.dat'
        self.ocbs = qaps+'cl_obs_'+qest+'_'+stag+ltag+otag+'.dat'
        self.rcls = qaps+'cl_real_'+qest+'_'+stag+ltag+'.dat'
        self.rcbs = qaps+'cl_real_'+qest+'_'+stag+ltag+otag+'.dat'

        # reconstructed alm/map and RDN0
        self.alm  = [qalm+'alm_'+qest+'_'+stag+ltag+'_'+x+'.fits' for x in ids]
        self.cl   = [qalm+'cl_'+qest+'_'+stag+ltag+'_'+x+'.dat' for x in ids]
        self.rdn0 = [qrdn+'rdn0_'+qest+'_'+stag+ltag+'_'+x+'.dat' for x in ids]


# * Define class filename
class filename:

    # The code assumes the following directory structure:
    # 
    # - curvedsky/
    #     - cmb/
    #         - map/
    #         - alm/
    #         - aps/
    # - input/
    # 
    # - actsim/
    #     - alpha/  : input cosmic birefringence map/alm
    #     - mask/   : mask defined in flatsky grid
    #
    # - mask/ : mask defined in curvedsky
    #     
    #


    def __init__(self,params):

        #//// root directories ////#
        Dir    = '/global/cscratch1/sd/toshiyan/ACT/curvedsky/'
        # input cl
        d_inp  = Dir+'../input/'
        # cmb, kappa
        d_act  = Dir+'../actsim/'
        d_map  = Dir+'cmb/map/'
        d_alm  = Dir+'cmb/alm/'
        d_aps  = Dir+'cmb/aps/'
        # params mask
        d_msk  = Dir+'mask/'

        #//// basic tags ////#

        # map
        stag = params.stype+'_'+params.psa+'_ns'+str(params.nside)+'_a'+str(params.ascale)+'deg'

        # output multipole range
        otag = '_oL'+str(params.olmin)+'-'+str(params.olmax)+'_b'+str(params.bn)

        # kappa reconstruction multipole
        ltag = '_l'+str(params.rlmin)+'-'+str(params.rlmax)

        #//// index ////#
        ids = [str(i).zfill(5) for i in range(500)]
        # change 1st index
        if params.doreal: ids[0] = 'real'

        #//// CAMB cls ////#
        # aps of best fit cosmology
        self.lcl = d_inp+'lensed.dat'

        # window function
        #self.Fmask = d_act+'/mask/mask_'+params.PSA+'.fits'
        #self.Fmask = d_act+'/mask/mask_'+params.PSA+'_arot.fits'
        self.fmask = d_act+'/mask/'+params.PSA+'_'+params.stype+'.fits'
        #self.Rmask = d_msk+'/mask_'+params.psa+'.fits'
        self.rmask = d_msk+'/'+params.psa+'_'+params.stype+'.fits'
        self.amask = d_msk+'/'+params.psa+'_'+params.stype+'_a'+str(params.ascale)+'.fits'

        #//// CMB, noise, kappa, alpha, ... ////#
        self.palm = [d_act+'/alex/fullskyPhi_alm_'+x+'.fits' for x in ids]
        self.amap = [d_act+'/alpha/fullskyalpha_set0_id'+str(xi)+'.fits' for xi in range(501)]
        self.aalm = [d_act+'/alpha/aalm_'+str(x)+'.fits' for x in ids]

        self.imap = {}
        self.omap = {}
        self.alm  = {}
        for mtype in ['T','E','B']:
            if   params.stype == 'lcmb': 
                self.imap[mtype] = [d_act+'/K_space_prepared/preparedSimset00_Map'+x+'_'+mtype+'_'+params.PSA+'.fits' for x in ids]
            elif params.stype == 'arot':
                self.imap[mtype] = [d_act+'/alpha/preparedSimset00_Map'+x+'_'+mtype+'_'+params.PSA+'.fits' for x in ids]
            else:
                print('no valid cmb maps')
            self.omap[mtype] = [d_map+'/'+mtype+'_'+params.stype+'_'+params.psa+'_'+x+'.fits' for x in ids] #lensed cmb map
            self.alm[mtype]  = [d_alm+'/'+mtype+'_'+stag+'_'+x+'.fits' for x in ids] #lensed cmb alm
            # replace sim to real
            if params.doreal: self.imap[mtype][0] = d_act+'/cmb/preparedMap_'+mtype+'_'+params.PSA+'.fits'

        # cmb aps
        self.scl = d_aps+'aps_sim_1d_'+stag+'.dat'
        self.scb = d_aps+'aps_sim_1d_'+stag+otag+'.dat'
        self.ocl = d_aps+'aps_obs_1d_'+stag+'.dat'
        self.ocb = d_aps+'aps_obs_1d_'+stag+otag+'.dat'
        self.rcl = d_aps+'aps_real_1d_'+stag+'.dat'
        self.rcb = d_aps+'aps_real_1d_'+stag+otag+'.dat'

        self.quad = {}
        for q in params.qlist:
            self.quad[q] = quadest(params.qtype,q,Dir,stag,ltag,otag,ids)


class recfunc:

    def __init__(self,params,filename):

        #multipole
        self.eL = np.linspace(0,params.lmax,params.lmax+1)
        self.oL = np.linspace(0,params.olmax,params.olmax+1)
        self.kL = self.eL*(self.eL+1)*.5

        #binned multipole
        self.bp, self.bc = basic.aps.binning(params.bn,params.oL,spc=params.binspc)

        #theoretical cl
        self.lcl = basic.aps.read_cambcls(filename.lcl,params.lmin,params.lmax,4,bb=True)/Tcmb**2


#initial setup
def init(PSA=''):
    p = params(PSA)
    f = filename(p)
    r = recfunc(p,f)
    window(p,f,r)
    return p, f, r


def window(params,filename,r):
    #window
    if 'boss' in params.psa:
        wsf = hp.fitsfunc.read_map(filename.rmask)
        #wrf = hp.fitsfunc.read_map(filename.Rmask)*6.02e-5
        if params.stype=='arot': wsf *= 6e-05
    if 'deep56' in params.psa:
        wsf = hp.fitsfunc.read_map(filename.rmask)
        if params.stype=='lcmb': wsf *= 1.5
        if params.stype=='arot': wsf *= 0.000141
        #wrf = hp.fitsfunc.read_map(filename.Rmask)*0.000141

    r.w = hp.fitsfunc.read_map(filename.amask)
    #r.w2 = np.average(r.w**2)
    #r.w4 = np.average(r.w**4)

    totw = wsf*r.w
    #totW = wrf*r.w
    r.w2 = np.average(totw**2)
    r.w4 = np.average(totw**4)
    #r.W2 = np.average(totW**2)
    #r.W4 = np.average(totW**4)
    #r.Wx = np.average(totW**2*totw**2)
    #print(r.w2,r.w4,r.W2,r.W4,r.Wx)
    print(r.w2,r.w4)


def make_qrec_filter(params,filename,r):

    ocl  = np.loadtxt(filename.scl,unpack=True,usecols=(1,2,3,4))
    r.oc = ocl

    r.Fl = {}
    for mtype in ['T','E','B']:
        r.Fl[mtype] = np.zeros((params.lmax+1,params.lmax+1))

    for l in range(params.rlmin,params.rlmax+1):
        r.Fl['T'][l,0:l+1] = 1./ocl[0,l]
        r.Fl['E'][l,0:l+1] = 1./ocl[1,l]
        r.Fl['B'][l,0:l+1] = 1./ocl[2,l]




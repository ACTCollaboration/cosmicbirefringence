import numpy as np
import healpy as hp
import sys
import basic
import configparser
import quad_class
import quad_func

#* Define parameters

Tcmb = 2.72e6

class params:

    def __init__(self,PSA='',stype=''):

        #//// load config file ////#
        config = configparser.ConfigParser()
        print('reading '+sys.argv[1])
        config.read(sys.argv[1])

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
        if stype != '':
            self.stype = stype
        else:
            self.stype = conf.get('stype','lcmb')
        if PSA!='':
            self.PSA  = PSA
        else:
            self.PSA  = conf.get('PSA','s14&15_deep56')
        self.ascale = conf.getint('ascale',1)
        self.doreal = conf.getboolean('doreal',False)
        self.chreal = conf.get('chreal','')
        self.lcut   = conf.getint('lcut',100)

        # reconstruction
        self.quad  = quad_func.quad(config['QUADREC'])

        #//// derived parameters ////#
        # total number of real + sim
        self.snum = self.snmax - self.snmin
        self.psa  = self.PSA.replace('&','+')

        #mtype
        if self.stype=='a1p0' or self.stype=='a0p3':
            self.mlist = ['E','B']
        else:
            self.mlist = ['T','E','B']


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
        Dir    = '/global/homes/t/toshiyan/Work/Ongoing/ACT/data/curvedsky/'
        # input cl
        d_inp  = Dir+'input/'
        # cmb, kappa
        d_ACT  = '/project/projectdirs/act/data/prepmaps/'
        #d_act  = '/global/cscratch1/sd/yguan/cmblens/output/K_space_prepared/'
        d_act  = '/global/homes/o/omard/cmblens/output/K_space_prepared/'
        d_maps = Dir+'cmb/map2d_lcmb/'
        d_alm  = Dir+'cmb/alm/'
        d_aps  = Dir+'cmb/aps/'
        # params mask
        d_msk  = Dir+'mask/'

        #//// basic tags ////#

        # map
        stag = params.stype+'_'+params.psa+'_ns'+str(params.nside)+'_lc'+str(params.lcut)+'_a'+str(params.ascale)+'deg'
        xtag = params.stype+'_ns'+str(params.nside)+'_lc'+str(params.lcut)+'_a'+str(params.ascale)+'deg'

        # output multipole range
        oltag = '_ol'+str(params.olmin)+'-'+str(params.olmax)+'_b'+str(params.bn)+params.binspc

        #//// index ////#
        ids  = [str(i).zfill(5) for i in range(500)]
        ids0 = [str(i).zfill(5) for i in range(500)]
        # change 1st index
        if params.doreal:      ids[0]  = 'real'
        if params.chreal!='':  ids[0] += '_'+params.chreal
        #if params.absrot!='':  ids = [str(i).zfill(5)+'_absrot' for i in range(500)]
        #if params.relrot!='':  ids = [str(i).zfill(5)+'_relrot' for i in range(500)]

        #//// CAMB cls ////#
        # aps of best fit cosmology
        self.lcl = d_inp+'lensed.dat'

        # window function
        self.fmask = d_ACT+'/mask_'+params.PSA+'.fits'
        self.rmask = d_msk+'/'+params.psa+'_n'+str(params.nside)+'.fits'
        self.amask = d_msk+'/'+params.psa+'_n'+str(params.nside)+'_a'+str(params.ascale)+'.fits'

        #//// CMB, noise, kappa, alpha, ... ////#
        self.palm = [d_act+'/alex/fullskyPhi_alm_'+x+'.fits' for x in ids0]
        self.amap = [d_act+'/alpha/fullskyalpha_set0_id'+str(xi)+'.fits' for xi in range(501)]
        self.aalm = [d_inp+'/aalm/aalm_'+str(x)+'.fits' for x in ids0]

        self.imap = {}
        self.alm  = {}
        for mtype in ['T','E','B']:
            if   params.stype == 'lcmb':
                #self.imap[mtype] = [Dir+'cmb/map2d_lcmb/preparedSimset00_Map'+x+'_'+mtype+'_'+params.PSA+'.fits' for x in ids]
                self.imap[mtype] = [d_act+'preparedSimset00_Map'+x+'_'+mtype+'_'+params.PSA+'.fits' for x in ids]
            elif params.stype == 'a1p0':
                self.imap[mtype] = [Dir+'cmb/map2d_a1p0/preparedSimset00_Map'+x+'_'+mtype+'_'+params.PSA+'.fits' for x in ids]
            elif params.stype == 'a0p3':
                self.imap[mtype] = [Dir+'cmb/map2d_a0p3/preparedSimset00_Map'+x+'_'+mtype+'_'+params.PSA+'.fits' for x in ids]
            elif params.stype == 'f150':
                self.imap[mtype] = [d_maps+'/preparedSimset00_Map'+x+'_'+mtype+'_'+params.PSA+'.fits' for x in ids]
            else:
                print('no valid cmb maps')
            # alm
            self.alm[mtype]  = [d_alm+'/'+mtype+'_'+stag+'_'+x+'.fits' for x in ids] #lensed cmb alm
            # replace sim to real
            if params.doreal: 
                self.imap[mtype][0] = d_act+'/preparedMap_'+mtype+'_'+params.PSA+'.fits'

        # cmb aps
        self.cbi = [d_alm+'/aps_'+stag+oltag+'_'+x+'.dat' for x in ids]
        self.scl = d_aps+'aps_sim_1d_'+stag+'.dat'
        self.scb = d_aps+'aps_sim_1d_'+stag+oltag+'.dat'
        self.ocl = d_aps+'aps_'+ids[0]+'_1d_'+stag+'.dat'
        self.ocb = d_aps+'aps_'+ids[0]+'_1d_'+stag+oltag+'.dat'

        # alpha alm, auto
        self.nul = {}
        quad_func.quad.fname(params.quad,Dir,ids,stag)
        for q in params.quad.qlist:
            self.nul[q]  = nullspec(params.quad.qtype,q,Dir,'_'+xtag+params.quad.ltag,params.quad.otag,ids)


class nullspec:

    def __init__(self,qtype,q,Dir,ltag,otag,ids):

        qaps = Dir + qtype + '/aps/'
        self.mxls = qaps+'xl_'+q+ltag+'.dat'
        self.mxbs = qaps+'xl_'+q+ltag+otag+'.dat'
        self.oxls = qaps+'xl_'+ids[0]+'_'+q+ltag+'.dat'
        self.oxbs = qaps+'xl_'+ids[0]+'_'+q+ltag+otag+'.dat'
        self.xl   = [qaps+'rlz/xl_'+q+ltag+'_'+x+'.dat' for x in ids]


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


#initial setup
def init(PSA='',stype='',loadw=True):
    p = params(PSA,stype)
    f = filename(p)
    r = recfunc(p,f)
    if loadw:
        r.w, r.w2, r.w4, tw = window(f)
    return p, f, r


def filename_init(PSA='',stype=''):
    p = params(PSA,stype)
    f = filename(p)
    return p, f


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



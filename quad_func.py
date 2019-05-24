
import numpy as np
import sys
from memory_profiler import profile
import curvedsky

if sys.version_info[:3] > (3,0):
  import pickle
elif sys.version_info[:3] > (2,5,2):
  import cPickle as pickle


# compute normalization
@profile
def al(p,f,r):
    '''
    Return normalization of the quadratic estimators
    '''

    for q in p.qlist:

        if q=='EB':
            Ag = curvedsky.norm_rot.qeb(p.lmax,p.rlmin,p.rlmax,r.lcl[1,:],r.oc[1,:],r.oc[2,:])

        # save
        np.savetxt(f.quad[q].al,np.array((r.eL,Ag)).T)  


def loadnorm(p,files):
    Ag = {}
    for q in p.qlist:
        Ag[q]  = np.loadtxt(files[q].al,unpack=True)[1]

    return Ag



@profile
def qrec(p,falm,fquad,r):
    '''
    Return quadratic estimators
    '''

    # load normalization and weights
    Ag = loadnorm(p,fquad)

    # loop for realizations
    for i in range(p.snmin,p.snmax):
        print(i)

        gmv = 0.

        for q in p.qlist:

            if 'E' in q:  Ealm = r.Fl['E'] * pickle.load(open(falm['E'][i],"rb"))
            if 'B' in q:  Balm = r.Fl['B'] * pickle.load(open(falm['B'][i],"rb"))

            if q=='EB':  glm = curvedsky.rec_rot.qeb(p.lmax,p.rlmin,p.rlmax,r.lcl[1,:],Ealm,Balm,nside=p.nsidet)

            glm *= Ag[q][:,None]
            pickle.dump((glm),open(fquad[q].alm[i],"wb"),protocol=pickle.HIGHEST_PROTOCOL)


@profile
def n0(p,falm,fquad,r):
    '''
    The N0 bias calculation
    '''

    # load normalization and weights
    Ag = loadnorm(p,fquad)

    # power spectrum
    cl ={}
    for q in p.qlist:
        cl[q] = np.zeros((1,p.lmax+1))

    # loop for realizations
    for i in range(p.snn0):
        print (2*i+1, 2*i+2)

        gmv = 0.

        for q in p.qlist:

            q1, q2 = q[0], q[1]
            print(q1,q2)
            alm1 = r.Fl[q1] * pickle.load(open(falm[q1][2*i+1],"rb"))
            alm2 = r.Fl[q1] * pickle.load(open(falm[q1][2*i+2],"rb"))
            if q1 == q2:
                blm1 = alm1
                blm2 = alm2
            else:
                blm1 = r.Fl[q2] * pickle.load(open(falm[q2][2*i+1],"rb"))
                blm2 = r.Fl[q2] * pickle.load(open(falm[q2][2*i+2],"rb"))
            glm = qXY(q,p,r.lcl,alm1,alm2,blm1,blm2)
            glm *= Ag[q][:,None]

            cl[q][0,:] += curvedsky.utils.alm2cl(p.lmax,glm)/(2*r.w4*p.snn0)


    for q in p.qlist:

        if p.snn0>0:
            print ('save N0 data')
            np.savetxt(fquad[q].n0bl,np.concatenate((r.eL[None,:],cl[q])).T)


@profile
def rdn0(p,falm,fquad,r):
    '''
    The 1st set of the RDN0 bias calculation
    '''

    # load normalization and weights
    Ag = loadnorm(p,fquad)

    # load N0
    N0 = {}
    for q in p.qlist:
        N0[q] = np.loadtxt(fquad[q].n0bl,unpack=True,usecols=(1,2))

    # compute RDN0
    for i in range(p.snmin,p.snmax):
        print(i)

        # power spectrum
        cl = {}
        for q in p.qlist:
            cl[q] = np.zeros((1,p.lmax+1))

        # load alm
        almr = {}
        for cmb in ['T','E','B']:
            almr[cmb] = r.Fl[cmb]*pickle.load(open(falm[cmb][i],"rb"))

        # loop for I
        for I in range(1,p.snrd+1):

            gmv = 0.

            # load alm
            alms = {}
            for cmb in ['T','E','B']:
                alms[cmb] = r.Fl[cmb]*pickle.load(open(falm[cmb][I],"rb"))

            for q in p.qlist:

                q1, q2 = q[0], q[1]

            if I==i: continue
            print(I)

            glm = qXY(q,p,r.lcl,almr[q1],alms[q1],almr[q2],alms[q2])
            glm *= Ag[q][:,None]

            cl[q][0,:] += curvedsky.utils.alm2cl(p.lmax,glm)


    if p.snrd>0:
        if i==0:  sn = p.snrd
        if i!=0:  sn = p.snrd-1
        for q in p.qlist:
            cl[q] = cl[q]/(r.w4*sn) - N0[q]
            print ('save RDN0')
            np.savetxt(fquad[q].rdn0[i],np.concatenate((r.eL[None,:],cl[q])).T)



def qXY(q,p,lcl,alm1,alm2,blm1,blm2):

    if q=='EB':
        rlm1 = curvedsky.rec_rot.qeb(p.lmax,p.rlmin,p.rlmax,lcl[1,:],alm1,blm2,nside=p.nsidet)
        rlm2 = curvedsky.rec_rot.qeb(p.lmax,p.rlmin,p.rlmax,lcl[1,:],alm2,blm1,nside=p.nsidet)

    return rlm1+rlm2


@profile
def mean(p,fquad,r):

  for q in p.qlist:

    print('load data first',q)
    glm = np.zeros((p.snmf,p.lmax+1,p.lmax+1),dtype=np.complex)

    for I in range(1,p.snmf+1):
      print(I)
      glm[I-1,:,:] = pickle.load(open(fquad[q].alm[I],"rb"))

    print('compute mean field')

    for i in range(p.snmin,p.snmax):
      print(i)
      mfg = np.average(glm,axis=0)
      if i!=0: 
        mfg -= glm[i-1,:,:]/p.snmf
        mfg *= p.snmf/(p.snmf-1.)

      print('save to file')
      pickle.dump((mfg),open(fquad[q].mfb[i],"wb"),protocol=pickle.HIGHEST_PROTOCOL)

      # compute mf cls
      print('cl')
      cl = np.zeros((1,p.lmax+1))
      cl[0,:] = curvedsky.utils.alm2cl(p.lmax,mfg)/r.w4
      np.savetxt(fquad[q].ml[i],np.concatenate((r.eL[None,:],cl)).T)



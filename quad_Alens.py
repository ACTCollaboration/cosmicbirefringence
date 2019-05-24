# calculate amplitude parameters
import numpy as np
import analysis as ana
import prjlib

p = prjlib.params()
f = prjlib.filename(p)
b0, b1 = 0, p.bn

# compute amplitude
for q in p.qlist:

  # load data
  b, obs = (np.loadtxt(f.quad[q].ocbs,usecols=(0,1))).T[:,b0:b1]
  sim = np.array([np.loadtxt(f.quad[q].cl[i],unpack=True)[1][b0:b1] for i in range(1,p.snmax)])

  # amplitude estimate
  print q
  st = ana.statistics(obs,sim)
  ana.statistics.get_amp(st)
  print 'obs A', st.oA, 'sigma(A)', st.sA, 'S/N', 1./st.sA, 'ratio', st.oA/st.sA, 'A>oA', st.p


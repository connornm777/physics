import numpy as np
from prompt_toolkit.filters import vi_insert_multiple_mode
from samba.vgp_motd_ext import vgp_motd_ext

m_e=0.511
m_mu=105.7
m_tau=1777

m_v_e=0.0000022
m_v_mu=0.17
m_v_tau=15.5

m_u=2.3
m_d=4.8
m_c=1275
m_s=95
m_t=173000
m_b=4180

m_Z=91200
m_W=80400
m_H=126000

def koide(*args):
    num = sum(args)
    den = sum(np.sqrt(arg) for arg in args)**2
    ave = 0.5*(1+1.0/len(args))
    return num/den, ave
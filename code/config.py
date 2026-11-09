"""Parameter set of the manuscript (Table 1).

From the MD: T = 250 K, Delta = 48 kJ/mol, w_MD = 3.025 nm, D_lcd = 1.23e4 nm^2/us, bead
counts n_lcd = 154, n_K = 292, n_P = 527, N_chain = 200.  Units L* = 85.93 nm, T* = 0.6003 us.
Fitted: c0K = -1.6, KIN_RETARD = 50, RATE_SCALE = 0.70, c0P/c1P = 2/3.
"""

import numpy as _np

# volume fraction per enzyme molecule, alpha_E = phi_bar n_E / (N_chain n_lcd)
PHI_BAR = 0.30
ALPHA_P = PHI_BAR * 527 / (200 * 154)       # per calcineurin (527 beads)
ALPHA_K = PHI_BAR * 292 / (200 * 154)       # per kinase      (292 beads)


def alphas(phi_bar):
    """The two enzyme volume fractions at a given phi_bar."""
    return (phi_bar * 292 / (200 * 154), phi_bar * 527 / (200 * 154))


# free-draining mobilities Lam_E = Lam_A n_lcd / n_E; the kinase is reduced by KIN_RETARD
LAMK_ROUSE = 1.0 * 154 / 292                # 0.5274
LAMP_ROUSE = 1.0 * 154 / 527                # 0.2922
KIN_RETARD = 50.0

# MD set-ups by outcome, as (N_p, N_k[, pbar]); oscillating:
MD_POINTS = [(10, 2, 0.331), (12, 2, 0.344), (12, 3, 0.348),
             (15, 3, 0.360), (20, 4, 0.350), (14, 4, float('nan'))]

# stable condensate / no condensate:
MD_POINTS_STEADY = [(14, 2), (16, 2), (18, 2), (20, 2), (24, 2), (18, 3)]
MD_POINTS_DISSOLVED = [(8, 4)]

# driving in k_BT; code sign convention DMU_K = -dmu_k, DMU_P = -dmu_p of the manuscript
DELTA_KJMOL = 48.0
T_MD = 250.0                   # K
RT_KJMOL = 8.314462618e-3 * T_MD
DMU_K = 4.5
DELTA = DELTA_KJMOL / RT_KJMOL      # 23.09 at 250 K
DMU_P = DMU_K - DELTA               # -18.59 at 250 K

# forward coefficients (units 1/T*); Table 1 reports K_KIN and K_PHO e^{omega_B}
K_KIN_BASE = 89.05 * 10.0 / 3.0      # 296.83
K_PHO_BASE = 2.26 * 10.0 / 3.0       # 7.5333

RATE_SCALE = 0.70              # joint scale on both forward coefficients
K_KIN = K_KIN_BASE * RATE_SCALE          # 207.78
K_PHO = K_PHO_BASE * RATE_SCALE          # 5.2733


def invariant_K_pho(K_pho=None, w1=4.0):
    """K_PHO e^{omega_B}, the phosphatase coefficient reported in Table 1."""
    return (K_PHO if K_pho is None else K_pho) * _np.exp(w1)


NGRID = 151                    # grid points
LBOX = 1.16                    # box, 1.16 L* = 99.7 nm

EXPLICIT = dict(
    N=NGRID, L=LBOX,
    dmu=DMU_K, dmup=DMU_P,
    kkin=K_KIN * _np.exp(-DMU_K),
    kpho=K_PHO * _np.exp(DMU_P),
    c0P=2.0, c1P=3.0,                       # phosphatase-protein interaction
    c0K=-1.6,                               # kinase-protein attraction (fitted)
    LamK=LAMK_ROUSE / KIN_RETARD,
    LamP=LAMP_ROUSE,
    retard=0.0,
)


def set_Kpho(P, K_pho):
    """Set the phosphatase forward coefficient K_pho."""
    P = dict(P)
    P['kpho'] = K_pho * _np.exp(P['dmup'])
    return P


def params(model, N_k=None, N_p=None, K_pho=None, **kw):
    """Parameters: model.DEFAULT, the overrides above, and the enzyme numbers N_k, N_p."""
    P = dict(model.DEFAULT)
    P.update(EXPLICIT)
    pb = kw.get('phi', P.get('phi', PHI_BAR))
    aK, aP = alphas(pb)
    if N_k is not None:
        P['Ktot'] = N_k * aK
    if N_p is not None:
        P['Ptot'] = N_p * aP
    if K_pho is not None:
        P['kpho'] = K_pho * _np.exp(P['dmup'])
    P.update(kw)
    return P

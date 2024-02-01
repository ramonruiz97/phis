import numpy as np
import complot
import matplotlib.pyplot as plt
import uproot3 as uproot
import ipanema
import pandas as pd
import math
import complot
from uncertainties import unumpy as unp
import uncertainties as unc


# Load data and physics parameters {{{

# Full Run 2 data
data = pd.concat(
    [
        uproot.open(f"/scratch49/marcos.romero/sidecar/{y}/Bs2JpsiPhi/v4r0~mX6@LcosK.root")['DecayTree'].pandas.df(flatten=None) for y in (2015, 2016, 2017, 2018)
    ]
)
df = data  # .query("X_M<1008")


pars = ipanema.Parameters.load("/scratch49/forMoncho/v4r0~mX6@LcosK_auto_run2Dual_vgc_peilian_simul3_amsrd_combined.json").valuesdict(False)
# let's do a gitanada
pars['eta_os'] = 0.3546
pars['eta_ss'] = 0.42388
pars['p0_os'] = 0.389
pars['p1_os'] = 0.8486
pars['p0_ss'] = 0.4325
pars['p1_ss'] = 0.9241
pars['dp0_os'] = 0.009
pars['dp1_os'] = 0.0143
pars['dp0_ss'] = 0
pars['dp1_ss'] = 0

timeres = {
    "sigma_offset": 1.29700e-02,
    "sigma_slope": 8.44600e-01,
    "sigma_curvature": 0
}

# }}}


# some functs {{{

def get_omega(eta, tag, p0, p1, p2, dp0, dp1, dp2, eta_bar):
  result = 0
  result += (p0 + tag * 0.5 * dp0)
  result += (p1 + tag * 0.5 * dp1) * (eta - eta_bar)
  result += (p2 + tag * 0.5 * dp2) * (eta - eta_bar) * (eta - eta_bar)

  if result < 0.0:
    return 0
  return result


def calculate_fk(cosK, cosL, hphi):
  fk = []
  sinK = np.sqrt(1. - cosK * cosK)
  sinL = np.sqrt(1. - cosL * cosL)
  sinphi = np.sin(hphi)
  cosphi = np.cos(hphi)

  fk.append(cosK * cosK * sinL * sinL)
  fk.append(0.5 * sinK * sinK * (1. - cosphi * cosphi * sinL * sinL))
  fk.append(0.5 * sinK * sinK * (1. - sinphi * sinphi * sinL * sinL))
  fk.append(sinK * sinK * sinL * sinL * sinphi * cosphi)
  fk.append(np.sqrt(2.) * sinK * cosK * sinL * cosL * cosphi)
  fk.append(-np.sqrt(2.) * sinK * cosK * sinL * cosL * sinphi)
  fk.append(sinL * sinL / 3.)
  fk.append(2. * sinK * sinL * cosL * cosphi / np.sqrt(6.))
  fk.append(-2. * sinK * sinL * cosL * sinphi / np.sqrt(6.))
  fk.append(2. * cosK * sinL * sinL / np.sqrt(3.))
  return fk

# }}}


# Compute the Aplus and Aminus for the asymmetry {{{

weight_string = "sWeight"
tagging = "data"
asymmetry_plot_plus = []
asymmetry_plot_minus = []
tLL = 0.0
tUL = 2 * np.pi / pars['DM']
num_bins = 5
time_offset = 0.3
num_events = df.shape[0]
print("All events:", num_events)


for i in range(num_events):
  # weight
  wP = df[weight_string].iat[i]
  wM = df[weight_string].iat[i]

  # print(wP)
  tagOS = 0
  tagSS = 0
  dilutionOS = 0.0
  dilutionOSB = 0.0
  dilutionOSBbar = 0.0
  dilutionSS = 0.0
  dilutionSSB = 0.0
  dilutionSSBbar = 0.0

  if tagging == "perfect":
    tagSS = df['tagOSdec'].iat[i]
    tagOS = df['tagSSdec'].iat[i]
    dilutionOSB = 1
    dilutionSSB = 1
    dilutionOSBbar = 1
    dilutionSSBbar = 1
  else:
    tagOS = df['tagOSdec'].values[i]
    tagSS = df['tagSSdec'].values[i]
    omegaOSB = get_omega(df['tagOSeta'].iat[i], +1, pars['p0_os'], pars['p1_os'], 0 * pars['p1_os'], pars['dp0_os'], pars['dp1_os'], 0 * pars['dp1_os'], pars['eta_os'])
    omegaOSBbar = get_omega(df['tagOSeta'].iat[i], -1, pars['p0_os'], pars['p1_os'], 0 * pars['p1_os'], pars['dp0_os'], pars['dp1_os'], 0 * pars['dp1_os'], pars['eta_os'])
    omegaSSB = get_omega(df['tagSSeta'].iat[i], +1, pars['p0_ss'], pars['p1_ss'], 0 * pars['p1_ss'], pars['dp0_ss'], pars['dp1_ss'], 0 * pars['dp1_ss'], pars['eta_ss'])
    omegaSSBbar = get_omega(df['tagSSeta'].iat[i], -1, pars['p0_ss'], pars['p1_ss'], 0 * pars['p1_ss'], pars['dp0_ss'], pars['dp1_ss'], 0 * pars['dp1_ss'], pars['eta_ss'])
    dilutionOSB = 1 - 2 * omegaOSB
    dilutionSSB = 1 - 2 * omegaSSB
    dilutionOSBbar = 1 - 2 * omegaOSBbar
    dilutionSSBbar = 1 - 2 * omegaSSBbar

  frac = ((1 + tagOS * dilutionOSB) * (1 + tagSS * dilutionSSB))
  frac /= ((1 + tagOS * dilutionOSB) * (1 + tagSS * dilutionSSB) + (1 - tagOS * dilutionOSBbar) * (1 - tagSS * dilutionSSBbar))
  wP *= frac
  wM *= (1 - frac)
  # 2. tagging dilution weight
  wM *= np.abs(frac - 0.5) * 2
  wP *= np.abs(frac - 0.5) * 2
  # 3. resolution weight
  sigma_t = df['sigmat'].iat[i]
  delta_t = timeres['sigma_offset'] + timeres['sigma_slope'] * sigma_t + timeres['sigma_curvature'] * sigma_t**2
  wP *= np.exp(-delta_t**2 * pars['DM'] / 2.0)
  wM *= np.exp(-delta_t**2 * pars['DM'] / 2.0)
  # 1. angular weight
  AS = 0.1
  AP = 1 - AS
  fk = calculate_fk(df['cosK'].iat[i], df['cosL'].iat[i], df['hphi'].iat[i])
  _even = AP * pars['fPlon'] * fk[0] + AP * pars['fPper'] * fk[1]
  _odd = AP * pars['fPper'] * fk[2] + AS * fk[6]
  wP *= (_even - _odd) / (_even + _odd)
  wM *= (_even - _odd) / (_even + _odd)

  # fold into one DM period
  asymmetry_plot_plus.append([math.fmod(df['time'].iat[i] - time_offset, tUL), wP])
  asymmetry_plot_minus.append([math.fmod(df['time'].iat[i] - time_offset, tUL), wM])

asymmetry_plot_plus = np.array(asymmetry_plot_plus)
asymmetry_plot_minus = np.array(asymmetry_plot_minus)

_plus = complot.hist(asymmetry_plot_plus[:, 0], num_bins, weights=asymmetry_plot_plus[:, 1], range=(tLL, tUL))
_minus = complot.hist(asymmetry_plot_minus[:, 0], num_bins, weights=asymmetry_plot_minus[:, 1], range=(tLL, tUL))
_uplus = np.sqrt(complot.hist(asymmetry_plot_plus[:, 0], num_bins, weights=asymmetry_plot_plus[:, 1]**2, range=(tLL, tUL)).counts)
_uminus = np.sqrt(complot.hist(asymmetry_plot_minus[:, 0], num_bins, weights=asymmetry_plot_minus[:, 1]**2, range=(tLL, tUL)).counts)

# }}}


# compute the points and lines for the plot {{{

for i in range(num_bins):
  print(f"{_plus.bins[i]:.4f}  | {_plus.counts[i]:+.4f}  | {_minus.counts[i]:+.4f}")

# calculate stuff to plot
__plus = unp.uarray(_plus.counts, _uplus)
__minus = unp.uarray(_minus.counts, _uminus)
asym = (__plus - __minus) / (__plus + __minus)

# calculate an averaged eta
avg_swave = np.mean([v for k, v in pars.items() if k.startswith('fS')])
avg_pwave = 1 - avg_swave
_eta_even = (avg_pwave * pars['fPlon'] + avg_pwave * pars['fPper'])
_eta_odd = (avg_pwave * (1 - pars['fPlon'] - pars['fPper']) + avg_swave * 1)
_eta = (_eta_even - _eta_odd) / (_eta_even + _eta_odd)

# the mparameters phis DG and DM have systematics, lets increare their errors
_phis = unc.ufloat(pars['pPlon'], 0.0023)
_dg = unc.ufloat(pars['DGs'], 0.0050)
_dm = unc.ufloat(pars['DM'], 0.035)
print("phis", _phis)

# compute the approx pdf -- GITANADA
_t = np.linspace(0, tUL, 50) - 0.0
_y = _eta
_y *= unp.sin(_phis)
_y *= unp.sin(_dm * (_t) - np.pi / 2)
_y /= (unp.cosh(0.5 * _dg * _t) - _eta * unp.cos(_phis) * unp.sinh(0.5 * _dg * _t))

# }}}


# plot stuff {{{

fig, axplot, axpull = complot.axes_plotpull()
pulls = complot.compute_pdfpulls(_t, unp.nominal_values(_y), _plus.bins, unp.nominal_values(asym), unp.std_devs(asym), unp.std_devs(asym))
axpull.fill_between(_plus.bins, pulls, 0, step='mid')
axpull.plot(_plus.bins, _plus.bins * 0, '-', color='C0')

axplot.plot(_t, unp.nominal_values(_y))
axplot.fill_between(_t,
                    unp.nominal_values(_y) + unp.std_devs(_y),
                    unp.nominal_values(_y) - unp.std_devs(_y),
                    facecolor="C0",
                    alpha=0.3)
axplot.errorbar(_plus.bins,
                unp.nominal_values(asym),
                yerr=(unp.std_devs(asym))**1,
                xerr=_plus.xerr,
                fmt='.', color='k')

axplot.set_ylim(-0.03, 0.03)
axplot.set_xlim(0, tUL)
axpull.set_xlabel("$t-0.3$ (modulo $2\pi/\Delta m_{s}$) [ps]")
axplot.set_ylabel("$A_{CP}(t)$")
fig.savefig("asymmetry_cp.pdf")

# }}}


# vim: fdm=marker

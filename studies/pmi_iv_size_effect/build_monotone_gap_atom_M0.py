from pathlib import Path
import json, math
import numpy as np
from scipy.optimize import least_squares
from scipy.interpolate import CubicSpline
from scipy.integrate import cumulative_trapezoid
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FixedFormatter, NullFormatter

ROOT=Path('/scratch/gautschi/gonza226/pmi_monotone_gap_atom_M0_20261002')
ROOT.mkdir(parents=True,exist_ok=True)

paths={
 'raw50k':Path('/scratch/gautschi/gonza226/pmi_raw50k_original_20261002/pair_distribution_50k_raw_original.npz'),
 'tol5e-5':Path('/scratch/gautschi/gonza226/pmi_raw50k_tightov_20261002/pair_distribution_50k_tighter_overlap.npz'),
 'tol5e-6':Path('/scratch/gautschi/gonza226/pmi_raw50k_tol5e-6_20261002/pair_distribution_50k_tol5e-6.npz'),
 'raw400k':Path('/scratch/gautschi/gonza226/pmi_400k_raw_noadjust_20261002/pair_distribution_400k_as_generated.npz'),
}
for p in paths.values():
    if not p.exists(): raise FileNotFoundError(str(p))

VALID=['raw50k','tol5e-5','tol5e-6']
X0=1e-6
MIN_PER_REALIZATION=50

data={}
for name,p in paths.items():
    q=np.asarray(np.load(p)['q'],float)
    x=1.0-q
    data[name]={
        'x':x,
        'gaps':x[x<0],
        'contacts':x[x>0],
        'total':int(len(x)),
        'ngap':int(np.count_nonzero(x<0)),
        'ncontact':int(np.count_nonzero(x>0)),
    }
    data[name]['contact_fraction']=data[name]['ncontact']/data[name]['total']

chi_vals=np.array([data[n]['contact_fraction'] for n in VALID],float)
chi=float(np.mean(chi_vals))
pair50eq=float(np.mean([data[n]['total'] for n in VALID]))
contact50eq=float(chi*pair50eq)
gap50eq=float((1-chi)*pair50eq)

# Common signed-log gap bins.
mag=np.geomspace(1e-10,0.5,620)
edges=np.r_[-mag[::-1],0.0]
nb=len(edges)-1

per_counts=[]
per_prob=[]
for n in VALID:
    c,_=np.histogram(data[n]['gaps'],bins=edges)
    per_counts.append(c)
    per_prob.append(c/c.sum())
per_counts=np.asarray(per_counts)
per_prob=np.asarray(per_prob)
mean_prob=np.mean(per_prob,axis=0)
actual_counts=np.sum(per_counts,axis=0)

# Adaptive merging, working from zero outward so the near-contact structure is preserved.
idx=np.flatnonzero(mean_prob>0)
groups=[]
hi=int(idx[-1]); ac=0; ap=0.0
for j in idx[::-1]:
    ac += int(actual_counts[j])
    ap += float(mean_prob[j])
    if ac >= MIN_PER_REALIZATION*len(VALID):
        groups.append((int(j),hi,ac,ap))
        hi=int(j)-1; ac=0; ap=0.0
if ac>0:
    if groups:
        a,b,c,p=groups[-1]
        groups[-1]=(int(idx[0]),b,c+ac,p+ap)
    else:
        groups=[(int(idx[0]),int(idx[-1]),ac,ap)]
groups=sorted(groups,key=lambda z:z[0])

xc=[]; dens=[]; counts=[]; bins=[]
for a,b,c,p in groups:
    lo=float(edges[a]); hi=float(edges[b+1])
    cc=-math.sqrt(abs(lo*hi)) if hi<0 else lo/2
    xc.append(cc); dens.append(p/(hi-lo)); counts.append(c); bins.append((lo,hi))
xc=np.asarray(xc); dens=np.asarray(dens); counts=np.asarray(counts,float)
u_obs=np.arcsinh(xc/X0)
logd_obs=np.log(dens)

# Model: d(log p_g)/du = exp(g(u)) > 0.
# g is a natural cubic spline through a modest number of control points.
# Therefore p_g is smooth and strictly monotone increasing with x.
umin=float(np.min(u_obs))
umax=float(np.max(u_obs))
NCTRL=16
u_ctrl=np.linspace(umin,umax,NCTRL)

# Dense fixed integration grid for every optimization evaluation.
u_dense=np.linspace(umin,umax,80000)
x_dense=X0*np.sinh(u_dense)
dxdu=X0*np.cosh(u_dense)

# Initial derivative estimate from observed log-density slope, forced positive.
slope=np.gradient(logd_obs,u_obs)
slope=np.maximum(slope,1e-4)
g_init=np.interp(u_ctrl,u_obs,np.log(slope),left=np.log(slope[0]),right=np.log(slope[-1]))
g_init=np.clip(g_init,-9,5)

sqrtw=np.sqrt(counts)
sqrtw/=np.median(sqrtw)

def build_density(c):
    cs=CubicSpline(u_ctrl,c,bc_type='natural')
    gd=np.clip(cs(u_dense),-12,8)
    deriv=np.exp(gd)
    h=cumulative_trapezoid(deriv,u_dense,initial=0.0)
    # arbitrary additive constant removed by x-normalization
    h-=np.max(h)
    p_u_shape=np.exp(h)
    # This is p_x(x(u)); normalize with dx/du.
    Z=np.trapezoid(p_u_shape*dxdu,u_dense)
    p_x=p_u_shape/Z
    return p_x,deriv,cs

def predict_at_obs(c):
    p_x,_,_=build_density(c)
    return np.interp(u_obs,u_dense,p_x)

# Smoothness regularization selected conservatively: monotonicity is hard constraint,
# regularization only suppresses unnecessary curvature of g.
LAMBDA=0.08
def residual(c):
    pred=predict_at_obs(c)
    r_data=(np.log(np.maximum(pred,1e-300))-logd_obs)*sqrtw
    d2=np.diff(c,2)
    r_smooth=np.sqrt(LAMBDA)*d2
    return np.r_[r_data,r_smooth]

sol=least_squares(residual,g_init,bounds=(-12*np.ones(NCTRL),8*np.ones(NCTRL)),
                  xtol=1e-11,ftol=1e-11,gtol=1e-11,max_nfev=5000,verbose=0)
if not sol.success:
    raise RuntimeError(sol.message)

pg,log_slope,cs=build_density(sol.x)

# Extend smoothly toward 0^- using the same positive derivative at the fitted near-contact end.
# This avoids the previous artificial taper-to-zero. Extrapolation is only over the tiny
# interval between the last observed gap bin and x=0.
u0=0.0
u_ext=np.linspace(umax,u0,30000)
if u_ext[-1] < u_ext[0]:
    raise RuntimeError('unexpected u ordering')
cs_ext=CubicSpline(u_ctrl,sol.x,bc_type='natural',extrapolate=True)
g_ext=np.clip(cs_ext(u_ext),-12,8)
deriv_ext=np.exp(g_ext)
# Start h at log pg(end), integrate positive derivative in u.
h0=np.log(pg[-1])
h_ext=h0+cumulative_trapezoid(deriv_ext,u_ext,initial=0.0)
x_ext=X0*np.sinh(u_ext)
p_ext=np.exp(np.clip(h_ext,-745,700))

# Assemble full conditional gap curve and renormalize over [-0.5,0).
xg=np.r_[x_dense,x_ext[1:]]
pg_all=np.r_[pg,p_ext[1:]]
pg_all/=np.trapezoid(pg_all,xg)

# Strict numerical monotonicity verification.
dp=np.diff(pg_all)
monotone_violations=int(np.count_nonzero(dp < -1e-12*np.max(pg_all)))
min_dp=float(np.min(dp))
p0minus=float(pg_all[-1])

# Model bin probabilities for validation.
cdf=np.r_[0.0,cumulative_trapezoid(pg_all,xg)]
cdf/=cdf[-1]
cdf_edges=np.interp(edges,xg,cdf,left=0.0,right=1.0)
fit_prob=np.maximum(0.0,np.diff(cdf_edges))
fit_prob/=fit_prob.sum()

def metrics(sample):
    c,_=np.histogram(sample,bins=edges)
    p=c/c.sum()
    l1=float(np.sum(np.abs(p-fit_prob)))
    ks=float(np.max(np.abs(np.cumsum(p)-np.cumsum(fit_prob))))
    return {'L1':l1,'TV':0.5*l1,'KS':ks}

validation={n:metrics(data[n]['gaps']) for n in data}

# Pooled target comparison.
pool_l1=float(np.sum(np.abs(mean_prob-fit_prob)))
pool_ks=float(np.max(np.abs(np.cumsum(mean_prob)-np.cumsum(fit_prob))))

# First two conditional gap moments.
target_m1=float(np.mean([np.mean(data[n]['gaps']) for n in VALID]))
target_m2=float(np.mean([np.mean(data[n]['gaps']**2) for n in VALID]))
fit_m1=float(np.trapezoid(xg*pg_all,xg))
fit_m2=float(np.trapezoid(xg*xg*pg_all,xg))

# Full representative measure moments: contact atom at zero contributes zero to x^k, k>0.
full_m1=(1-chi)*fit_m1
full_m2=(1-chi)*fit_m2

# CDF of physical measure: continuous gap branch carries 1-chi and jumps by chi at zero.
Fg=np.r_[0.0,cumulative_trapezoid(pg_all,xg)]
Fg/=Fg[-1]
Fphys=(1-chi)*Fg

summary={
    'model':'physical measure = smooth monotone conditional gap density on x<0 plus contact atom at x=0',
    'signed_coordinate':'x=gamma/(2R)=1-q',
    'training_packings':VALID,
    'independent_validation':'raw400k',
    'contact_atom_mass':chi,
    'contact_fraction_range':[float(np.min(chi_vals)),float(np.max(chi_vals))],
    'representative_50k_pair_count':pair50eq,
    'representative_50k_gap_count':gap50eq,
    'representative_50k_contact_count':contact50eq,
    'gap_fit':{
        'construction':'d log(p_g)/du = exp(g(u)), g natural cubic spline',
        'u':'asinh(x/1e-6)',
        'n_control_points':NCTRL,
        'smoothness_lambda':LAMBDA,
        'adaptive_min_actual_observations_per_merged_bin':MIN_PER_REALIZATION*len(VALID),
        'merged_bins':int(len(xc)),
        'optimizer_cost':float(sol.cost),
        'optimizer_nfev':int(sol.nfev),
        'monotone_violations':monotone_violations,
        'minimum_discrete_dp':min_dp,
        'p_g_0minus':p0minus,
        'pooled_TV':0.5*pool_l1,
        'pooled_KS':pool_ks,
        'validation':validation,
        'pooled_target_moment1':target_m1,
        'pooled_target_moment2':target_m2,
        'fit_moment1':fit_m1,
        'fit_moment2':fit_m2,
        'moment1_relative_error':(fit_m1-target_m1)/abs(target_m1),
        'moment2_relative_error':(fit_m2-target_m2)/target_m2,
    },
    'representative_measure_moments':{
        'mean_x':full_m1,
        'mean_x2':full_m2,
        'hertz_moment_initial':0.0,
    },
    'physical_interpretation':'RCP positive-overlap depths are discarded. The finite contact population is represented by chi*delta_0. The gap density remains monotone increasing to contact and is not tapered to zero.',
    'numerical_regularization':'None is imposed in the physical model. If a solver cannot represent an atom, regularize the contact atom separately after defining this measure; do not alter the gap fit.',
}
(ROOT/'monotone_gap_atom_summary.json').write_text(json.dumps(summary,indent=2))
np.savetxt(ROOT/'monotone_gap_density.csv',np.c_[xg,pg_all,(1-chi)*pg_all,pair50eq*(1-chi)*pg_all],
           delimiter=',',header='x_gamma_over_2R,p_gap_conditional,P0_gap_continuous,M50k_gap_continuous',comments='')
np.savetxt(ROOT/'physical_measure_cdf.csv',np.c_[xg,Fphys],
           delimiter=',',header='x_gamma_over_2R,F0_before_contact_jump',comments='')
np.savetxt(ROOT/'fit_control_points.csv',np.c_[u_ctrl,sol.x],
           delimiter=',',header='u_control,g_log_slope_control',comments='')

plt.rcParams.update({'figure.facecolor':'white','axes.facecolor':'white','savefig.facecolor':'white',
'font.family':'serif','font.serif':['Times New Roman','Times','DejaVu Serif'],'mathtext.fontset':'dejavuserif',
'font.size':9.0,'axes.labelsize':10.2,'xtick.labelsize':8.2,'ytick.labelsize':8.7,'legend.fontsize':7.0,
'axes.linewidth':0.85,'xtick.major.width':0.85,'ytick.major.width':0.85,'xtick.minor.width':0.65,'ytick.minor.width':0.65,
'xtick.direction':'in','ytick.direction':'in','lines.linewidth':1.55,'pdf.fonttype':42,'ps.fonttype':42})

def setup_logx(ax,ylabel):
    ax.axvline(0,color='.45',ls=':',lw=.85)
    ax.set_xscale('symlog',linthresh=1e-8,linscale=1,base=10)
    ax.set_xlim(-0.5,2e-6)
    ticks=[-1e-1,-1e-3,-1e-5,-1e-7,0]
    labels=[r'$-10^{-1}$',r'$-10^{-3}$',r'$-10^{-5}$',r'$-10^{-7}$','0']
    ax.xaxis.set_major_locator(FixedLocator(ticks)); ax.xaxis.set_major_formatter(FixedFormatter(labels)); ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel(r'$\gamma/(2R)$'); ax.set_ylabel(ylabel)
    ax.tick_params(which='major',top=True,right=True,length=4); ax.tick_params(which='minor',top=True,right=True,length=2.5)
    ax.grid(False)

# Figure 1: discrete conditional gaps and hard-monotone smooth fit.
fig,ax=plt.subplots(figsize=(6.8,3.5))
for n,marker in zip(['raw50k','tol5e-5','tol5e-6','raw400k'],['o','s','^','d']):
    c,_=np.histogram(data[n]['gaps'],bins=edges)
    den=(c/c.sum())/np.diff(edges); cen=0.5*(edges[:-1]+edges[1:]); m=den>0
    ax.plot(cen[m],den[m],ls='none',marker=marker,mfc='none',ms=2.0,label=n+' discrete gaps')
ax.plot(xg,pg_all,lw=2.0,label='monotone representative gap fit')
setup_logx(ax,r'$p_g(x\mid x<0)$')
ax.set_yscale('log')
vv=pg_all[pg_all>0]; ax.set_ylim(max(1e-3,float(vv.min())*.4),float(vv.max())*2)
ax.legend(frameon=False,ncol=2,loc='best')
fig.tight_layout(pad=.25)
fig.savefig(ROOT/'Monotone_gap_fit_vs_discrete_Rodrigues.png',dpi=300,bbox_inches='tight',pad_inches=.02)
fig.savefig(ROOT/'Monotone_gap_fit_vs_discrete_Rodrigues.pdf',bbox_inches='tight',pad_inches=.02)
plt.close(fig)

# Figure 2: near-contact zoom proving monotonic continuation to zero.
fig,ax=plt.subplots(figsize=(3.3,3.0))
mask=xg>-2e-4
ax.plot(xg[mask],pg_all[mask],lw=1.8,label='monotone gap fit')
for n,marker in zip(VALID,['o','s','^']):
    c,_=np.histogram(data[n]['gaps'],bins=edges)
    den=(c/c.sum())/np.diff(edges); cen=0.5*(edges[:-1]+edges[1:])
    m=(den>0)&(cen>-2e-4)
    ax.plot(cen[m],den[m],ls='none',marker=marker,mfc='none',ms=2.0,label=n)
setup_logx(ax,r'$p_g(x\mid x<0)$')
ax.set_xlim(-2e-4,1e-8); ax.set_yscale('log')
ax.legend(frameon=False,loc='best')
fig.tight_layout(pad=.25)
fig.savefig(ROOT/'Monotone_gap_near_contact_Rodrigues.png',dpi=300,bbox_inches='tight',pad_inches=.02)
fig.savefig(ROOT/'Monotone_gap_near_contact_Rodrigues.pdf',bbox_inches='tight',pad_inches=.02)
plt.close(fig)

# Figure 3: physical measure CDF, with explicit contact jump.
fig,ax=plt.subplots(figsize=(6.8,3.5))
ax.plot(xg,Fphys,lw=1.8,label='continuous gap contribution')
ax.plot([0,0],[1-chi,1],lw=2.4,label=rf'contact atom $\chi_c={chi:.3f}$')
ax.scatter([0],[1],s=16,zorder=3)
ax.set_xscale('symlog',linthresh=1e-8,linscale=1,base=10)
ax.set_xlim(-0.5,2e-6); ax.set_ylim(0,1.03)
ax.set_xlabel(r'$\gamma/(2R)$'); ax.set_ylabel(r'$F_0(x)$')
ax.tick_params(which='major',top=True,right=True,length=4); ax.tick_params(which='minor',top=True,right=True,length=2.5)
ax.grid(False); ax.legend(frameon=False,loc='best')
fig.tight_layout(pad=.25)
fig.savefig(ROOT/'Physical_initial_measure_CDF_Rodrigues.png',dpi=300,bbox_inches='tight',pad_inches=.02)
fig.savefig(ROOT/'Physical_initial_measure_CDF_Rodrigues.pdf',bbox_inches='tight',pad_inches=.02)
plt.close(fig)

# Figure 4: 50k-equivalent continuous gap population with explicit atom annotation.
fig,ax=plt.subplots(figsize=(6.8,3.5))
Mgap=pair50eq*(1-chi)*pg_all
ax.plot(xg,Mgap,lw=2.0,label='continuous gap population')
setup_logx(ax,r'$M_{50k}^{\rm gap}(x)$')
ax.set_yscale('log')
mv=Mgap[Mgap>0]; ax.set_ylim(max(1e-1,float(mv.min())*.4),float(mv.max())*2)
ax.text(0.98,0.92,rf'contact atom at $x=0$: $N_c^{{50k}}={contact50eq:,.0f}$',
        transform=ax.transAxes,ha='right',va='top')
ax.legend(frameon=False,loc='best')
fig.tight_layout(pad=.25)
fig.savefig(ROOT/'Physical_initial_M50k_gap_plus_atom_Rodrigues.png',dpi=300,bbox_inches='tight',pad_inches=.02)
fig.savefig(ROOT/'Physical_initial_M50k_gap_plus_atom_Rodrigues.pdf',bbox_inches='tight',pad_inches=.02)
plt.close(fig)

print('MONOTONE_GAP_ATOM_SUMMARY='+json.dumps(summary,sort_keys=True),flush=True)
print('MONOTONE_GAP_ATOM_RESULT_DIR='+str(ROOT),flush=True)

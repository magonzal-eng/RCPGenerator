from pathlib import Path
import json, math
import numpy as np
from scipy.optimize import minimize, LinearConstraint
from scipy.interpolate import BSpline
from scipy.integrate import cumulative_trapezoid
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FixedFormatter, NullFormatter

ROOT=Path('/scratch/gautschi/gonza226/pmi_nearcontact_monotone_gap_atom_M0_20261002')
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
N_BASIS=20
SMOOTH_LAMBDA=1e-4
MOMENT_WEIGHT=200.0

data={}
for name,p in paths.items():
    q=np.asarray(np.load(p)['q'],float)
    x=1.0-q
    gaps=x[x<0]
    contacts=x[x>0]
    data[name]=dict(x=x,gaps=gaps,contacts=contacts,total=int(len(x)),
                    ngap=int(len(gaps)),ncontact=int(len(contacts)))
    data[name]['contact_fraction']=data[name]['ncontact']/data[name]['total']

chi_vals=np.array([data[n]['contact_fraction'] for n in VALID],float)
chi=float(np.mean(chi_vals))
pair50eq=float(np.mean([data[n]['total'] for n in VALID]))
contact50eq=float(chi*pair50eq)
gap50eq=float((1-chi)*pair50eq)

# Common logarithmic gap bins.
mag=np.geomspace(1e-10,0.5,620)
edges=np.r_[-mag[::-1],0.0]
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

# Adaptive merge from contact outward, >=50 observations per realization.
idx=np.flatnonzero(mean_prob>0)
groups=[]; hi=int(idx[-1]); ac=0; ap=0.0
for j in idx[::-1]:
    ac += int(actual_counts[j]); ap += float(mean_prob[j])
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

xc=[]; dens=[]; counts=[]; merged_bins=[]
for a,b,c,p in groups:
    lo=float(edges[a]); hi=float(edges[b+1])
    cc=-math.sqrt(abs(lo*hi)) if hi<0 else lo/2
    xc.append(cc); dens.append(p/(hi-lo)); counts.append(c); merged_bins.append((lo,hi))
xc=np.asarray(xc); dens=np.asarray(dens); counts=np.asarray(counts,float)
u_obs=np.arcsinh(xc/X0)
logd_obs=np.log(dens)

# Data-driven onset of the final monotonically increasing branch:
# earliest bin in the maximal suffix with nondecreasing pooled adaptive density.
mono_start=len(dens)-1
while mono_start>0 and dens[mono_start-1] <= dens[mono_start]:
    mono_start-=1
x_cut=float(xc[mono_start]); u_cut=float(u_obs[mono_start])

# Cubic B-spline log-density on the whole support. It is unconstrained for x<x_cut.
degree=3
u_min=float(np.arcsinh(-0.5/X0)); u_max=0.0
n_internal=N_BASIS-(degree+1)
internal=np.linspace(u_min,u_max,n_internal+2)[1:-1]
knots=np.r_[np.repeat(u_min,degree+1),internal,np.repeat(u_max,degree+1)]

def basis_matrix(u,der=0):
    u=np.atleast_1d(u)
    M=np.empty((len(u),N_BASIS))
    for j in range(N_BASIS):
        c=np.zeros(N_BASIS); c[j]=1.0
        sp=BSpline(knots,c,degree,extrapolate=False)
        if der: sp=sp.derivative(der)
        M[:,j]=sp(u)
    return M

Bobs=basis_matrix(u_obs)
u_dense=np.linspace(u_min,u_max,12000)
x_dense=X0*np.sinh(u_dense)
dxdu=X0*np.cosh(u_dense)
Bd=basis_matrix(u_dense)

# Hard derivative constraint only on near-contact branch.
u_mon=np.linspace(u_cut,u_max,220)
Dmon=basis_matrix(u_mon,der=1)
u_smooth=np.linspace(u_min,u_max,320)
D2=basis_matrix(u_smooth,der=2)

sqrtw=np.sqrt(counts); sqrtw/=np.median(sqrtw)

# Exact equal-realization conditional gap moments from raw samples.
target_m1=float(np.mean([np.mean(data[n]['gaps']) for n in VALID]))
target_m2=float(np.mean([np.mean(data[n]['gaps']**2) for n in VALID]))

# Smooth weighted least-squares initial coefficient estimate.
A=np.vstack([Bobs*sqrtw[:,None],np.sqrt(SMOOTH_LAMBDA)*D2])
yy=np.r_[logd_obs*sqrtw,np.zeros(len(u_smooth))]
c0=np.linalg.lstsq(A,yy,rcond=None)[0]

# Make starting point feasible by adding a linear ramp in u if needed.
min_initial=float(np.min(Dmon@c0))
if min_initial<0:
    c_u=np.linalg.lstsq(Bd,u_dense,rcond=None)[0]
    c0 += (-min_initial+0.1)*c_u

def normalized_values(c):
    h_dense=Bd@c
    hm=float(np.max(h_dense))
    sh=np.exp(np.clip(h_dense-hm,-700,0))
    Z=float(np.trapezoid(sh*dxdu,u_dense))
    logZ=math.log(Z)+hm
    p_dense=np.exp(np.clip(h_dense-logZ,-745,700))
    logp_obs=Bobs@c-logZ
    m1=float(np.trapezoid(x_dense*p_dense,x_dense))
    m2=float(np.trapezoid(x_dense*x_dense*p_dense,x_dense))
    return logp_obs,p_dense,m1,m2

def objective(c):
    logp,pd,m1,m2=normalized_values(c)
    r=(logp-logd_obs)*sqrtw
    smooth=D2@c
    moment_pen=MOMENT_WEIGHT*len(r)*(
        ((m1-target_m1)/abs(target_m1))**2+
        ((m2-target_m2)/target_m2)**2
    )
    return float(np.dot(r,r)+SMOOTH_LAMBDA*len(r)*np.mean(smooth*smooth)+moment_pen)

constraint=LinearConstraint(Dmon,0.0,np.inf)
sol=minimize(objective,c0,method='SLSQP',constraints=[constraint],
             options={'maxiter':2500,'ftol':1e-11,'disp':False})
if not sol.success:
    raise RuntimeError('constrained fit failed: '+str(sol.message))

logp_obs,p_gap,m1_fit,m2_fit=normalized_values(sol.x)

# Dense hard-monotonicity audit on near-contact branch.
u_audit=np.linspace(u_cut,0.0,20000)
Daudit=basis_matrix(u_audit,der=1)
deriv_audit=Daudit@sol.x
min_deriv=float(np.min(deriv_audit))
violations=int(np.count_nonzero(deriv_audit < -1e-10))
if violations:
    raise RuntimeError(f'monotonicity audit failed: {violations} violations, min={min_deriv}')

# Convert model to source-bin probabilities for TV/KS validation.
cdf=np.r_[0.0,cumulative_trapezoid(p_gap,x_dense)]
cdf/=cdf[-1]
cdf_edges=np.interp(edges,x_dense,cdf,left=0.0,right=1.0)
fit_prob=np.maximum(0.0,np.diff(cdf_edges)); fit_prob/=fit_prob.sum()

def gap_metrics(sample):
    c,_=np.histogram(sample,bins=edges)
    pp=c/c.sum()
    l1=float(np.sum(np.abs(pp-fit_prob)))
    ks=float(np.max(np.abs(np.cumsum(pp)-np.cumsum(fit_prob))))
    return dict(L1=l1,TV=0.5*l1,KS=ks)

validation={n:gap_metrics(data[n]['gaps']) for n in data}
pool_l1=float(np.sum(np.abs(mean_prob-fit_prob)))
pool_ks=float(np.max(np.abs(np.cumsum(mean_prob)-np.cumsum(fit_prob))))

# Physical measure: continuous gap branch plus contact atom at zero.
Pgap=(1-chi)*p_gap
Mgap=pair50eq*Pgap
Fg=np.r_[0.0,cumulative_trapezoid(p_gap,x_dense)]; Fg/=Fg[-1]
Fphys=(1-chi)*Fg

summary={
 'model':'PMI-IV initial measure = smooth fitted gap density plus finite contact atom',
 'signed_coordinate':'x=gamma/(2R)=1-q',
 'training_packings':VALID,
 'independent_validation':'raw400k',
 'contact_atom_mass':chi,
 'contact_fraction_range':[float(np.min(chi_vals)),float(np.max(chi_vals))],
 'representative_50k_pair_count':pair50eq,
 'representative_50k_gap_count':gap50eq,
 'representative_50k_contact_count':contact50eq,
 'gap_fit':{
   'basis':'cubic B-spline for log p_g',
   'u':'asinh(x/1e-6)',
   'n_basis':N_BASIS,
   'smoothness_lambda':SMOOTH_LAMBDA,
   'moment_weight':MOMENT_WEIGHT,
   'adaptive_min_actual_observations_per_merged_bin':MIN_PER_REALIZATION*len(VALID),
   'merged_bins':int(len(xc)),
   'near_contact_monotonicity_cut_x':x_cut,
   'near_contact_monotonicity_cut_u':u_cut,
   'cut_selection':'earliest bin in maximal nondecreasing suffix of pooled adaptive discrete gap density',
   'constraint':'d log(p_g)/du >= 0 only for x >= x_cut',
   'monotonicity_audit_points':int(len(u_audit)),
   'monotonicity_violations':violations,
   'minimum_near_contact_log_density_derivative':min_deriv,
   'p_g_0minus':float(p_gap[-1]),
   'pooled_TV':0.5*pool_l1,
   'pooled_KS':pool_ks,
   'validation':validation,
   'target_gap_moment1':target_m1,
   'target_gap_moment2':target_m2,
   'fit_gap_moment1':m1_fit,
   'fit_gap_moment2':m2_fit,
   'moment1_relative_error':(m1_fit-target_m1)/abs(target_m1),
   'moment2_relative_error':(m2_fit-target_m2)/target_m2,
   'optimizer_success':bool(sol.success),
   'optimizer_iterations':int(sol.nit),
   'objective':float(sol.fun),
 },
 'representative_measure_moments':{
   'mean_x':(1-chi)*m1_fit,
   'mean_x2':(1-chi)*m2_fit,
   'hertz_moment_initial':0.0,
 },
 'physical_interpretation':'The gap density is allowed to retain its measured nonmonotone broad shape away from contact, but is hard-constrained to increase monotonically on the data-supported near-contact branch. RCP positive overlap depths are discarded; their stable population is retained as chi*delta_0.',
 'numerical_regularization':'No contact mollifier is part of the physical initial measure. Any solver-specific regularization must be applied to the atom afterward without altering p_g.',
}
(ROOT/'nearcontact_monotone_gap_atom_summary.json').write_text(json.dumps(summary,indent=2))
np.savetxt(ROOT/'nearcontact_monotone_gap_density.csv',
           np.c_[x_dense,p_gap,Pgap,Mgap],
           delimiter=',',header='x_gamma_over_2R,p_gap_conditional,P0_gap_continuous,M50k_gap_continuous',comments='')
np.savetxt(ROOT/'physical_measure_cdf.csv',np.c_[x_dense,Fphys],
           delimiter=',',header='x_gamma_over_2R,F0_before_contact_jump',comments='')
np.savetxt(ROOT/'fit_control_coefficients.csv',np.c_[np.arange(N_BASIS),sol.x],
           delimiter=',',header='basis_index,coefficient',comments='')

plt.rcParams.update({'figure.facecolor':'white','axes.facecolor':'white','savefig.facecolor':'white',
'font.family':'serif','font.serif':['Times New Roman','Times','DejaVu Serif'],'mathtext.fontset':'dejavuserif',
'font.size':9.0,'axes.labelsize':10.2,'xtick.labelsize':8.2,'ytick.labelsize':8.7,'legend.fontsize':7.0,
'axes.linewidth':0.85,'xtick.major.width':0.85,'ytick.major.width':0.85,'xtick.minor.width':0.65,'ytick.minor.width':0.65,
'xtick.direction':'in','ytick.direction':'in','lines.linewidth':1.55,'pdf.fonttype':42,'ps.fonttype':42})

def setup(ax,ylabel,xlim=(-0.5,1e-8)):
    ax.axvline(0,color='.45',ls=':',lw=.85)
    ax.set_xscale('symlog',linthresh=1e-8,linscale=1,base=10)
    ax.set_xlim(*xlim)
    ticks=[-1e-1,-1e-3,-1e-5,-1e-7,0]
    labels=[r'$-10^{-1}$',r'$-10^{-3}$',r'$-10^{-5}$',r'$-10^{-7}$','0']
    ax.xaxis.set_major_locator(FixedLocator(ticks)); ax.xaxis.set_major_formatter(FixedFormatter(labels)); ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel(r'$\gamma/(2R)$'); ax.set_ylabel(ylabel)
    ax.tick_params(which='major',top=True,right=True,length=4); ax.tick_params(which='minor',top=True,right=True,length=2.5)
    ax.grid(False)

# Full gap fit vs discrete data.
fig,ax=plt.subplots(figsize=(6.8,3.5))
for n,marker in zip(['raw50k','tol5e-5','tol5e-6','raw400k'],['o','s','^','d']):
    c,_=np.histogram(data[n]['gaps'],bins=edges)
    den=(c/c.sum())/np.diff(edges); cen=0.5*(edges[:-1]+edges[1:]); m=den>0
    ax.plot(cen[m],den[m],ls='none',marker=marker,mfc='none',ms=2.0,label=n+' discrete gaps')
ax.plot(x_dense,p_gap,lw=2.0,label='representative constrained fit')
ax.axvline(x_cut,ls='--',lw=.9,label=rf'monotone branch onset $x_c={x_cut:.2e}$')
setup(ax,r'$p_g(x\mid x<0)$')
ax.set_yscale('log'); vv=p_gap[p_gap>0]; ax.set_ylim(max(1e-3,float(vv.min())*.4),float(vv.max())*2)
ax.legend(frameon=False,ncol=2,loc='best')
fig.tight_layout(pad=.25)
fig.savefig(ROOT/'Nearcontact_monotone_gap_fit_vs_discrete_Rodrigues.png',dpi=300,bbox_inches='tight',pad_inches=.02)
fig.savefig(ROOT/'Nearcontact_monotone_gap_fit_vs_discrete_Rodrigues.pdf',bbox_inches='tight',pad_inches=.02)
plt.close(fig)

# Near-contact zoom.
fig,ax=plt.subplots(figsize=(3.3,3.0))
mask=x_dense>=min(-2e-4,5*x_cut)
ax.plot(x_dense[mask],p_gap[mask],lw=1.8,label='constrained fit')
for n,marker in zip(VALID,['o','s','^']):
    c,_=np.histogram(data[n]['gaps'],bins=edges)
    den=(c/c.sum())/np.diff(edges); cen=0.5*(edges[:-1]+edges[1:])
    m=(den>0)&(cen>=min(-2e-4,5*x_cut))
    ax.plot(cen[m],den[m],ls='none',marker=marker,mfc='none',ms=2.0,label=n)
ax.axvline(x_cut,ls='--',lw=.9)
setup(ax,r'$p_g(x\mid x<0)$',xlim=(min(-2e-4,5*x_cut),1e-8))
ax.set_yscale('log'); ax.legend(frameon=False,loc='best')
fig.tight_layout(pad=.25)
fig.savefig(ROOT/'Nearcontact_monotone_gap_zoom_Rodrigues.png',dpi=300,bbox_inches='tight',pad_inches=.02)
fig.savefig(ROOT/'Nearcontact_monotone_gap_zoom_Rodrigues.pdf',bbox_inches='tight',pad_inches=.02)
plt.close(fig)

# Physical measure CDF with explicit atom.
fig,ax=plt.subplots(figsize=(6.8,3.5))
ax.plot(x_dense,Fphys,lw=1.8,label='continuous gap contribution')
ax.plot([0,0],[1-chi,1],lw=2.4,label=rf'contact atom $\chi_c={chi:.3f}$')
ax.scatter([0],[1],s=16,zorder=3)
ax.set_xscale('symlog',linthresh=1e-8,linscale=1,base=10); ax.set_xlim(-0.5,2e-6); ax.set_ylim(0,1.03)
ax.set_xlabel(r'$\gamma/(2R)$'); ax.set_ylabel(r'$F_0(x)$')
ax.tick_params(which='major',top=True,right=True,length=4); ax.tick_params(which='minor',top=True,right=True,length=2.5)
ax.grid(False); ax.legend(frameon=False,loc='best')
fig.tight_layout(pad=.25)
fig.savefig(ROOT/'Nearcontact_physical_initial_measure_CDF_Rodrigues.png',dpi=300,bbox_inches='tight',pad_inches=.02)
fig.savefig(ROOT/'Nearcontact_physical_initial_measure_CDF_Rodrigues.pdf',bbox_inches='tight',pad_inches=.02)
plt.close(fig)

# 50k-equivalent population.
fig,ax=plt.subplots(figsize=(6.8,3.5))
ax.plot(x_dense,Mgap,lw=2.0,label='continuous gap population')
setup(ax,r'$M_{50k}^{\rm gap}(x)$')
ax.set_yscale('log'); mv=Mgap[Mgap>0]; ax.set_ylim(max(1e-1,float(mv.min())*.4),float(mv.max())*2)
ax.text(.98,.92,rf'contact atom at $x=0$: $N_c^{{50k}}={contact50eq:,.0f}$',transform=ax.transAxes,ha='right',va='top')
ax.legend(frameon=False,loc='best')
fig.tight_layout(pad=.25)
fig.savefig(ROOT/'Nearcontact_physical_initial_M50k_Rodrigues.png',dpi=300,bbox_inches='tight',pad_inches=.02)
fig.savefig(ROOT/'Nearcontact_physical_initial_M50k_Rodrigues.pdf',bbox_inches='tight',pad_inches=.02)
plt.close(fig)

print('NEARCONTACT_MONOTONE_GAP_ATOM_SUMMARY='+json.dumps(summary,sort_keys=True),flush=True)
print('NEARCONTACT_MONOTONE_GAP_ATOM_RESULT_DIR='+str(ROOT),flush=True)

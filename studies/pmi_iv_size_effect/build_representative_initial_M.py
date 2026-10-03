from pathlib import Path
import json, math, time
import numpy as np
from scipy.interpolate import UnivariateSpline
from scipy.integrate import cumulative_trapezoid
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FixedFormatter, NullFormatter

ROOT=Path('/scratch/gautschi/gonza226/pmi_representative_initial_M_20261002')
ROOT.mkdir(parents=True,exist_ok=True)

paths={
 'raw50k':Path('/scratch/gautschi/gonza226/pmi_raw50k_original_20261002/pair_distribution_50k_raw_original.npz'),
 'tol5e-5':Path('/scratch/gautschi/gonza226/pmi_raw50k_tightov_20261002/pair_distribution_50k_tighter_overlap.npz'),
 'tol5e-6':Path('/scratch/gautschi/gonza226/pmi_raw50k_tol5e-6_20261002/pair_distribution_50k_tol5e-6.npz'),
 'raw400k':Path('/scratch/gautschi/gonza226/pmi_400k_raw_noadjust_20261002/pair_distribution_400k_as_generated.npz'),
}
for p in paths.values():
    if not p.exists(): raise FileNotFoundError(str(p))

X0=1e-6
S=32.0
MIN_COUNT_PER_REALIZATION=50
VALID=['raw50k','tol5e-5','tol5e-6']
EPS_LIST=np.array([5e-6,2e-6,1e-6,5e-7,2e-7,1e-7,5e-8],float)

data={}
for name,p in paths.items():
    q=np.asarray(np.load(p)['q'],float)
    x=1.0-q
    gaps=x[x<0]
    contacts=x[x>0]
    data[name]=dict(q=q,x=x,gaps=gaps,contacts=contacts,total=len(x),
                    ngap=len(gaps),ncontact=len(contacts),
                    contact_fraction=len(contacts)/len(x))

chi_vals=np.array([data[n]['contact_fraction'] for n in VALID])
chi=float(np.mean(chi_vals))
chi_range=[float(np.min(chi_vals)),float(np.max(chi_vals))]
pair_counts=np.array([data[n]['total'] for n in VALID],float)
pair50eq=float(np.mean(pair_counts))
contact50eq=float(chi*pair50eq)

# Negative signed-log source bins, enough to resolve near-contact gaps.
mag=np.geomspace(1e-10,0.5,520)
edges=-mag[::-1]
edges=np.r_[edges,0.0]
nb=len(edges)-1

# Equal-realization conditional gap density. Each valid 50k realization contributes unit mass.
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

# Adaptive merge from near zero outward, requiring ~50 actual observations per realization.
# Build groups on reversed bins (closest to zero first), then sort.
idx=np.flatnonzero(mean_prob>0)
groups=[]
acc_actual=0
acc_prob=0.0
group_hi=int(idx[-1])
for j in idx[::-1]:
    acc_actual += int(actual_counts[j])
    acc_prob += float(mean_prob[j])
    if acc_actual >= MIN_COUNT_PER_REALIZATION*len(VALID):
        groups.append((int(j),group_hi,acc_actual,acc_prob))
        group_hi=int(j)-1; acc_actual=0; acc_prob=0.0
if acc_actual>0:
    if groups:
        a,b,c,p=groups[-1]
        groups[-1]=(int(idx[0]),b,c+acc_actual,p+acc_prob)
    else:
        groups=[(int(idx[0]),int(idx[-1]),acc_actual,acc_prob)]
groups=sorted(groups,key=lambda z:z[0])

xc=[]; dens=[]; w=[]; merged_prob=[]
for a,b,c,p in groups:
    lo=float(edges[a]); hi=float(edges[b+1])
    cc=-math.sqrt(abs(lo*hi)) if hi<0 else lo/2
    xc.append(cc); dens.append(p/(hi-lo)); w.append(math.sqrt(c)); merged_prob.append(p)
xc=np.asarray(xc); dens=np.asarray(dens); w=np.asarray(w); merged_prob=np.asarray(merged_prob)

u=np.arcsinh(xc/X0)
sp=UnivariateSpline(u,np.log(dens),w=w,s=len(u)*S,k=3)

# Dense negative x grid via uniform u, preserving near-zero resolution.
umin=float(np.arcsinh(-0.5/X0))
umax=float(np.arcsinh(-1e-12/X0))
ug=np.linspace(umin,umax,160000)
xg=X0*np.sinh(ug)
raw=np.exp(np.clip(sp(ug),-745,700))
# normalize conditional gap density in x
pg=raw/np.trapezoid(raw,xg)

# Empirical gap validation on common negative bins.
def gap_metrics(sample):
    c,_=np.histogram(sample,bins=edges)
    p=c/c.sum()
    l1=float(np.sum(np.abs(p-mean_prob)))
    ks=float(np.max(np.abs(np.cumsum(p)-np.cumsum(mean_prob))))
    return dict(L1=l1,TV=0.5*l1,KS=ks)
gap_validation={n:gap_metrics(data[n]['gaps']) for n in data}

# Helpers for smooth C2 model.
def smoothstep5(t):
    return 10*t**3-15*t**4+6*t**5

def beta44_kernel(x,eps):
    y=np.zeros_like(x)
    m=(x>0)&(x<eps)
    t=x[m]/eps
    y[m]=(140.0/eps)*t**3*(1-t)**3
    return y

# master evaluation grid: negative via u grid + dense positive local grid.
# For each epsilon, the final distribution is C2 because the gap taper and beta(4,4)
# both vanish with first two derivatives at x=0; contact kernel also vanishes C2 at x=eps.
family=[]
selected_curves={}
for eps in EPS_LIST:
    taper=np.ones_like(xg)
    m=(xg>-eps)&(xg<0)
    t=(-xg[m])/eps
    taper[m]=smoothstep5(t)
    pg_eps=pg*taper
    Zg=float(np.trapezoid(pg_eps,xg))
    pg_eps/=Zg

    # positive grid resolves compact contact kernel
    xp=np.linspace(0.0,eps,24000)
    kp=beta44_kernel(xp,eps)
    Zk=float(np.trapezoid(kp,xp))
    kp/=Zk

    # full moments, including contact mass.
    # gap moments from conditional gap component; contact from kernel.
    mg1=float(np.trapezoid(xg*pg_eps,xg))
    mg2=float(np.trapezoid(xg*xg*pg_eps,xg))
    mc1=float(np.trapezoid(xp*kp,xp))
    mc2=float(np.trapezoid(xp*xp*kp,xp))
    mch=float(np.trapezoid((xp**1.5)*kp,xp))
    m1=(1-chi)*mg1+chi*mc1
    m2=(1-chi)*mg2+chi*mc2
    hertz=chi*mch

    # Taper-induced distortion of gap geometry relative to pooled gap fit.
    # interpolate both on xg; both normalized conditional gap densities.
    gap_l1=float(np.trapezoid(np.abs(pg_eps-pg),xg))

    # positive peak
    im=int(np.argmax(kp))
    family.append(dict(
        epsilon=float(eps),
        gap_taper_L1=gap_l1,
        gap_taper_TV=0.5*gap_l1,
        gap_mass_before_renorm=Zg,
        contact_kernel_integral=Zk,
        contact_kernel_mean_x=mc1,
        contact_kernel_second_moment=mc2,
        contact_hertz_moment_x_3over2=hertz,
        full_first_moment=m1,
        full_second_moment=m2,
        contact_peak_x=float(xp[im]),
        contact_peak_density=float(chi*kp[im]),
    ))
    selected_curves[float(eps)]=(pg_eps,xp,kp)

# Select smallest epsilon not below 1e-7 as a numerically resolved default,
# while requiring taper TV < 0.1%. This selection criterion is explicit and auditable.
eligible=[r for r in family if r['epsilon']>=1e-7 and r['gap_taper_TV']<1e-3]
selected=min(eligible,key=lambda r:r['epsilon']) if eligible else min(family,key=lambda r:r['epsilon'])
eps_sel=float(selected['epsilon'])
pg_sel,xp_sel,kp_sel=selected_curves[eps_sel]

# Assemble selected full PDF on a signed output grid.
# Negative and positive components are disjoint except at x=0, where both are zero C2.
xout=np.r_[xg, xp_sel[1:]]
pout=np.r_[(1-chi)*pg_sel, chi*kp_sel[1:]]
Z=float(np.trapezoid(pout,xout))
pout/=Z
Mout=pair50eq*pout

# Diagnostics of exact mass split.
neg_mass=float(np.trapezoid(pout[xout<0],xout[xout<0]))
pos_mass=float(np.trapezoid(pout[xout>0],xout[xout>0]))

# Moment comparison to valid raw distributions, using empirical dimensionless signed x.
empirical={}
for n in VALID:
    xx=data[n]['x']
    empirical[n]=dict(
        mean_x=float(np.mean(xx)),
        mean_x2=float(np.mean(xx**2)),
        contact_fraction=float(data[n]['contact_fraction']),
        hertz_moment=float(np.mean(np.where(xx>0,xx**1.5,0.0))),
    )
# representative moments
rep=dict(
    mean_x=float(np.trapezoid(xout*pout,xout)),
    mean_x2=float(np.trapezoid(xout*xout*pout,xout)),
    contact_fraction_target=chi,
    hertz_moment=float(np.trapezoid(np.where(xout>0,xout**1.5,0.0)*pout,xout)),
)

summary=dict(
    model='representative hard-sphere-inspired initial signed distribution: pooled gap geometry plus stable contact mass regularized with compact C2 beta(4,4) kernel',
    signed_coordinate='x=gamma/(2R)=1-q',
    training_packings=VALID,
    validation_packing='raw400k',
    contact_fraction_mean=chi,
    contact_fraction_range=chi_range,
    representative_50k_pair_count=pair50eq,
    representative_50k_contact_mass=contact50eq,
    pooled_gap_fit=dict(
        source='equal-weight conditional gap distributions of three valid same-seed 50k packings',
        adaptive_min_actual_observations_per_merged_bin=MIN_COUNT_PER_REALIZATION*len(VALID),
        u='asinh(x/1e-6)',
        log_density_spline=True,
        scipy_smoothing_s=S,
        merged_bins=int(len(xc)),
        validation=gap_validation,
    ),
    regularization_family=family,
    selected_epsilon=eps_sel,
    selection_rule='smallest epsilon >=1e-7 with gap-taper TV <0.1%; epsilon is numerical regularization, not RCP packing tolerance',
    selected_mass_check=dict(total=Z,negative_gap_mass=neg_mass,positive_contact_mass=pos_mass,target_gap_mass=1-chi,target_contact_mass=chi),
    empirical_valid_moments=empirical,
    representative_moments=rep,
    interpretation='positive RCP overlap-depth tail is not fitted; only stable contact mass is retained. epsilon->0 tends to a hard-sphere contact delta at x=0 and zero Hertz moment.',
)
(ROOT/'representative_initial_M_summary.json').write_text(json.dumps(summary,indent=2))
np.savetxt(ROOT/'representative_initial_distribution.csv',np.c_[xout,pout,Mout],
           delimiter=',',header='x_gamma_over_2R,P0_representative,M50k_equiv',comments='')
np.savetxt(ROOT/'regularization_sensitivity.csv',
           np.array([[r[k] for k in ['epsilon','gap_taper_TV','contact_kernel_mean_x','contact_hertz_moment_x_3over2','full_first_moment','full_second_moment']] for r in family]),
           delimiter=',',header='epsilon,gap_taper_TV,contact_kernel_mean_x,contact_hertz_moment_x_3over2,full_first_moment,full_second_moment',comments='')

plt.rcParams.update({'figure.facecolor':'white','axes.facecolor':'white','savefig.facecolor':'white',
'font.family':'serif','font.serif':['Times New Roman','Times','DejaVu Serif'],'mathtext.fontset':'dejavuserif',
'font.size':9.0,'axes.labelsize':10.2,'xtick.labelsize':8.2,'ytick.labelsize':8.7,'legend.fontsize':7.0,
'axes.linewidth':0.85,'xtick.major.width':0.85,'ytick.major.width':0.85,'xtick.minor.width':0.65,'ytick.minor.width':0.65,
'xtick.direction':'in','ytick.direction':'in','lines.linewidth':1.55,'pdf.fonttype':42,'ps.fonttype':42})

def setup(ax,ylabel):
    ax.axvline(0,color='.45',ls=':',lw=.85)
    ax.set_xscale('symlog',linthresh=1e-8,linscale=1,base=10); ax.set_yscale('log')
    ax.set_xlim(-0.5,1e-4)
    ticks=[-1e-1,-1e-3,-1e-5,-1e-7,0,1e-8,1e-7,1e-6,1e-5]
    labels=[r'$-10^{-1}$',r'$-10^{-3}$',r'$-10^{-5}$',r'$-10^{-7}$','0',r'$10^{-8}$',r'$10^{-7}$',r'$10^{-6}$',r'$10^{-5}$']
    ax.xaxis.set_major_locator(FixedLocator(ticks)); ax.xaxis.set_major_formatter(FixedFormatter(labels)); ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel(r'$\gamma/(2R)$'); ax.set_ylabel(ylabel)
    ax.tick_params(which='major',top=True,right=True,length=4); ax.tick_params(which='minor',top=True,right=True,length=2.5)
    ax.grid(False)

# Figure 1: pooled gap fit vs discrete conditional gap histograms and 400k validation.
fig,ax=plt.subplots(figsize=(6.8,3.5))
for n,marker in zip(['raw50k','tol5e-5','tol5e-6','raw400k'],['o','s','^','d']):
    c,_=np.histogram(data[n]['gaps'],bins=edges)
    prob=c/c.sum(); width=np.diff(edges)
    den=prob/width; cen=0.5*(edges[:-1]+edges[1:])
    m=den>0
    ax.plot(cen[m],den[m],ls='none',marker=marker,mfc='none',ms=2.0,label=n+' gaps')
ax.plot(xg,pg,lw=1.8,label='equal-weight pooled gap fit')
setup(ax,r'$p_g(x\mid x<0)$')
positive=np.r_[pg]
ax.set_ylim(max(1e-3,float(positive[positive>0].min())*.5),float(positive.max())*2)
ax.legend(frameon=False,ncol=2,loc='best')
fig.tight_layout(pad=.25)
fig.savefig(ROOT/'Representative_gap_pool_validation_Rodrigues.png',dpi=300,bbox_inches='tight',pad_inches=.02)
fig.savefig(ROOT/'Representative_gap_pool_validation_Rodrigues.pdf',bbox_inches='tight',pad_inches=.02)
plt.close(fig)

# Figure 2: selected representative full P0 and valid discrete packings.
fig,ax=plt.subplots(figsize=(6.8,3.5))
for n,marker in zip(['raw50k','tol5e-5','tol5e-6'],['o','s','^']):
    # common signed log histogram for discrete display
    neg=np.geomspace(1e-10,0.5,350); pos=np.geomspace(1e-10,1e-3,240)
    ed=np.unique(np.r_[-neg[::-1],0.0,pos]); ed.sort()
    c,_=np.histogram(data[n]['x'],bins=ed)
    den=c/c.sum()/np.diff(ed); cen=0.5*(ed[:-1]+ed[1:]); m=den>0
    ax.plot(cen[m],den[m],ls='none',marker=marker,mfc='none',ms=2.0,label=n+' discrete')
ax.plot(xout,pout,lw=2.0,label=rf'representative $P_0$, $\varepsilon={eps_sel:.0e}$')
setup(ax,r'$P_0(x)$')
pp=pout[pout>0]
ax.set_ylim(max(1e-3,float(pp.min())*.4),float(pp.max())*2)
ax.legend(frameon=False,ncol=2,loc='best')
fig.tight_layout(pad=.25)
fig.savefig(ROOT/'Representative_initial_P0_Rodrigues.png',dpi=300,bbox_inches='tight',pad_inches=.02)
fig.savefig(ROOT/'Representative_initial_P0_Rodrigues.pdf',bbox_inches='tight',pad_inches=.02)
plt.close(fig)

# Figure 3: regularization family near contact.
fig,ax=plt.subplots(figsize=(6.8,3.5))
for eps in EPS_LIST:
    pge,xp,kp=selected_curves[float(eps)]
    xx=np.r_[xg,xp[1:]]
    yy=np.r_[(1-chi)*pge,chi*kp[1:]]
    yy/=np.trapezoid(yy,xx)
    ax.plot(xx,yy,lw=1.2,label=rf'$\varepsilon={eps:.0e}$')
setup(ax,r'$P_0^{\varepsilon}(x)$')
ax.set_xlim(-1e-3,1e-5)
vals=[]
for eps in EPS_LIST:
    pge,xp,kp=selected_curves[float(eps)]
    vals.extend(((1-chi)*pge)[(1-chi)*pge>0].tolist()); vals.extend((chi*kp)[chi*kp>0].tolist())
ax.set_ylim(max(1e-2,min(vals)*.4),max(vals)*2)
ax.legend(frameon=False,ncol=2,loc='best')
fig.tight_layout(pad=.25)
fig.savefig(ROOT/'Representative_regularization_family_Rodrigues.png',dpi=300,bbox_inches='tight',pad_inches=.02)
fig.savefig(ROOT/'Representative_regularization_family_Rodrigues.pdf',bbox_inches='tight',pad_inches=.02)
plt.close(fig)

# Figure 4: M50k-equivalent selected distribution
fig,ax=plt.subplots(figsize=(6.8,3.5))
ax.plot(xout,Mout,lw=2.0,label=rf'$M_{{50k}}^{{eq}}$, $\varepsilon={eps_sel:.0e}$')
setup(ax,r'$M_{50k}^{\rm eq}(x)$')
mm=Mout[Mout>0]; ax.set_ylim(max(1e-1,float(mm.min())*.4),float(mm.max())*2)
ax.legend(frameon=False,loc='best')
fig.tight_layout(pad=.25)
fig.savefig(ROOT/'Representative_initial_M50k_Rodrigues.png',dpi=300,bbox_inches='tight',pad_inches=.02)
fig.savefig(ROOT/'Representative_initial_M50k_Rodrigues.pdf',bbox_inches='tight',pad_inches=.02)
plt.close(fig)

print('REPRESENTATIVE_INITIAL_M_SUMMARY='+json.dumps(summary,sort_keys=True),flush=True)
print('REPRESENTATIVE_INITIAL_M_RESULT_DIR='+str(ROOT),flush=True)

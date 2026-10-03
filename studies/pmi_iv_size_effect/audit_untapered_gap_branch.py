from pathlib import Path
import json, math
import numpy as np
from scipy.interpolate import UnivariateSpline
from scipy.integrate import cumulative_trapezoid
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FixedFormatter, NullFormatter

ROOT=Path('/scratch/gautschi/gonza226/pmi_gap_branch_audit_20261002')
ROOT.mkdir(parents=True, exist_ok=True)

paths={
 'raw50k':Path('/scratch/gautschi/gonza226/pmi_raw50k_original_20261002/pair_distribution_50k_raw_original.npz'),
 'tol5e-5':Path('/scratch/gautschi/gonza226/pmi_raw50k_tightov_20261002/pair_distribution_50k_tighter_overlap.npz'),
 'tol5e-6':Path('/scratch/gautschi/gonza226/pmi_raw50k_tol5e-6_20261002/pair_distribution_50k_tol5e-6.npz'),
 'raw400k':Path('/scratch/gautschi/gonza226/pmi_400k_raw_noadjust_20261002/pair_distribution_400k_as_generated.npz'),
}
VALID=['raw50k','tol5e-5','tol5e-6']
X0=1e-6
S=32.0
MIN_PER_REALIZATION=50

data={}
for n,p in paths.items():
    q=np.asarray(np.load(p)['q'],float)
    x=1.0-q
    data[n]=dict(x=x,gaps=x[x<0],contacts=x[x>0],total=len(x))
    data[n]['contact_fraction']=len(data[n]['contacts'])/len(x)

chi=float(np.mean([data[n]['contact_fraction'] for n in VALID]))
pair50eq=float(np.mean([data[n]['total'] for n in VALID]))

# Equal-realization pooled gap histogram.
mag=np.geomspace(1e-10,0.5,620)
edges=np.r_[-mag[::-1],0.0]
per_counts=[]; per_prob=[]
for n in VALID:
    c,_=np.histogram(data[n]['gaps'],bins=edges)
    per_counts.append(c)
    per_prob.append(c/c.sum())
per_counts=np.asarray(per_counts)
per_prob=np.asarray(per_prob)
mean_prob=np.mean(per_prob,axis=0)
actual=np.sum(per_counts,axis=0)

idx=np.flatnonzero(mean_prob>0)
groups=[]
hi=int(idx[-1]); ac=0; ap=0.0
for j in idx[::-1]:
    ac += int(actual[j]); ap += float(mean_prob[j])
    if ac >= MIN_PER_REALIZATION*len(VALID):
        groups.append((int(j),hi,ac,ap))
        hi=int(j)-1; ac=0; ap=0.0
if ac:
    if groups:
        a,b,c,p=groups[-1]
        groups[-1]=(int(idx[0]),b,c+ac,p+ap)
    else:
        groups=[(int(idx[0]),int(idx[-1]),ac,ap)]
groups=sorted(groups)

xc=[]; dens=[]; cnt=[]
for a,b,c,p in groups:
    lo=float(edges[a]); hi=float(edges[b+1])
    cc=-math.sqrt(abs(lo*hi)) if hi<0 else lo/2
    xc.append(cc); dens.append(p/(hi-lo)); cnt.append(c)
xc=np.asarray(xc); dens=np.asarray(dens); cnt=np.asarray(cnt,float)

u=np.arcsinh(xc/X0)
sp=UnivariateSpline(u,np.log(dens),w=np.sqrt(cnt),s=len(u)*S,k=3)

# Evaluate all the way to 0^- without any taper.
umin=float(np.arcsinh(-0.5/X0))
ug=np.linspace(umin,0.0,220000,endpoint=False)
xg=X0*np.sinh(ug)
raw=np.exp(np.clip(sp(ug),-745,700))
pg=raw/np.trapezoid(raw,xg)

# Derivative diagnostics.
dlogdu=sp.derivative(1)(ug)
dpdx = pg*dlogdu/(X0*np.cosh(ug))
# near-contact monotone onset = earliest x such that derivative stays nonnegative thereafter.
ok=dpdx>=0
suffix=np.logical_and.accumulate(ok[::-1])[::-1]
ii=np.flatnonzero(suffix)
x_monotone_onset=float(xg[ii[0]]) if len(ii) else float('nan')
u_monotone_onset=float(ug[ii[0]]) if len(ii) else float('nan')
violations_near0=int(np.count_nonzero(dpdx[xg>-1e-3] < 0))
violations_near1e4=int(np.count_nonzero(dpdx[xg>-1e-4] < 0))
violations_near1e5=int(np.count_nonzero(dpdx[xg>-1e-5] < 0))

# Discrete pooled density monotonicity in adaptive bins: locate last local minimum before contact.
# Use log-density differences ordered x increasing to 0.
dd=np.diff(dens)
# find earliest adaptive-bin center from which all subsequent changes are >=0
okd=dd>=0
suffixd=np.logical_and.accumulate(okd[::-1])[::-1]
idc=np.flatnonzero(suffixd)
x_discrete_monotone_onset=float(xc[idc[0]]) if len(idc) else float('nan')

# Validate the untapered fit by bin probabilities.
cdf=np.r_[0.0,cumulative_trapezoid(pg,xg)]
cdf/=cdf[-1]
cdf_edges=np.interp(edges,xg,cdf,left=0.0,right=1.0)
fitprob=np.maximum(0.0,np.diff(cdf_edges)); fitprob/=fitprob.sum()

def metrics(sample):
    c,_=np.histogram(sample,bins=edges)
    p=c/c.sum()
    l1=float(np.sum(np.abs(p-fitprob)))
    ks=float(np.max(np.abs(np.cumsum(p)-np.cumsum(fitprob))))
    return dict(L1=l1,TV=0.5*l1,KS=ks)

validation={n:metrics(data[n]['gaps']) for n in data}

summary={
 'contact_atom_mass':chi,
 'representative_50k_pair_count':pair50eq,
 'representative_50k_contact_count':chi*pair50eq,
 'untapered_gap_fit':{
   'fit':'single log-density cubic spline in u=asinh(x/1e-6), equal-realization pooled gaps, no taper',
   'smoothing_s':S,
   'merged_bins':int(len(xc)),
   'p_g_0minus_limit_approx':float(pg[-1]),
   'dlogp_du_0minus_approx':float(dlogdu[-1]),
   'dpdx_0minus_approx':float(dpdx[-1]),
   'continuous_monotone_onset_x':x_monotone_onset,
   'continuous_monotone_onset_u':u_monotone_onset,
   'adaptive_discrete_monotone_onset_x':x_discrete_monotone_onset,
   'negative_derivative_points_x_gt_minus1e3':violations_near0,
   'negative_derivative_points_x_gt_minus1e4':violations_near1e4,
   'negative_derivative_points_x_gt_minus1e5':violations_near1e5,
   'validation':validation
 }
}
(ROOT/'untapered_gap_branch_audit.json').write_text(json.dumps(summary,indent=2))
np.savetxt(ROOT/'untapered_gap_fit.csv',np.c_[xg,pg,dpdx,dlogdu],
           delimiter=',',header='x,p_gap,dpdx,dlogp_du',comments='')

plt.rcParams.update({'figure.facecolor':'white','axes.facecolor':'white','savefig.facecolor':'white',
'font.family':'serif','font.serif':['Times New Roman','Times','DejaVu Serif'],'mathtext.fontset':'dejavuserif',
'font.size':9.0,'axes.labelsize':10.2,'xtick.labelsize':8.2,'ytick.labelsize':8.7,'legend.fontsize':7.0,
'axes.linewidth':0.85,'xtick.major.width':0.85,'ytick.major.width':0.85,'xtick.minor.width':0.65,'ytick.minor.width':0.65,
'xtick.direction':'in','ytick.direction':'in','pdf.fonttype':42})

fig,ax=plt.subplots(figsize=(6.8,3.5))
for n,mk in zip(VALID+['raw400k'],['o','s','^','d']):
    c,_=np.histogram(data[n]['gaps'],bins=edges)
    den=(c/c.sum())/np.diff(edges); cen=0.5*(edges[:-1]+edges[1:]); m=den>0
    ax.plot(cen[m],den[m],ls='none',marker=mk,mfc='none',ms=2,label=n)
ax.plot(xg,pg,lw=2,label='untapered pooled gap fit')
ax.axvline(0,ls=':',lw=.8)
ax.set_xscale('symlog',linthresh=1e-8,linscale=1,base=10); ax.set_yscale('log')
ax.set_xlim(-0.5,1e-8)
ax.set_xlabel(r'$\gamma/(2R)$'); ax.set_ylabel(r'$p_g(x\mid x<0)$')
ax.legend(frameon=False,ncol=2)
fig.tight_layout(pad=.25)
fig.savefig(ROOT/'Untapered_gap_fit_vs_discrete_Rodrigues.png',dpi=300,bbox_inches='tight',pad_inches=.02)
fig.savefig(ROOT/'Untapered_gap_fit_vs_discrete_Rodrigues.pdf',bbox_inches='tight',pad_inches=.02)
plt.close(fig)

fig,ax=plt.subplots(figsize=(3.3,3.0))
m=xg>-1e-3
ax.plot(xg[m],pg[m],lw=1.8)
if np.isfinite(x_monotone_onset): ax.axvline(x_monotone_onset,ls='--',lw=.9,label='monotone onset')
ax.axvline(0,ls=':',lw=.8)
ax.set_xscale('symlog',linthresh=1e-8,linscale=1,base=10); ax.set_yscale('log')
ax.set_xlim(-1e-3,1e-8)
ax.set_xlabel(r'$\gamma/(2R)$'); ax.set_ylabel(r'$p_g$')
if np.isfinite(x_monotone_onset): ax.legend(frameon=False)
fig.tight_layout(pad=.25)
fig.savefig(ROOT/'Untapered_gap_near_contact_Rodrigues.png',dpi=300,bbox_inches='tight',pad_inches=.02)
fig.savefig(ROOT/'Untapered_gap_near_contact_Rodrigues.pdf',bbox_inches='tight',pad_inches=.02)
plt.close(fig)

print('UNTAPERED_GAP_AUDIT='+json.dumps(summary,sort_keys=True),flush=True)
print('UNTAPERED_GAP_RESULT_DIR='+str(ROOT),flush=True)

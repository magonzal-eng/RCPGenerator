from pathlib import Path
import json, math
import numpy as np
from scipy.interpolate import UnivariateSpline
from scipy.integrate import cumulative_trapezoid
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FixedFormatter, NullFormatter

ROOT=Path('/scratch/gautschi/gonza226/pmi_threeway_raw_tolerance_20261002')
ROOT.mkdir(parents=True,exist_ok=True)
P400=Path('/scratch/gautschi/gonza226/pmi_400k_raw_noadjust_20261002')
P50=Path('/scratch/gautschi/gonza226/pmi_raw50k_original_20261002')
PT=Path('/scratch/gautschi/gonza226/pmi_raw50k_tightov_20261002')
HERE=Path(__file__).resolve().parent
MIN_COUNT=50
X0=1e-6
S=32.0

required=[
 P400/'pair_distribution_400k_as_generated.npz',
 P400/'packing_400k_as_generated.npz',
 P50/'pair_distribution_50k_raw_original.npz',
 P50/'packing_50k_raw_original.npz',
 P50/'raw50k_provenance.json',
 PT/'pair_distribution_50k_tighter_overlap.npz',
 PT/'packing_50k_tighter_overlap.npz',
 PT/'metadata.json',
 HERE/'reference_50k_radial_counts.json',
]
missing=[str(p) for p in required if not p.exists()]
if missing: raise FileNotFoundError('missing prerequisites: '+json.dumps(missing))

ref=json.loads((HERE/'reference_50k_radial_counts.json').read_text())
q_edges=np.asarray(ref['edges'],float)

def load_dataset(name,N,pair_path,packing_path,meta_path=None):
    z=np.load(pair_path); q=np.asarray(z['q'],float)
    pk=np.load(packing_path); phi=float(np.asarray(pk['phi']).reshape(-1)[0])
    x=np.asarray(pk['positions'],float); box=np.asarray(pk['box'],float); diam=np.asarray(pk['diameters'],float)
    contacts=int(np.count_nonzero(q<1.0))
    counts,_=np.histogram(q,bins=q_edges)
    g=1.0-q
    meta={}
    if meta_path and Path(meta_path).exists(): meta=json.loads(Path(meta_path).read_text())
    return dict(name=name,N=N,q=q,g=g,counts=counts,phi=phi,x=x,box=box,diam=diam,contacts=contacts,meta=meta)

D=[
 load_dataset('raw400k',400000,P400/'pair_distribution_400k_as_generated.npz',P400/'packing_400k_as_generated.npz',P400/'summary.json'),
 load_dataset('raw50k',50000,P50/'pair_distribution_50k_raw_original.npz',P50/'packing_50k_raw_original.npz',P50/'raw50k_provenance.json'),
 load_dataset('tight50k',50000,PT/'pair_distribution_50k_tighter_overlap.npz',PT/'packing_50k_tighter_overlap.npz',PT/'metadata.json'),
]
by={d['name']:d for d in D}

# Provenance guards.
if abs(by['raw50k']['phi']-0.6451654159495952)>5e-13 or by['raw50k']['contacts']!=145987:
    raise RuntimeError('raw50k provenance guard failed')
if abs(float(by['tight50k']['meta'].get('RCP_OSC_OV_THRESH',np.nan))-5e-5)>1e-15:
    raise RuntimeError('tight50k overlap-threshold guard failed')
if abs(float(by['tight50k']['meta'].get('tolerance_ratio',np.nan))-0.1)>1e-15:
    raise RuntimeError('tight50k tolerance ratio guard failed')

def merge_side(g_edges,counts,idx,total):
    groups=[]; start=int(idx[0]); acc=0
    for j in idx:
        acc+=int(counts[j])
        if acc>=MIN_COUNT:
            groups.append((start,int(j),acc)); start=int(j)+1; acc=0
    if start<=int(idx[-1]):
        if groups:
            a,b,c=groups[-1]; groups[-1]=(a,int(idx[-1]),c+acc)
        else: groups=[(start,int(idx[-1]),acc)]
    cen=[]; dens=[]; n=[]
    for a,b,c in groups:
        lo,hi=float(g_edges[a]),float(g_edges[b+1])
        if hi==0: cc=lo/2
        elif lo==0: cc=hi/2
        elif hi<0: cc=-np.sqrt(abs(lo*hi))
        else: cc=np.sqrt(lo*hi)
        cen.append(cc); dens.append(c/(total*(hi-lo))); n.append(c)
    return np.asarray(cen),np.asarray(dens),np.asarray(n)

def merged(d):
    g_edges=1.0-q_edges[::-1]; c=d['counts'][::-1]
    nz=np.flatnonzero(c); lo,hi=int(nz[0]),int(nz[-1])
    total=int(c[lo:hi+1].sum()); ii=np.arange(len(c))
    neg=np.where((ii>=lo)&(ii<=hi)&(g_edges[1:]<=0))[0]
    pos=np.where((ii>=lo)&(ii<=hi)&(g_edges[:-1]>=0))[0]
    parts=[]
    if len(neg): parts.append(merge_side(g_edges,c,neg,total))
    if len(pos): parts.append(merge_side(g_edges,c,pos,total))
    gd=np.concatenate([p[0] for p in parts]); pd=np.concatenate([p[1] for p in parts]); cd=np.concatenate([p[2] for p in parts])
    o=np.argsort(gd)
    return gd[o],pd[o],cd[o],total,(float(g_edges[lo]),float(g_edges[hi+1]))

def fit(gd,pd,cd,s=S,n=60000):
    u=np.arcsinh(gd/X0); o=np.argsort(u)
    sp=UnivariateSpline(u[o],np.log(pd[o]),w=np.sqrt(cd[o]),s=len(u)*s,k=3)
    ug=np.linspace(u[o][0],u[o][-1],n); g=X0*np.sinh(ug)
    raw=np.exp(np.clip(sp(ug),-745,700)); p=raw/np.trapezoid(raw,g)
    return g,p,ug,sp

def roots(x,y):
    idx=np.where(np.sign(y[:-1])*np.sign(y[1:])<0)[0]; out=[]
    for i in idx: out.append(float(x[i]-y[i]*(x[i+1]-x[i])/(y[i+1]-y[i])))
    return np.asarray(out)

def curvature(gd,pd,cd):
    rr={}
    for sm in (16.,32.,64.):
        g,p,u,sp=fit(gd,pd,cd,sm,50000)
        rr[sm]=roots(u,sp.derivative(2)(u))
    out=[]
    for r in rr[32.]:
        d16=float(np.min(np.abs(rr[16.]-r))) if len(rr[16.]) else np.inf
        d64=float(np.min(np.abs(rr[64.]-r))) if len(rr[64.]) else np.inf
        if d16<=0.25 and d64<=0.25:
            out.append({'u':float(r),'gamma':float(X0*np.sinh(r)),'du_s16':d16,'du_s64':d64})
    return out

def peak(g,p):
    i=int(np.argmax(p)); return {'gamma':float(g[i]),'height':float(p[i])}
def dpeak(g,p,c):
    i=int(np.argmax(p)); return {'gamma':float(g[i]),'height':float(p[i]),'count':int(c[i])}

for d in D:
    gd,pd,cd,total,support=merged(d); G,P,U,SP=fit(gd,pd,cd)
    d.update(gd=gd,pd=pd,cd=cd,total=total,support=support,G=G,P=P,
             peak=peak(G,P),dpeak=dpeak(gd,pd,cd),curvature=curvature(gd,pd,cd))

def pairwise(a,b):
    A=by[a]; B=by[b]
    pa=A['counts']/A['counts'].sum(); pb=B['counts']/B['counts'].sum()
    emp_l1=float(np.sum(np.abs(pa-pb))); emp_ks=float(np.max(np.abs(np.cumsum(pa)-np.cumsum(pb))))
    lo=min(A['G'].min(),B['G'].min()); hi=max(A['G'].max(),B['G'].max())
    x=np.linspace(lo,hi,250000)
    fa=np.interp(x,A['G'],A['P'],left=0,right=0); fb=np.interp(x,B['G'],B['P'],left=0,right=0)
    fa/=np.trapezoid(fa,x); fb/=np.trapezoid(fb,x)
    l1=float(np.trapezoid(np.abs(fa-fb),x))
    ca=np.r_[0,cumulative_trapezoid(fa,x)]; cb=np.r_[0,cumulative_trapezoid(fb,x)]
    ks=float(np.max(np.abs(ca-cb)))
    sa=50000/A['N']; sb=50000/B['N']
    scaled_l1=float(np.sum(np.abs(A['counts']*sa-B['counts']*sb))/np.sum(A['counts']*sa))
    return {
      'empirical_pdf_L1':emp_l1,'empirical_total_variation':0.5*emp_l1,'empirical_CDF_KS':emp_ks,
      'fit_pdf_L1':l1,'fit_total_variation':0.5*l1,'fit_CDF_KS':ks,
      'scaled_50k_basis_count_relative_L1':scaled_l1,
      'contact_fraction_a':A['contacts']/A['total'],'contact_fraction_b':B['contacts']/B['total'],
      'contact_fraction_absolute_difference':B['contacts']/B['total']-A['contacts']/A['total'],
      'phi_a':A['phi'],'phi_b':B['phi'],'phi_difference':B['phi']-A['phi'],
      'peak_a':A['peak'],'peak_b':B['peak'],
    }

# Same-seed coordinate sensitivity raw50k -> tight50k.
a=by['raw50k']; b=by['tight50k']
if a['x'].shape != b['x'].shape: raise RuntimeError('50k coordinate shape mismatch')
box=a['box']
dx=b['x']-a['x']; dx-=box*np.rint(dx/box)
disp=np.linalg.norm(dx,axis=1)
dref=float(np.mean(a['diam']))
coord_sens={
 'rms_displacement_over_d':float(np.sqrt(np.mean(disp**2))/dref),
 'mean_displacement_over_d':float(np.mean(disp)/dref),
 'max_displacement_over_d':float(np.max(disp)/dref),
 'rms_coordinate_component_over_d':float(np.sqrt(np.mean(dx**2))/dref),
 'diameter_raw50k':float(np.mean(a['diam'])),'diameter_tight50k':float(np.mean(b['diam'])),
 'diameter_relative_difference':float(np.mean(b['diam'])/np.mean(a['diam'])-1),
}

def dataset_summary(d):
    g=d['g']
    return {
      'N':d['N'],'phi':d['phi'],'retained_pairs':d['total'],'strict_contacts_q_lt_1':d['contacts'],
      'strict_contact_fraction':d['contacts']/d['total'],
      'max_fractional_overlap':float(max(0.0,np.max(g))),
      'contacts_with_overlap_gt_5e-5':int(np.count_nonzero(g>5e-5)),
      'contacts_with_overlap_gt_5e-4':int(np.count_nonzero(g>5e-4)),
      'contacts_in_0_to_5e-5':int(np.count_nonzero((g>0)&(g<=5e-5))),
      'contacts_in_5e-5_to_5e-4':int(np.count_nonzero((g>5e-5)&(g<=5e-4))),
      'support_gamma':list(d['support']),'merged_bins':int(len(d['gd'])),
      'fit_peak':d['peak'],'discrete_merged_peak':d['dpeak'],'robust_log_density_curvature':d['curvature'],
    }

summary={
 'scope':'three-way raw packing comparison: 400k baseline, historical raw 50k, same-seed 50k with 10x tighter maximum-overlap convergence threshold',
 'generator_commit':'974d703a655b337b93379dbcd1a6eef47386d674',
 'baseline_overlap_convergence_threshold':5e-4,
 'tighter_overlap_convergence_threshold':5e-5,
 'overlap_classification_note':'Core force/contact classification is exact dist < Ri+Rj; RCP_OSC_OV_THRESH is the maximum fractional-overlap convergence threshold, not a fuzzy contact classification band.',
 'datasets':{d['name']:dataset_summary(d) for d in D},
 'pairwise':{
   'raw50k_vs_raw400k':pairwise('raw50k','raw400k'),
   'tight50k_vs_raw400k':pairwise('tight50k','raw400k'),
   'raw50k_vs_tight50k':pairwise('raw50k','tight50k'),
 },
 'same_seed_50k_coordinate_sensitivity':coord_sens,
 'fit_spec':{'signed_variable':'gamma/(2R)=1-q','min_actual_observations_per_merged_bin':MIN_COUNT,
             'single_positive_C2_signed_domain_fit':True,'u':'asinh((gamma/(2R))/1e-6)',
             'weights':'sqrt(merged bin count)','normalization':'unit integral over populated signed support',
             'official_smoothing_multiplier':16.0,'scipy_smoothing_s':S},
}
(ROOT/'summary_threeway.json').write_text(json.dumps(summary,indent=2))

centers=0.5*(q_edges[:-1]+q_edges[1:])
np.savetxt(ROOT/'threeway_scaled_counts.csv',
    np.c_[centers,by['raw400k']['counts']*0.125,by['raw50k']['counts'],by['tight50k']['counts']],
    delimiter=',',header='q_center,raw400k_scaled_1_over_8,raw50k,tight50k',comments='')

# Rodrigues-derived styles.
plt.rcParams.update({'figure.facecolor':'white','axes.facecolor':'white','savefig.facecolor':'white',
'font.family':'serif','font.serif':['Times New Roman','Times','DejaVu Serif'],'mathtext.fontset':'dejavuserif',
'font.size':9.0,'axes.labelsize':10.2,'xtick.labelsize':8.2,'ytick.labelsize':8.7,'legend.fontsize':7.0,
'axes.linewidth':0.85,'xtick.major.width':0.85,'ytick.major.width':0.85,'xtick.minor.width':0.65,'ytick.minor.width':0.65,
'xtick.direction':'in','ytick.direction':'in','lines.linewidth':1.55,'pdf.fonttype':42,'ps.fonttype':42})
ticks=[-1e-1,-1e-3,-1e-5,0,1e-5,1e-4,1e-3]
labels=[r'$-10^{-1}$',r'$-10^{-3}$',r'$-10^{-5}$','0',r'$10^{-5}$',r'$10^{-4}$',r'$10^{-3}$']

# Full-width 3-way figure.
fig,ax=plt.subplots(figsize=(6.8,3.2))
spec=[('raw400k','s','--','raw 400k'),('raw50k','o','-','raw 50k'),('tight50k','^','-.','50k, 10x tighter')]
for name,mk,ls,label in spec:
    d=by[name]
    ax.plot(d['gd'],d['pd'],ls='none',marker=mk,mfc='none',ms=2.5,label=label+' discrete')
    ax.plot(d['G'],d['P'],ls=ls,lw=1.55,label=label+' 16x fit')
ax.axvline(0,color='.45',ls=':',lw=.85)
ax.set_xscale('symlog',linthresh=1e-6,linscale=1,base=10); ax.set_yscale('log')
ax.set_xlim(-0.5,max(1e-3,max(d['support'][1] for d in D)))
positive=np.concatenate([np.r_[d['pd'],d['P']] for d in D]); positive=positive[positive>0]
ax.set_ylim(max(1e-3,float(positive.min())*.65),float(positive.max())*1.4)
ax.xaxis.set_major_locator(FixedLocator(ticks)); ax.xaxis.set_major_formatter(FixedFormatter(labels)); ax.xaxis.set_minor_formatter(NullFormatter())
ax.set_xlabel(r'$\gamma/(2R)$'); ax.set_ylabel(r'$P\!\left(\gamma/(2R)\right)$')
ax.tick_params(which='major',top=True,right=True,length=4); ax.tick_params(which='minor',top=True,right=True,length=2.5)
ax.grid(False)
for s in ax.spines.values(): s.set_linewidth(.85)
ax.legend(frameon=False,ncol=2,loc='best')
fig.tight_layout(pad=.25)
fig.savefig(ROOT/'Pgamma_three_packings_Rodrigues_fullwidth.png',dpi=300,bbox_inches='tight',pad_inches=.02)
fig.savefig(ROOT/'Pgamma_three_packings_Rodrigues_fullwidth.pdf',bbox_inches='tight',pad_inches=.02)
plt.close(fig)

# Half-width same-seed tolerance sensitivity.
fig,ax=plt.subplots(figsize=(3.3,3.3))
for name,mk,ls,label in [('raw50k','o','-','raw 50k'),('tight50k','^','--','10x tighter 50k')]:
    d=by[name]
    ax.plot(d['gd'],d['pd'],ls='none',marker=mk,mfc='none',ms=2.6,label=label+' discrete')
    ax.plot(d['G'],d['P'],ls=ls,lw=1.6,label=label+' 16x fit')
ax.axvline(0,color='.45',ls=':',lw=.85)
ax.set_xscale('symlog',linthresh=1e-6,linscale=1,base=10); ax.set_yscale('log')
ax.set_xlim(-0.5,max(1e-3,a['support'][1],b['support'][1]))
pos=np.r_[a['pd'],a['P'],b['pd'],b['P']]; pos=pos[pos>0]
ax.set_ylim(max(1e-3,float(pos.min())*.65),float(pos.max())*1.4)
ax.xaxis.set_major_locator(FixedLocator(ticks)); ax.xaxis.set_major_formatter(FixedFormatter(labels)); ax.xaxis.set_minor_formatter(NullFormatter())
ax.set_xlabel(r'$\gamma/(2R)$'); ax.set_ylabel(r'$P\!\left(\gamma/(2R)\right)$')
ax.tick_params(which='major',top=True,right=True,length=4); ax.tick_params(which='minor',top=True,right=True,length=2.5)
ax.grid(False)
for s in ax.spines.values(): s.set_linewidth(.85)
ax.legend(frameon=False,loc='best')
fig.tight_layout(pad=.25)
fig.savefig(ROOT/'Pgamma_50k_tolerance_sensitivity_Rodrigues_halfwidth.png',dpi=300,bbox_inches='tight',pad_inches=.02)
fig.savefig(ROOT/'Pgamma_50k_tolerance_sensitivity_Rodrigues_halfwidth.pdf',bbox_inches='tight',pad_inches=.02)
plt.close(fig)
print('THREEWAY_SUMMARY='+json.dumps(summary,sort_keys=True),flush=True)
print('THREEWAY_RESULT_DIR='+str(ROOT),flush=True)

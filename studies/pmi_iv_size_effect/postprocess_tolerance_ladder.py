from pathlib import Path
import json, math
import numpy as np
from scipy.interpolate import UnivariateSpline
from scipy.integrate import cumulative_trapezoid
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FixedFormatter, NullFormatter

ROOT=Path('/scratch/gautschi/gonza226/pmi_tolerance_ladder_post_20261002')
ROOT.mkdir(parents=True,exist_ok=True)

cases = [
    dict(name='raw400k', label='400k raw', N=400000,
         pair=Path('/scratch/gautschi/gonza226/pmi_400k_raw_noadjust_20261002/pair_distribution_400k_as_generated.npz'),
         packing=Path('/scratch/gautschi/gonza226/pmi_400k_raw_noadjust_20261002/packing_400k_as_generated.npz'),
         tol=5e-4, scale=0.125),
    dict(name='raw50k', label=r'50k $5\\times10^{-4}$', N=50000,
         pair=Path('/scratch/gautschi/gonza226/pmi_raw50k_original_20261002/pair_distribution_50k_raw_original.npz'),
         packing=Path('/scratch/gautschi/gonza226/pmi_raw50k_original_20261002/packing_50k_raw_original.npz'),
         tol=5e-4, scale=1.0),
    dict(name='tol5e-5', label=r'50k $5\\times10^{-5}$', N=50000,
         pair=Path('/scratch/gautschi/gonza226/pmi_raw50k_tightov_20261002/pair_distribution_50k_tighter_overlap.npz'),
         packing=Path('/scratch/gautschi/gonza226/pmi_raw50k_tightov_20261002/packing_50k_tighter_overlap.npz'),
         tol=5e-5, scale=1.0),
    dict(name='tol5e-6', label=r'50k $5\\times10^{-6}$', N=50000,
         pair=Path('/scratch/gautschi/gonza226/pmi_raw50k_tol5e-6_20261002/pair_distribution_50k_tol5e-6.npz'),
         packing=Path('/scratch/gautschi/gonza226/pmi_raw50k_tol5e-6_20261002/packing_50k_tol5e-6.npz'),
         tol=5e-6, scale=1.0),
    dict(name='tol5e-7', label=r'50k $5\\times10^{-7}$', N=50000,
         pair=Path('/scratch/gautschi/gonza226/pmi_raw50k_tol5e-7_20261002/pair_distribution_50k_tol5e-7.npz'),
         packing=Path('/scratch/gautschi/gonza226/pmi_raw50k_tol5e-7_20261002/packing_50k_tol5e-7.npz'),
         tol=5e-7, scale=1.0),
    dict(name='tol5e-8', label=r'50k $5\\times10^{-8}$', N=50000,
         pair=Path('/scratch/gautschi/gonza226/pmi_raw50k_tol5e-8_20261002/pair_distribution_50k_tol5e-8.npz'),
         packing=Path('/scratch/gautschi/gonza226/pmi_raw50k_tol5e-8_20261002/packing_50k_tol5e-8.npz'),
         tol=5e-8, scale=1.0),
]
missing=[str(p) for c in cases for p in [c['pair'],c['packing']] if not p.exists()]
if missing: raise FileNotFoundError('missing prerequisites: '+json.dumps(missing))

# Common signed logarithmic source bins. Resolve the tightest case by >100x below 5e-8.
# Gap side spans to -0.5; contact side safely spans to 1e-3.
eps=1e-10
neg_mag=np.geomspace(eps,0.5,360)
pos=np.geomspace(eps,1e-3,280)
x_edges=np.unique(np.r_[-neg_mag[::-1],0.0,pos])
x_edges.sort()

MIN_COUNT=50
X0=1e-6
S=32.0

def merge_side(edges,counts,idx,total):
    groups=[]; start=int(idx[0]); acc=0
    for j in idx:
        acc+=int(counts[j])
        if acc>=MIN_COUNT:
            groups.append((start,int(j),acc)); start=int(j)+1; acc=0
    if start<=int(idx[-1]):
        if groups:
            a,b,c=groups[-1]; groups[-1]=(a,int(idx[-1]),c+acc)
        else: groups=[(start,int(idx[-1]),acc)]
    cen=[]; dens=[]; n=[]; bins=[]
    for a,b,c in groups:
        lo=float(edges[a]); hi=float(edges[b+1])
        if hi==0: cc=lo/2
        elif lo==0: cc=hi/2
        elif hi<0: cc=-math.sqrt(abs(lo*hi))
        else: cc=math.sqrt(lo*hi)
        cen.append(cc); dens.append(c/(total*(hi-lo))); n.append(c); bins.append((lo,hi))
    return np.asarray(cen),np.asarray(dens),np.asarray(n),bins

def merge_counts(edges,counts):
    nz=np.flatnonzero(counts)
    lo,hi=int(nz[0]),int(nz[-1]); total=int(counts[lo:hi+1].sum())
    ii=np.arange(len(counts))
    neg=np.where((ii>=lo)&(ii<=hi)&(edges[1:]<=0))[0]
    posi=np.where((ii>=lo)&(ii<=hi)&(edges[:-1]>=0))[0]
    parts=[]
    if len(neg): parts.append(merge_side(edges,counts,neg,total))
    if len(posi): parts.append(merge_side(edges,counts,posi,total))
    gd=np.concatenate([p[0] for p in parts]); pd=np.concatenate([p[1] for p in parts]); cd=np.concatenate([p[2] for p in parts])
    bins=sum([p[3] for p in parts],[])
    o=np.argsort(gd)
    return gd[o],pd[o],cd[o],[bins[i] for i in o],total,(float(edges[lo]),float(edges[hi+1]))

def fit(gd,pd,cd,s=S,n=80000):
    u=np.arcsinh(gd/X0); o=np.argsort(u)
    sp=UnivariateSpline(u[o],np.log(pd[o]),w=np.sqrt(cd[o]),s=len(u)*s,k=3)
    ug=np.linspace(u[o][0],u[o][-1],n)
    x=X0*np.sinh(ug)
    raw=np.exp(np.clip(sp(ug),-745,700))
    p=raw/np.trapezoid(raw,x)
    return x,p,ug,sp

def roots(x,y):
    idx=np.where(np.sign(y[:-1])*np.sign(y[1:])<0)[0]; out=[]
    for i in idx:
        out.append(float(x[i]-y[i]*(x[i+1]-x[i])/(y[i+1]-y[i])))
    return np.asarray(out)

def robust_curvature(gd,pd,cd):
    rr={}
    for sm in (16.,32.,64.):
        X,P,U,SP=fit(gd,pd,cd,sm,60000)
        rr[sm]=roots(U,SP.derivative(2)(U))
    out=[]
    for r in rr[32.]:
        d16=float(np.min(np.abs(rr[16.]-r))) if len(rr[16.]) else np.inf
        d64=float(np.min(np.abs(rr[64.]-r))) if len(rr[64.]) else np.inf
        if d16<=0.25 and d64<=0.25:
            out.append({'u':float(r),'x':float(X0*np.sinh(r)),'du_s16':d16,'du_s64':d64})
    return out

def peak(X,P):
    i=int(np.argmax(P)); return {'x':float(X[i]),'height':float(P[i])}

data={}
for c in cases:
    z=np.load(c['pair']); q=np.asarray(z['q'],float); x=1.0-q
    pk=np.load(c['packing']); phi=float(np.asarray(pk['phi']).reshape(-1)[0])
    counts,_=np.histogram(x,bins=x_edges)
    gd,pd,cd,bins,total,support=merge_counts(x_edges,counts)
    X,P,U,SP=fit(gd,pd,cd)
    positive=x[x>0]
    d=dict(c)
    d.update(q=q,x=x,counts=counts,phi=phi,gd=gd,pd=pd,cd=cd,bins=bins,total=total,support=support,X=X,P=P,
             contacts=int(np.count_nonzero(x>0)), max_overlap=float(np.max(positive)) if len(positive) else 0.0,
             peak=peak(X,P), curvature=robust_curvature(gd,pd,cd))
    data[c['name']]=d

def pairwise(a,b):
    A=data[a]; B=data[b]
    pa=A['counts']/A['counts'].sum(); pb=B['counts']/B['counts'].sum()
    emp_l1=float(np.sum(np.abs(pa-pb))); emp_ks=float(np.max(np.abs(np.cumsum(pa)-np.cumsum(pb))))
    lo=min(A['X'].min(),B['X'].min()); hi=max(A['X'].max(),B['X'].max())
    grid=np.linspace(lo,hi,300000)
    fa=np.interp(grid,A['X'],A['P'],left=0,right=0); fb=np.interp(grid,B['X'],B['P'],left=0,right=0)
    fa/=np.trapezoid(fa,grid); fb/=np.trapezoid(fb,grid)
    fit_l1=float(np.trapezoid(np.abs(fa-fb),grid))
    ca=np.r_[0,cumulative_trapezoid(fa,grid)]; cb=np.r_[0,cumulative_trapezoid(fb,grid)]
    fit_ks=float(np.max(np.abs(ca-cb)))
    sa=50000/A['N']; sb=50000/B['N']
    scaled_l1=float(np.sum(np.abs(A['counts']*sa-B['counts']*sb))/np.sum(A['counts']*sa))
    return {
      'empirical_L1':emp_l1,'empirical_TV':0.5*emp_l1,'empirical_KS':emp_ks,
      'fit_L1':fit_l1,'fit_TV':0.5*fit_l1,'fit_KS':fit_ks,
      'scaled_50k_basis_count_relative_L1':scaled_l1,
      'phi_a':A['phi'],'phi_b':B['phi'],'phi_difference':B['phi']-A['phi'],
      'contact_fraction_a':A['contacts']/A['total'],'contact_fraction_b':B['contacts']/B['total'],
      'peak_a':A['peak'],'peak_b':B['peak'],
    }

tol_names=['raw50k','tol5e-5','tol5e-6','tol5e-7','tol5e-8']
pairwise_out={}
for i in range(len(tol_names)-1):
    pairwise_out[f'{tol_names[i]}_vs_{tol_names[i+1]}']=pairwise(tol_names[i],tol_names[i+1])
for name in tol_names:
    pairwise_out[f'{name}_vs_raw400k']=pairwise(name,'raw400k')

# Same-seed coordinate sensitivity relative to baseline raw50k.
base=np.load(cases[1]['packing']); xbase=np.asarray(base['positions'],float); box=np.asarray(base['box'],float); dbase=float(np.mean(base['diameters']))
coord={}
for c in cases[2:]:
    pk=np.load(c['packing']); xx=np.asarray(pk['positions'],float)
    dx=xx-xbase; dx-=box*np.rint(dx/box); disp=np.linalg.norm(dx,axis=1)
    coord[c['name']]={
      'rms_displacement_over_d':float(np.sqrt(np.mean(disp**2))/dbase),
      'mean_displacement_over_d':float(np.mean(disp)/dbase),
      'max_displacement_over_d':float(np.max(disp)/dbase),
    }

thresholds=[5e-4,5e-5,5e-6,5e-7,5e-8,5e-9]
def dataset_summary(d):
    x=d['x']
    bands={}
    for hi,lo in zip(thresholds[:-1],thresholds[1:]):
        bands[f'overlap_{lo:g}_to_{hi:g}']=int(np.count_nonzero((x>lo)&(x<=hi)))
    bands[f'overlap_0_to_{thresholds[-1]:g}']=int(np.count_nonzero((x>0)&(x<=thresholds[-1])))
    return {
      'N':d['N'],'requested_overlap_threshold':d['tol'],'phi':d['phi'],
      'retained_pairs':d['total'],'strict_contacts':d['contacts'],'strict_contact_fraction':d['contacts']/d['total'],
      'max_fractional_overlap_rebuilt':d['max_overlap'],'max_over_threshold_ratio':d['max_overlap']/d['tol'],
      'overlap_bands':bands,'support':list(d['support']),'merged_bins':int(len(d['gd'])),
      'fit_peak':d['peak'],'robust_curvature':d['curvature'],
      'total_pairs_50k_equivalent':d['total']*(50000/d['N']),
      'contacts_50k_equivalent':d['contacts']*(50000/d['N']),
    }

summary={
 'scope':'PMI raw packing overlap-tolerance ladder plus 400k reference',
 'generator_commit':'974d703a655b337b93379dbcd1a6eef47386d674',
 'contact_definition':'strict geometric x=gamma/(2R)=1-q>0; no fuzzy contact band',
 'tolerance_definition':'RCP_OSC_OV_THRESH is maximum fractional-overlap convergence threshold',
 'datasets':{name:dataset_summary(d) for name,d in data.items()},
 'pairwise':pairwise_out,
 'same_seed_coordinate_sensitivity_from_raw50k':coord,
 'fit_spec':{
   'source_bins':'common signed logarithmic bins resolving |x| down to 1e-10',
   'min_actual_observations_per_merged_bin':MIN_COUNT,
   'single_positive_C2_signed_domain_fit':True,
   'u':'asinh(x/1e-6)','weights':'sqrt(merged bin count)',
   'normalization':'unit integral over populated support',
   'official_smoothing_multiplier':16.0,'scipy_smoothing_s':S
 }
}
(ROOT/'summary_tolerance_ladder.json').write_text(json.dumps(summary,indent=2))

# Common source-bin counts, scaled to 50k basis.
centers=0.5*(x_edges[:-1]+x_edges[1:])
arr=[centers]
headers=['x_center']
for c in cases:
    d=data[c['name']]
    arr.append(d['counts']*(50000/d['N']))
    headers.append(c['name']+'_count_50k_equiv')
np.savetxt(ROOT/'tolerance_ladder_counts_50k_equiv.csv',np.column_stack(arr),delimiter=',',header=','.join(headers),comments='')

plt.rcParams.update({'figure.facecolor':'white','axes.facecolor':'white','savefig.facecolor':'white',
'font.family':'serif','font.serif':['Times New Roman','Times','DejaVu Serif'],'mathtext.fontset':'dejavuserif',
'font.size':9.0,'axes.labelsize':10.2,'xtick.labelsize':8.2,'ytick.labelsize':8.7,'legend.fontsize':7.0,
'axes.linewidth':0.85,'xtick.major.width':0.85,'ytick.major.width':0.85,'xtick.minor.width':0.65,'ytick.minor.width':0.65,
'xtick.direction':'in','ytick.direction':'in','lines.linewidth':1.55,'pdf.fonttype':42,'ps.fonttype':42})

def setup(ax,ylabel):
    ax.axvline(0,color='.45',ls=':',lw=.85)
    ax.set_xscale('symlog',linthresh=1e-8,linscale=1,base=10); ax.set_yscale('log')
    ax.set_xlim(-0.5,1e-3)
    ticks=[-1e-1,-1e-3,-1e-5,-1e-7,0,1e-8,1e-7,1e-6,1e-5,1e-4]
    labels=[r'$-10^{-1}$',r'$-10^{-3}$',r'$-10^{-5}$',r'$-10^{-7}$','0',r'$10^{-8}$',r'$10^{-7}$',r'$10^{-6}$',r'$10^{-5}$',r'$10^{-4}$']
    ax.xaxis.set_major_locator(FixedLocator(ticks)); ax.xaxis.set_major_formatter(FixedFormatter(labels)); ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel(r'$\gamma/(2R)$'); ax.set_ylabel(ylabel)
    ax.tick_params(which='major',top=True,right=True,length=4); ax.tick_params(which='minor',top=True,right=True,length=2.5)
    ax.grid(False)
    for s in ax.spines.values(): s.set_linewidth(.85)

# Normalized PDF tolerance ladder.
fig,ax=plt.subplots(figsize=(6.8,3.5))
styles=['-','--','-.',':',(0,(3,1,1,1)),(0,(5,1))]
for c,ls in zip(cases,styles):
    d=data[c['name']]
    ax.plot(d['X'],d['P'],ls=ls,lw=1.5,label=c['label'])
setup(ax,r'$P\!\left(\gamma/(2R)\right)$')
allp=np.concatenate([data[c['name']]['P'] for c in cases]); allp=allp[allp>0]
ax.set_ylim(max(1e-3,float(allp.min())*.5),float(allp.max())*1.5)
ax.legend(frameon=False,ncol=2,loc='best')
fig.tight_layout(pad=.25)
fig.savefig(ROOT/'Pgamma_tolerance_ladder_normalized_Rodrigues.png',dpi=300,bbox_inches='tight',pad_inches=.02)
fig.savefig(ROOT/'Pgamma_tolerance_ladder_normalized_Rodrigues.pdf',bbox_inches='tight',pad_inches=.02)
plt.close(fig)

# 50k-equivalent population density M_50eq = P * retained pairs * 50k/N.
fig,ax=plt.subplots(figsize=(6.8,3.5))
allm=[]
for c,ls in zip(cases,styles):
    d=data[c['name']]
    M=d['P']*d['total']*(50000/d['N'])
    allm.append(M)
    ax.plot(d['X'],M,ls=ls,lw=1.5,label=c['label'])
setup(ax,r'$M_{50k}^{\rm eq}\!\left(\gamma/(2R)\right)$')
am=np.concatenate(allm); am=am[am>0]
ax.set_ylim(max(1e-1,float(am.min())*.5),float(am.max())*1.5)
ax.legend(frameon=False,ncol=2,loc='best')
fig.tight_layout(pad=.25)
fig.savefig(ROOT/'Mgamma_tolerance_ladder_50k_equiv_Rodrigues.png',dpi=300,bbox_inches='tight',pad_inches=.02)
fig.savefig(ROOT/'Mgamma_tolerance_ladder_50k_equiv_Rodrigues.pdf',bbox_inches='tight',pad_inches=.02)
plt.close(fig)

# Convergence summary vs tolerance.
tolvals=[]; phis=[]; contacts=[]; maxov=[]
for name in tol_names:
    d=data[name]; tolvals.append(d['tol']); phis.append(d['phi']); contacts.append(d['contacts']); maxov.append(d['max_overlap'])
np.savetxt(ROOT/'tolerance_convergence_summary.csv',np.c_[tolvals,phis,contacts,maxov],
           delimiter=',',header='requested_tol,phi,strict_contacts,max_overlap',comments='')
print('TOLERANCE_LADDER_SUMMARY='+json.dumps(summary,sort_keys=True),flush=True)
print('TOLERANCE_LADDER_RESULT_DIR='+str(ROOT),flush=True)

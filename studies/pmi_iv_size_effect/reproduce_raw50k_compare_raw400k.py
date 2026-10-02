from pathlib import Path
import json, time
import numpy as np
from scipy.spatial import cKDTree
from scipy.interpolate import UnivariateSpline
from scipy.integrate import cumulative_trapezoid
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FixedFormatter, NullFormatter
import rcpgenerator

N50=50_000
SEED50=20260922
PHI_INIT=0.05
QMAX=1.5
X0=1e-6
MIN_COUNT=50
SCIPY_S=32.0
EXPECTED_PHI50=0.64516541595
EXPECTED_CONTACTS50=145987
PHI_TOL=5e-10
ROOT=Path('/scratch/gautschi/gonza226/pmi_raw50k_original_20261002')
ROOT.mkdir(parents=True,exist_ok=True)
HERE=Path(__file__).resolve().parent
RAW400=Path('/scratch/gautschi/gonza226/pmi_400k_raw_noadjust_20261002/pair_distribution_400k_as_generated.npz')
SUM400=Path('/scratch/gautschi/gonza226/pmi_400k_raw_noadjust_20261002/summary.json')

def merge_side(g_edges, counts, idx, total):
    groups=[]; start=int(idx[0]); acc=0
    for j in idx:
        acc+=int(counts[j])
        if acc>=MIN_COUNT:
            groups.append((start,int(j),acc)); start=int(j)+1; acc=0
    if start<=int(idx[-1]):
        if groups:
            a,b,c=groups[-1]; groups[-1]=(a,int(idx[-1]),c+acc)
        else:
            groups=[(start,int(idx[-1]),acc)]
    cen=[]; dens=[]; n=[]
    for a,b,c in groups:
        lo,hi=float(g_edges[a]),float(g_edges[b+1])
        if hi==0: cc=lo/2
        elif lo==0: cc=hi/2
        elif hi<0: cc=-np.sqrt(abs(lo*hi))
        else: cc=np.sqrt(lo*hi)
        cen.append(cc); dens.append(c/(total*(hi-lo))); n.append(c)
    return np.asarray(cen),np.asarray(dens),np.asarray(n)

def merge_counts(q_edges,q_counts):
    g_edges=1.0-q_edges[::-1]
    counts=q_counts[::-1]
    nz=np.flatnonzero(counts)
    if not len(nz): raise RuntimeError('empty histogram')
    lo,hi=int(nz[0]),int(nz[-1])
    total=int(counts[lo:hi+1].sum())
    ii=np.arange(len(counts))
    neg=np.where((ii>=lo)&(ii<=hi)&(g_edges[1:]<=0))[0]
    pos=np.where((ii>=lo)&(ii<=hi)&(g_edges[:-1]>=0))[0]
    pieces=[]
    if len(neg): pieces.append(merge_side(g_edges,counts,neg,total))
    if len(pos): pieces.append(merge_side(g_edges,counts,pos,total))
    gd=np.concatenate([p[0] for p in pieces]); pd=np.concatenate([p[1] for p in pieces]); cd=np.concatenate([p[2] for p in pieces])
    o=np.argsort(gd)
    return gd[o],pd[o],cd[o],counts,(float(g_edges[lo]),float(g_edges[hi+1])),total

def fit(gd,pd,cd,s=SCIPY_S,ngrid=60000):
    u=np.arcsinh(gd/X0); o=np.argsort(u)
    sp=UnivariateSpline(u[o],np.log(pd[o]),w=np.sqrt(cd[o]),s=len(u)*s,k=3)
    ug=np.linspace(u[o][0],u[o][-1],ngrid)
    g=X0*np.sinh(ug); raw=np.exp(np.clip(sp(ug),-745,700))
    p=raw/np.trapezoid(raw,g)
    return g,p,ug,sp

def zero_crossings(x,y):
    idx=np.where(np.sign(y[:-1])*np.sign(y[1:])<0)[0]
    out=[]
    for i in idx:
        out.append(float(x[i]-y[i]*(x[i+1]-x[i])/(y[i+1]-y[i])))
    return np.asarray(out)

def robust_curvature(gd,pd,cd):
    roots={}
    for sm in (16.,32.,64.):
        g,p,u,sp=fit(gd,pd,cd,sm,50000)
        roots[sm]=zero_crossings(u,sp.derivative(2)(u))
    feats=[]
    for r in roots[32.]:
        d16=float(np.min(np.abs(roots[16.]-r))) if len(roots[16.]) else np.inf
        d64=float(np.min(np.abs(roots[64.]-r))) if len(roots[64.]) else np.inf
        if d16<=0.25 and d64<=0.25:
            feats.append({'u':float(r),'gamma':float(X0*np.sinh(r)),'du_s16':d16,'du_s64':d64})
    return feats

def match_features(a,b,max_du=0.25):
    if not a or not b: return []
    ub=np.array([x['u'] for x in b])
    out=[]
    for x in a:
        j=int(np.argmin(np.abs(ub-x['u']))); du=float(abs(ub[j]-x['u']))
        if du<=max_du: out.append({'gamma50':x['gamma'],'gamma400':b[j]['gamma'],'du':du})
    return out

def peak(g,p):
    i=int(np.argmax(p)); return {'gamma':float(g[i]),'height':float(p[i])}
def dpeak(g,p,c):
    i=int(np.argmax(p)); return {'gamma':float(g[i]),'height':float(p[i]),'count':int(c[i])}

if not RAW400.exists() or not SUM400.exists():
    raise FileNotFoundError('raw 400k prerequisite missing')

# Reproduce historical raw 50k exactly at pack() endpoint.
rcpgenerator.set_num_threads(1)
t0=time.time()
p=rcpgenerator.Packing(phi=PHI_INIT,N=N50,Ndim=3,box=[1,1,1],walls=[0,0,0],
                       dist={'type':'mono','d':1.0},neighbor_max=0,seed=SEED50)
p.pack(verbose=True,progress_interval=2000)
x=np.asarray(p.positions,float).copy(); box=np.asarray(p.box,float).copy(); diam=np.asarray(p.diameters,float).copy(); x%=box
d0=float(np.mean(diam))
phi50=float(np.sum(np.pi/6.0*diam**3)/np.prod(box))
tree=cKDTree(x,boxsize=box)
pairs=tree.query_pairs(QMAX*d0,output_type='ndarray')
dv=x[pairs[:,1]]-x[pairs[:,0]]; dv-=box*np.rint(dv/box)
q50=np.linalg.norm(dv,axis=1)/d0
contacts50=int(np.count_nonzero(q50<1.0))

provenance={'phi_raw':phi50,'contacts':contacts50,'expected_phi':EXPECTED_PHI50,'expected_contacts':EXPECTED_CONTACTS50,
            'phi_error':phi50-EXPECTED_PHI50,'contact_match':contacts50==EXPECTED_CONTACTS50,
            'protocol':{'commit':'974d703a655b337b93379dbcd1a6eef47386d674','seed':SEED50,'N':N50,
                        'phi_constructor':PHI_INIT,'threads':1,'preparation':'pack() endpoint only; no dilation/rescaling/preload/relaxation'}}
(ROOT/'raw50k_provenance.json').write_text(json.dumps(provenance,indent=2))
if abs(phi50-EXPECTED_PHI50)>PHI_TOL or contacts50!=EXPECTED_CONTACTS50:
    raise RuntimeError('raw50k provenance mismatch: '+json.dumps(provenance,sort_keys=True))

np.savez_compressed(ROOT/'packing_50k_raw_original.npz',positions=x,box=box,diameters=diam,phi=phi50,seed=SEED50)
np.savez_compressed(ROOT/'pair_distribution_50k_raw_original.npz',q=q50,gamma_over_2R=1.0-q50)

# Same canonical source edges; counts are recomputed from BOTH raw packings.
ref=json.loads((HERE/'reference_50k_radial_counts.json').read_text())
q_edges=np.asarray(ref['edges'],float)
counts50,_=np.histogram(q50,bins=q_edges)
raw400=np.load(RAW400)
q400=np.asarray(raw400['q'],float)
counts400,_=np.histogram(q400,bins=q_edges)

g50,p50d,c50,src50,supp50,n50=merge_counts(q_edges,counts50)
g400,p400d,c400,src400,supp400,n400=merge_counts(q_edges,counts400)
G50,P50,U50,S50=fit(g50,p50d,c50)
G400,P400,U400,S400=fit(g400,p400d,c400)

sum400=json.loads(SUM400.read_text())
phi400=float(sum400['metadata_400k']['phi_as_generated'])
contact400=int(np.count_nonzero(q400<1.0))
frac50=contacts50/n50; frac400=contact400/n400
scale=0.125

prob50=counts50/counts50.sum(); prob400=counts400/counts400.sum()
emp_l1=float(np.sum(np.abs(prob50-prob400)))
emp_ks=float(np.max(np.abs(np.cumsum(prob50)-np.cumsum(prob400))))
lo=min(G50.min(),G400.min()); hi=max(G50.max(),G400.max())
grid=np.linspace(lo,hi,250000)
A=np.interp(grid,G50,P50,left=0,right=0); B=np.interp(grid,G400,P400,left=0,right=0)
A/=np.trapezoid(A,grid); B/=np.trapezoid(B,grid)
fit_l1=float(np.trapezoid(np.abs(A-B),grid))
cdfA=np.r_[0,cumulative_trapezoid(A,grid)]; cdfB=np.r_[0,cumulative_trapezoid(B,grid)]
fit_ks=float(np.max(np.abs(cdfA-cdfB)))
curv50=robust_curvature(g50,p50d,c50); curv400=robust_curvature(g400,p400d,c400)
summary={
 'comparison':'raw original 50k generator endpoint vs raw independent 400k generator endpoint; neither density-adjusted nor relaxed',
 'phi_50k_raw':phi50,'phi_400k_raw':phi400,'phi_absolute_difference':phi400-phi50,'phi_relative_difference':(phi400-phi50)/phi50,
 'particle_count_50k':N50,'particle_count_400k':400000,
 'retained_pairs_50k':n50,'retained_pairs_400k':n400,
 'raw_contact_count_50k':contacts50,'raw_contact_count_400k':contact400,
 'scaled_contact_count_400k_to_50k':contact400*scale,
 'scaled_contact_count_relative_difference':(contact400*scale-contacts50)/contacts50,
 'raw_contact_fraction_50k':frac50,'raw_contact_fraction_400k':frac400,
 'raw_contact_fraction_absolute_difference':frac400-frac50,'raw_contact_fraction_relative_difference':(frac400-frac50)/frac50,
 'empirical_binned_pdf_L1':emp_l1,'empirical_binned_total_variation':0.5*emp_l1,'empirical_binned_cdf_KS':emp_ks,
 'fit_pdf_L1':fit_l1,'fit_total_variation':0.5*fit_l1,'fit_cdf_KS':fit_ks,
 'fixed_source_bin_scaled_count_relative_L1':float(np.sum(np.abs(counts400*scale-counts50))/np.sum(counts50)),
 'peak_50k':peak(G50,P50),'peak_400k':peak(G400,P400),
 'discrete_merged_peak_50k':dpeak(g50,p50d,c50),'discrete_merged_peak_400k':dpeak(g400,p400d,c400),
 'observed_signed_support_50k':list(supp50),'observed_signed_support_400k':list(supp400),
 'merged_bin_count_50k':int(len(g50)),'merged_bin_count_400k':int(len(g400)),
 'robust_log_density_curvature_50k':curv50,'robust_log_density_curvature_400k':curv400,
 'matched_robust_curvature_features':match_features(curv50,curv400),
 'fit_spec':{'signed_variable':'gamma/(2R)=1-q','min_actual_observations_per_merged_bin':MIN_COUNT,
             'single_positive_C2_signed_domain_fit':True,'u':'asinh((gamma/(2R))/1e-6)',
             'weights':'sqrt(merged bin count)','normalization':'unit integral over populated signed support',
             'official_smoothing_multiplier':16.0,'scipy_smoothing_s':SCIPY_S},
 'raw50k_provenance':provenance,
 'raw400k_provenance':sum400['metadata_400k'],
 'seconds_total':time.time()-t0
}
(ROOT/'summary.json').write_text(json.dumps(summary,indent=2))
np.savetxt(ROOT/'scaled_fixed_bin_counts.csv',np.c_[0.5*(q_edges[:-1]+q_edges[1:]),counts50,counts400*scale],
           delimiter=',',header='q_center,count_raw50k,count_raw400k_scaled_by_1_over_8',comments='')

plt.rcParams.update({'figure.facecolor':'white','axes.facecolor':'white','savefig.facecolor':'white',
'font.family':'serif','font.serif':['Times New Roman','Times','DejaVu Serif'],'mathtext.fontset':'dejavuserif',
'font.size':9.0,'axes.labelsize':10.2,'xtick.labelsize':8.2,'ytick.labelsize':8.7,'legend.fontsize':7.0,
'axes.linewidth':0.85,'xtick.major.width':0.85,'ytick.major.width':0.85,'xtick.minor.width':0.65,'ytick.minor.width':0.65,
'xtick.direction':'in','ytick.direction':'in','lines.linewidth':1.55,'pdf.fonttype':42,'ps.fonttype':42})
fig,ax=plt.subplots(figsize=(3.3,3.3))
ax.plot(g50,p50d,ls='none',marker='o',mfc='none',ms=2.6,label='raw 50k discrete')
ax.plot(g400,p400d,ls='none',marker='s',mfc='none',ms=2.4,label='raw 400k discrete')
ax.plot(G50,P50,ls='-',lw=1.65,label='raw 50k 16x fit')
ax.plot(G400,P400,ls='--',lw=1.55,label='raw 400k 16x fit')
ax.axvline(0,color='.45',ls=':',lw=.85)
ax.set_xscale('symlog',linthresh=1e-6,linscale=1,base=10); ax.set_yscale('log')
ax.set_xlim(-0.5,max(5e-4,supp50[1],supp400[1]))
positive=np.r_[p50d,p400d,P50,P400]; positive=positive[positive>0]
ax.set_ylim(max(1e-3,float(positive.min())*.65),float(positive.max())*1.4)
ticks=[-1e-1,-1e-3,-1e-5,0,1e-5,1e-4,1e-3]
labels=[r'$-10^{-1}$',r'$-10^{-3}$',r'$-10^{-5}$','0',r'$10^{-5}$',r'$10^{-4}$',r'$10^{-3}$']
ax.xaxis.set_major_locator(FixedLocator(ticks)); ax.xaxis.set_major_formatter(FixedFormatter(labels)); ax.xaxis.set_minor_formatter(NullFormatter())
ax.set_xlabel(r'$\gamma/(2R)$'); ax.set_ylabel(r'$P\!\left(\gamma/(2R)\right)$')
ax.tick_params(which='major',top=True,right=True,length=4); ax.tick_params(which='minor',top=True,right=True,length=2.5)
ax.grid(False)
for s in ax.spines.values(): s.set_linewidth(.85)
ax.legend(frameon=False,loc='best')
fig.tight_layout(pad=.25)
fig.savefig(ROOT/'Pgamma_raw50k_vs_raw400k_Rodrigues.png',dpi=300,bbox_inches='tight',pad_inches=.02)
fig.savefig(ROOT/'Pgamma_raw50k_vs_raw400k_Rodrigues.pdf',bbox_inches='tight',pad_inches=.02)
plt.close(fig)

print('RAW50K_PROVENANCE='+json.dumps(provenance,sort_keys=True),flush=True)
print('RAW50K_400K_SUMMARY='+json.dumps(summary,sort_keys=True),flush=True)
print('RAW50K_400K_RESULT_DIR='+str(ROOT),flush=True)

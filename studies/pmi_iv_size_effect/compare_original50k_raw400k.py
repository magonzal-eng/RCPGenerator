from pathlib import Path
import json, time, os
import numpy as np
from scipy.spatial import cKDTree
from scipy.interpolate import UnivariateSpline
from scipy.integrate import cumulative_trapezoid
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FixedFormatter, NullFormatter
import rcpgenerator

N400 = 400_000
SEED400 = 20261001
PHI_INIT = 0.05
QMAX = 1.5
X0 = 1e-6
MIN_COUNT = 50
SCIPY_S = 32.0
OUT = Path('/scratch/gautschi/gonza226/pmi_400k_raw_noadjust_20261002')
OUT.mkdir(parents=True, exist_ok=True)
HERE = Path(__file__).resolve().parent

def merge_side(g_edges, counts, idx, total):
    groups = []
    start = int(idx[0]); acc = 0
    for j in idx:
        acc += int(counts[j])
        if acc >= MIN_COUNT:
            groups.append((start, int(j), acc))
            start = int(j) + 1
            acc = 0
    if start <= int(idx[-1]):
        if groups:
            a,b,c = groups[-1]
            groups[-1] = (a, int(idx[-1]), c + acc)
        else:
            groups = [(start, int(idx[-1]), acc)]
    centers=[]; dens=[]; merged=[]
    for a,b,c in groups:
        lo=float(g_edges[a]); hi=float(g_edges[b+1])
        if hi == 0:
            cen = lo/2
        elif lo == 0:
            cen = hi/2
        elif hi < 0:
            cen = -np.sqrt(abs(lo*hi))
        else:
            cen = np.sqrt(lo*hi)
        centers.append(cen)
        dens.append(c/(total*(hi-lo)))
        merged.append(c)
    return np.asarray(centers), np.asarray(dens), np.asarray(merged)

def merge_counts(q_edges, q_counts):
    g_edges = 1.0 - q_edges[::-1]
    counts = q_counts[::-1]
    nz = np.flatnonzero(counts)
    if len(nz) == 0:
        raise RuntimeError('No populated source bins')
    lo, hi = int(nz[0]), int(nz[-1])
    total = int(counts[lo:hi+1].sum())
    ii = np.arange(len(counts))
    neg = np.where((ii>=lo)&(ii<=hi)&(g_edges[1:]<=0))[0]
    pos = np.where((ii>=lo)&(ii<=hi)&(g_edges[:-1]>=0))[0]
    gn,pn,cn = merge_side(g_edges,counts,neg,total)
    gp,pp,cp = merge_side(g_edges,counts,pos,total)
    gd=np.r_[gn,gp]; pd=np.r_[pn,pp]; cd=np.r_[cn,cp]
    order=np.argsort(gd)
    return gd[order],pd[order],cd[order],counts,(float(g_edges[lo]),float(g_edges[hi+1])),total

def fit_from_merged(gd,pd,cd,s_mult=SCIPY_S,ngrid=60000):
    u=np.arcsinh(gd/X0)
    o=np.argsort(u)
    sp=UnivariateSpline(u[o],np.log(pd[o]),w=np.sqrt(cd[o]),s=len(u)*s_mult,k=3)
    ug=np.linspace(u[o][0],u[o][-1],ngrid)
    g=X0*np.sinh(ug)
    logp=sp(ug)
    raw=np.exp(np.clip(logp,-745,700))
    p=raw/np.trapezoid(raw,g)
    return g,p,ug,sp

def zero_crossings(x,y):
    s=np.sign(y)
    idx=np.where(s[:-1]*s[1:]<0)[0]
    roots=[]
    for i in idx:
        x0,x1=x[i],x[i+1]; y0,y1=y[i],y[i+1]
        roots.append(float(x0-y0*(x1-x0)/(y1-y0)))
    return np.asarray(roots)

def robust_curvature(gd,pd,cd):
    # Curvature landmarks are inflections of log P in transformed signed coordinate u.
    roots={}
    for sm in (16.0,32.0,64.0):
        g,p,u,sp=fit_from_merged(gd,pd,cd,sm,50000)
        d2=sp.derivative(2)(u)
        roots[sm]=zero_crossings(u,d2)
    base=roots[32.0]
    feats=[]
    for r in base:
        d16=float(np.min(np.abs(roots[16.0]-r))) if len(roots[16.0]) else np.inf
        d64=float(np.min(np.abs(roots[64.0]-r))) if len(roots[64.0]) else np.inf
        if d16 <= 0.25 and d64 <= 0.25:
            feats.append({'u':float(r),'gamma':float(X0*np.sinh(r)),
                          'du_s16':d16,'du_s64':d64})
    return feats

def nearest_feature_matches(a,b,max_du=0.25):
    out=[]
    if not a or not b:
        return out
    ub=np.array([x['u'] for x in b])
    for x in a:
        j=int(np.argmin(np.abs(ub-x['u'])))
        du=float(abs(ub[j]-x['u']))
        if du<=max_du:
            out.append({'gamma50':x['gamma'],'gamma400':b[j]['gamma'],'du':du})
    return out

def peak(g,p):
    i=int(np.argmax(p))
    return {'gamma':float(g[i]),'height':float(p[i])}

def discrete_peak(gd,pd,cd):
    i=int(np.argmax(pd))
    return {'gamma':float(gd[i]),'height':float(pd[i]),'count':int(cd[i])}

ref=json.loads((HERE/'reference_50k_radial_counts.json').read_text())
q_edges=np.asarray(ref['edges'],float)
counts50=np.asarray(ref['counts'],int)
if len(q_edges) != len(counts50)+1:
    raise RuntimeError('Invalid 50k reference histogram')

rcpgenerator.set_num_threads(1)
t0=time.time()
p=rcpgenerator.Packing(phi=PHI_INIT,N=N400,Ndim=3,box=[1,1,1],walls=[0,0,0],
                       dist={'type':'mono','d':1.0},neighbor_max=0,seed=SEED400)
p.pack(verbose=True,progress_interval=5000)

# IMPORTANT: use exactly the state returned by pack(); no dilation, density adjustment,
# preload, affine rescaling, or mechanical relaxation.
x=np.asarray(p.positions,float).copy()
box=np.asarray(p.box,float).copy()
diam=np.asarray(p.diameters,float).copy()
x%=box
d0=float(np.mean(diam))
phi_raw=float(np.sum(np.pi/6.0*diam**3)/np.prod(box))

tree=cKDTree(x,boxsize=box)
pairs=tree.query_pairs(QMAX*d0,output_type='ndarray')
dv=x[pairs[:,1]]-x[pairs[:,0]]
dv-=box*np.rint(dv/box)
r=np.linalg.norm(dv,axis=1)
q=r/d0
gamma=1.0-q
counts400,_=np.histogram(q,bins=q_edges)

np.savez_compressed(OUT/'packing_400k_as_generated.npz',
                    positions=x,box=box,diameters=diam,phi=phi_raw,seed=SEED400)
np.savez_compressed(OUT/'pair_distribution_400k_as_generated.npz',
                    q=q,gamma_over_2R=gamma)

g50,p50d,c50,source50,support50,n50=merge_counts(q_edges,counts50)
g400,p400d,c400,source400,support400,n400=merge_counts(q_edges,counts400)
G50,P50,U50,S50=fit_from_merged(g50,p50d,c50)
G400,P400,U400,S400=fit_from_merged(g400,p400d,c400)

contact50=int(counts50[q_edges[1:]<=1.0].sum())
contact400=int(np.count_nonzero(q<1.0))
frac50=contact50/n50
frac400=contact400/n400
scale=50000/400000

prob50=counts50/counts50.sum()
prob400=counts400/counts400.sum()
emp_l1=float(np.sum(np.abs(prob50-prob400)))
emp_ks=float(np.max(np.abs(np.cumsum(prob50)-np.cumsum(prob400))))

lo=min(G50.min(),G400.min()); hi=max(G50.max(),G400.max())
grid=np.linspace(lo,hi,250000)
A=np.interp(grid,G50,P50,left=0,right=0)
B=np.interp(grid,G400,P400,left=0,right=0)
A=A/np.trapezoid(A,grid); B=B/np.trapezoid(B,grid)
fit_l1=float(np.trapezoid(np.abs(A-B),grid))
cdfA=np.r_[0,cumulative_trapezoid(A,grid)]
cdfB=np.r_[0,cumulative_trapezoid(B,grid)]
fit_ks=float(np.max(np.abs(cdfA-cdfB)))

scaled400=counts400*scale
scaled_count_l1=float(np.sum(np.abs(scaled400-counts50))/np.sum(counts50))

curv50=robust_curvature(g50,p50d,c50)
curv400=robust_curvature(g400,p400d,c400)
matches=nearest_feature_matches(curv50,curv400)

pk50=peak(G50,P50); pk400=peak(G400,P400)
dpk50=discrete_peak(g50,p50d,c50); dpk400=discrete_peak(g400,p400d,c400)

summary={
 'comparison':'authoritative original 50k radial-count reference vs independently regenerated 400k as-generated packing; no 400k density adjustment or relaxation',
 'particle_count_50k':50000,'particle_count_400k':400000,
 'raw_count_scale_400k_to_50k':scale,
 'retained_pairs_50k':n50,'retained_pairs_400k':n400,
 'raw_contact_count_50k':contact50,'raw_contact_count_400k':contact400,
 'scaled_contact_count_400k_to_50k':contact400*scale,
 'scaled_contact_count_relative_difference':(contact400*scale-contact50)/contact50,
 'raw_contact_fraction_50k':frac50,'raw_contact_fraction_400k':frac400,
 'raw_contact_fraction_absolute_difference':frac400-frac50,
 'raw_contact_fraction_relative_difference':(frac400-frac50)/frac50,
 'empirical_binned_pdf_L1':emp_l1,'empirical_binned_total_variation':0.5*emp_l1,
 'empirical_binned_cdf_KS':emp_ks,
 'fit_pdf_L1':fit_l1,'fit_total_variation':0.5*fit_l1,'fit_cdf_KS':fit_ks,
 'fixed_source_bin_scaled_count_relative_L1':scaled_count_l1,
 'peak_50k':pk50,'peak_400k':pk400,
 'peak_gamma_difference':pk400['gamma']-pk50['gamma'],
 'peak_height_relative_difference':pk400['height']/pk50['height']-1,
 'discrete_merged_peak_50k':dpk50,'discrete_merged_peak_400k':dpk400,
 'observed_signed_support_50k':list(support50),'observed_signed_support_400k':list(support400),
 'merged_bin_count_50k':int(len(g50)),'merged_bin_count_400k':int(len(g400)),
 'robust_log_density_curvature_50k':curv50,
 'robust_log_density_curvature_400k':curv400,
 'matched_robust_curvature_features':matches,
 'fit_spec':{'signed_variable':'gamma/(2R)=1-q','min_actual_observations_per_merged_bin':MIN_COUNT,
             'single_positive_C2_signed_domain_fit':True,'u':'asinh((gamma/(2R))/1e-6)',
             'weights':'sqrt(merged bin count)','normalization':'unit integral over populated signed support',
             'official_smoothing_multiplier':16.0,'scipy_smoothing_s':32.0},
 'metadata_400k':{'N':N400,'seed':SEED400,'generator_initial_phi':PHI_INIT,
                  'phi_as_generated':phi_raw,'box':box.tolist(),'mean_diameter':d0,
                  'pair_count_q_le_1p5':int(len(q)),'contact_count':contact400,
                  'contact_fraction':frac400,'generator_commit':'974d703a655b337b93379dbcd1a6eef47386d674',
                  'preparation':'RCPGenerator pack() output exactly; no dilation, density matching, preload, affine rescaling, or relaxation',
                  'seconds':time.time()-t0}
}
(OUT/'summary.json').write_text(json.dumps(summary,indent=2))
np.savetxt(OUT/'scaled_fixed_bin_counts.csv',
           np.c_[0.5*(q_edges[:-1]+q_edges[1:]),counts50,scaled400],
           delimiter=',',header='q_center,count_50k,count_400k_scaled_by_1_over_8',comments='')

plt.rcParams.update({'figure.facecolor':'white','axes.facecolor':'white','savefig.facecolor':'white',
'font.family':'serif','font.serif':['Times New Roman','Times','DejaVu Serif'],'mathtext.fontset':'dejavuserif',
'font.size':9.0,'axes.labelsize':10.2,'xtick.labelsize':8.2,'ytick.labelsize':8.7,'legend.fontsize':7.0,
'axes.linewidth':0.85,'xtick.major.width':0.85,'ytick.major.width':0.85,
'xtick.minor.width':0.65,'ytick.minor.width':0.65,'xtick.direction':'in','ytick.direction':'in',
'lines.linewidth':1.55,'pdf.fonttype':42,'ps.fonttype':42})
fig,ax=plt.subplots(figsize=(3.3,3.3))
ax.plot(g50,p50d,ls='none',marker='o',mfc='none',ms=2.5,label='50k discrete')
ax.plot(g400,p400d,ls='none',marker='s',mfc='none',ms=2.3,label='400k discrete')
ax.plot(G50,P50,ls='-',lw=1.65,label='50k 16x fit')
ax.plot(G400,P400,ls='--',lw=1.55,label='400k raw 16x fit')
ax.axvline(0,color='.45',ls=':',lw=.85)
ax.set_xscale('symlog',linthresh=1e-6,linscale=1,base=10)
ax.set_yscale('log')
ax.set_xlim(-0.5,max(3e-4,support400[1],support50[1]))
positive=np.r_[p50d,p400d,P50,P400]; positive=positive[positive>0]
ax.set_ylim(max(1e-3,float(positive.min())*.65),float(positive.max())*1.4)
ticks=[-1e-1,-1e-3,-1e-5,0,1e-5,1e-4,1e-3]
labels=[r'$-10^{-1}$',r'$-10^{-3}$',r'$-10^{-5}$','0',r'$10^{-5}$',r'$10^{-4}$',r'$10^{-3}$']
ax.xaxis.set_major_locator(FixedLocator(ticks)); ax.xaxis.set_major_formatter(FixedFormatter(labels))
ax.xaxis.set_minor_formatter(NullFormatter())
ax.set_xlabel(r'$\gamma/(2R)$'); ax.set_ylabel(r'$P\!\left(\gamma/(2R)\right)$')
ax.tick_params(which='major',top=True,right=True,length=4)
ax.tick_params(which='minor',top=True,right=True,length=2.5)
ax.grid(False)
for s in ax.spines.values(): s.set_linewidth(.85)
ax.legend(frameon=False,loc='best')
fig.tight_layout(pad=.25)
fig.savefig(OUT/'Pgamma_original50k_vs_raw400k_Rodrigues.png',dpi=300,bbox_inches='tight',pad_inches=.02)
fig.savefig(OUT/'Pgamma_original50k_vs_raw400k_Rodrigues.pdf',bbox_inches='tight',pad_inches=.02)
plt.close(fig)
print('PMI400K_RAW_SUMMARY='+json.dumps(summary,sort_keys=True),flush=True)
print('PMI400K_RAW_RESULT_DIR='+str(OUT),flush=True)

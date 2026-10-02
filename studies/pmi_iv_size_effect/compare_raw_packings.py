from pathlib import Path
import json, time
import numpy as np
from scipy.spatial import cKDTree
from scipy.interpolate import UnivariateSpline
from scipy.integrate import cumulative_trapezoid
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FixedFormatter, NullFormatter
import rcpgenerator

OUT=Path('/scratch/gautschi/gonza226/pmi_raw_packings_40k_50k_400k_20261002')
OUT.mkdir(parents=True,exist_ok=True)
SEED=20261002
PHI_INIT=0.05
QMAX=1.5
X0=1e-6
S=32.0
MIN_COUNT=50

# Common source-bin geometry: signed log resolution with exact zero interface.
# Wide enough to contain the raw RCP-generated gap/contact population.
neg_mag=np.geomspace(1e-8,0.5,420)
pos_mag=np.geomspace(1e-8,0.02,320)
g_edges=np.unique(np.r_[-neg_mag[::-1],0.0,pos_mag])
q_edges=np.sort(1.0-g_edges)

def merge_side(g_edges,g_counts,idx,N):
    groups=[]; start=int(idx[0]); acc=0
    for j in idx:
        acc+=int(g_counts[j])
        if acc>=MIN_COUNT:
            groups.append((start,int(j),acc)); start=int(j)+1; acc=0
    if start<=int(idx[-1]):
        if groups:
            a,b,c=groups[-1]; groups[-1]=(a,int(idx[-1]),c+acc)
        else:
            groups=[(start,int(idx[-1]),acc)]
    centers=[]; dens=[]; counts=[]
    for a,b,c in groups:
        lo,hi=float(g_edges[a]),float(g_edges[b+1])
        if hi==0: cen=lo/2
        elif lo==0: cen=hi/2
        elif hi<0: cen=-np.sqrt(abs(lo*hi))
        else: cen=np.sqrt(lo*hi)
        centers.append(cen); dens.append(c/(N*(hi-lo))); counts.append(c)
    return np.asarray(centers),np.asarray(dens),np.asarray(counts)

def canonical_fit(gamma):
    counts,_=np.histogram(gamma,bins=g_edges)
    nz=np.flatnonzero(counts)
    lo=max(0,int(nz[0])); hi=min(len(counts)-1,int(nz[-1]))
    neg=np.where((np.arange(len(counts))>=lo)&(np.arange(len(counts))<=hi)&(g_edges[1:]<=0))[0]
    pos=np.where((np.arange(len(counts))>=lo)&(np.arange(len(counts))<=hi)&(g_edges[:-1]>=0))[0]
    gn,pn,cn=merge_side(g_edges,counts,neg,len(gamma))
    gp,pp,cp=merge_side(g_edges,counts,pos,len(gamma))
    gd=np.r_[gn,gp]; pd=np.r_[pn,pp]; cd=np.r_[cn,cp]
    u=np.arcsinh(gd/X0); o=np.argsort(u)
    sp=UnivariateSpline(u[o],np.log(pd[o]),w=np.sqrt(cd[o]),s=len(u)*S,k=3)
    support_lo=float(g_edges[lo]); support_hi=float(g_edges[hi+1])
    ug=np.linspace(np.arcsinh(support_lo/X0),np.arcsinh(support_hi/X0),30000)
    gg=X0*np.sinh(ug); raw=np.exp(sp(ug)); pf=raw/np.trapezoid(raw,gg)
    return gd,pd,cd,gg,pf,counts,(support_lo,support_hi)

def generate(N):
    folder=OUT/f'N{N}'
    folder.mkdir(exist_ok=True)
    t0=time.time()
    p=rcpgenerator.Packing(phi=PHI_INIT,N=N,Ndim=3,box=[1,1,1],walls=[0,0,0],
                           dist={'type':'mono','d':1.0},neighbor_max=0,seed=SEED)
    p.pack(verbose=True,progress_interval=5000)
    # EXACT as returned by packing code: no scaling, dilation, loading, or relaxation.
    x=np.asarray(p.positions,float).copy()
    box=np.asarray(p.box,float).copy()
    d=np.asarray(p.diameters,float).copy()
    d0=float(np.mean(d))
    phi=float(np.sum(np.pi/6*d**3)/np.prod(box))
    tree=cKDTree(x%box,boxsize=box)
    pairs=tree.query_pairs(QMAX*d0,output_type='ndarray')
    dv=x[pairs[:,1]]-x[pairs[:,0]]
    dv-=box*np.rint(dv/box)
    r=np.linalg.norm(dv,axis=1)
    # Monodisperse generator: 2R=d0.
    gamma=1.0-r/d0
    np.savez_compressed(folder/'packing_as_generated.npz',positions=x,box=box,diameters=d,phi=phi,seed=SEED)
    np.savez_compressed(folder/'pairs_q1p5.npz',gamma_over_2R=gamma,q=1-gamma)
    gd,pd,cd,gg,pf,counts,support=canonical_fit(gamma)
    contact=int(np.count_nonzero(gamma>0))
    im=int(np.argmax(pf))
    cdf=np.r_[0,cumulative_trapezoid(pf,gg)]
    meta=dict(N=N,seed=SEED,phi_init_argument=PHI_INIT,phi_as_generated=phi,
              box=box.tolist(),mean_diameter=d0,pair_count_q_le_1p5=int(len(gamma)),
              contact_count=contact,empirical_contact_fraction=contact/len(gamma),
              fit_contact_fraction=float(np.trapezoid(pf[gg>0],gg[gg>0])),
              fit_peak_gamma_over_2R=float(gg[im]),fit_peak_density=float(pf[im]),
              fit_support=list(support),minimum_merged_bin_count=int(cd.min()),
              generator_commit='974d703a655b337b93379dbcd1a6eef47386d674',
              preparation='RCPGenerator pack() output exactly; no rescaling, dilation, preload, or relaxation',
              canonical_fit={'x0':X0,'s':S,'min_count':MIN_COUNT,'single_signed_C2':True},
              seconds=time.time()-t0)
    (folder/'metadata.json').write_text(json.dumps(meta,indent=2))
    np.savez_compressed(folder/'canonical_fit.npz',discrete_x=gd,discrete_density=pd,
                        discrete_counts=cd,x=gg,density=pf,cdf=cdf,source_counts=counts)
    print('RAW_PACKING_RESULT='+json.dumps(meta,sort_keys=True),flush=True)
    return meta,(gd,pd,gg,pf,cdf,counts)

rcpgenerator.set_num_threads(1)
allmeta={}; fits={}
for N in (40000,50000,400000):
    allmeta[N],fits[N]=generate(N)

# Pairwise normalized-distribution metrics on common overlap support.
metrics={}
for a,b in ((40000,50000),(50000,400000),(40000,400000)):
    ga,pa,Ga,Pa,Ca,_=fits[a]; gb,pb,Gb,Pb,Cb,_=fits[b]
    lo=max(Ga.min(),Gb.min()); hi=min(Ga.max(),Gb.max())
    grid=np.linspace(lo,hi,200000)
    A=np.interp(grid,Ga,Pa); B=np.interp(grid,Gb,Pb)
    # Renormalize only on common comparison support.
    A=A/np.trapezoid(A,grid); B=B/np.trapezoid(B,grid)
    cA=np.r_[0,cumulative_trapezoid(A,grid)]
    cB=np.r_[0,cumulative_trapezoid(B,grid)]
    l1=float(np.trapezoid(np.abs(A-B),grid))
    ks=float(np.max(np.abs(cA-cB)))
    metrics[f'{a}_vs_{b}']={'pdf_L1':l1,'TV':0.5*l1,'CDF_KS':ks,'common_support':[float(lo),float(hi)]}

summary={'packings':allmeta,'pairwise_metrics':metrics}
(OUT/'summary.json').write_text(json.dumps(summary,indent=2))

# Presentation-format comparison
plt.rcParams.update({'figure.facecolor':'white','axes.facecolor':'white','savefig.facecolor':'white',
'font.family':'serif','font.serif':['Times New Roman','Times','DejaVu Serif'],'mathtext.fontset':'dejavuserif',
'font.size':9.0,'axes.labelsize':10.2,'xtick.labelsize':8.2,'ytick.labelsize':8.7,'legend.fontsize':7.0,
'axes.linewidth':0.85,'xtick.major.width':0.85,'ytick.major.width':0.85,'xtick.minor.width':0.65,'ytick.minor.width':0.65,
'xtick.direction':'in','ytick.direction':'in','lines.linewidth':1.55,'pdf.fonttype':42,'ps.fonttype':42})
fig,ax=plt.subplots(figsize=(3.3,3.3))
styles={40000:'--',50000:'-',400000:'-.'}
marks={40000:'o',50000:'s',400000:'^'}
for N in (40000,50000,400000):
    gd,pd,gg,pf,_,_=fits[N]
    ax.plot(gd,pd,ls='none',marker=marks[N],mfc='none',ms=2.2,alpha=.65)
    ax.plot(gg,pf,ls=styles[N],lw=1.55,label=f'{N//1000}k')
ax.axvline(0,color='.45',ls=':',lw=.85)
ax.set_xscale('symlog',linthresh=1e-6,linscale=1,base=10); ax.set_yscale('log')
ax.set_xlim(-0.5,0.02)
ticks=[-1e-1,-1e-3,-1e-5,0,1e-5,1e-3,1e-2]
labels=[r'$-10^{-1}$',r'$-10^{-3}$',r'$-10^{-5}$','0',r'$10^{-5}$',r'$10^{-3}$',r'$10^{-2}$']
ax.xaxis.set_major_locator(FixedLocator(ticks)); ax.xaxis.set_major_formatter(FixedFormatter(labels)); ax.xaxis.set_minor_formatter(NullFormatter())
ax.set_xlabel(r'$\gamma/(2R)$'); ax.set_ylabel(r'$P\!\left(\gamma/(2R)\right)$')
ax.tick_params(which='major',top=True,right=True,length=4); ax.tick_params(which='minor',top=True,right=True,length=2.5)
ax.grid(False)
for s in ax.spines.values(): s.set_linewidth(.85)
ax.legend(frameon=False,loc='best',title='as generated')
fig.tight_layout(pad=.25)
fig.savefig(OUT/'Pgamma_raw_40k_50k_400k_presentation.png',dpi=300,bbox_inches='tight',pad_inches=.02)
fig.savefig(OUT/'Pgamma_raw_40k_50k_400k_presentation.pdf',bbox_inches='tight',pad_inches=.02)
plt.close(fig)
print('RAW_COMPARISON_SUMMARY='+json.dumps(summary,sort_keys=True),flush=True)

from pathlib import Path
import json, numpy as np
from scipy.interpolate import UnivariateSpline
from scipy.integrate import cumulative_trapezoid
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FixedFormatter, NullFormatter

HERE=Path(__file__).resolve().parent
SCRATCH=Path('/scratch/gautschi/gonza226/pmi_400k_size_effect_20261001')
OUT=SCRATCH/'comparison'
OUT.mkdir(parents=True,exist_ok=True)
MIN_COUNT=50
X0=1e-6
SMOOTH_MULT=32.0

def merge_counts(g_edges,g_counts,N,min_count=50):
    neg=np.where((g_edges[:-1]>=-0.5)&(g_edges[1:]<=0))[0]
    pos=np.where((g_edges[:-1]>=0)&(g_edges[1:]<=3e-4))[0]
    def side(idx):
        groups=[]; start=int(idx[0]); acc=0
        for j in idx:
            acc+=int(g_counts[j])
            if acc>=min_count:
                groups.append((start,int(j),acc)); start=int(j)+1; acc=0
        if start<=int(idx[-1]):
            if groups:
                a,b,c=groups[-1]; groups[-1]=(a,int(idx[-1]),c+acc)
            else: groups=[(start,int(idx[-1]),acc)]
        c0=[]; d0=[]; n0=[]
        for a,b,c in groups:
            lo,hi=float(g_edges[a]),float(g_edges[b+1])
            if hi==0: cen=lo/2
            elif lo==0: cen=hi/2
            elif hi<0: cen=-np.sqrt(abs(lo*hi))
            else: cen=np.sqrt(lo*hi)
            c0.append(cen); d0.append(c/(N*(hi-lo))); n0.append(c)
        return np.asarray(c0),np.asarray(d0),np.asarray(n0)
    gn,pn,cn=side(neg); gp,pp,cp=side(pos)
    return np.r_[gn,gp],np.r_[pn,pp],np.r_[cn,cp]

def fit16(gd,pd,cd):
    u=np.arcsinh(gd/X0); o=np.argsort(u)
    sp=UnivariateSpline(u[o],np.log(pd[o]),w=np.sqrt(cd[o]),s=len(u)*SMOOTH_MULT,k=3)
    ug=np.linspace(np.arcsinh(-0.5/X0),np.arcsinh(3e-4/X0),30000)
    gg=X0*np.sinh(ug)
    raw=np.exp(sp(ug)); p=raw/np.trapezoid(raw,gg)
    return gg,p

def diagnostics(g,p):
    contact=float(np.trapezoid(p[g>0],g[g>0]))
    imax=int(np.argmax(p))
    d2=np.gradient(np.gradient(p,g),g)
    m=(g>0)&(g<1e-4); gm=g[m]; z=d2[m]
    infl=gm[:-1][np.sign(z[:-1])*np.sign(z[1:])<0]
    return dict(contact_fraction_fit=contact,peak_gamma_over_2R=float(g[imax]),peak_density=float(p[imax]),
                contact_side_inflections=[float(v) for v in infl])

ref=json.loads((HERE/'reference_50k_radial_counts.json').read_text())
q_edges=np.asarray(ref['edges'],float)
counts50=np.asarray(ref['counts'],int)
g_edges=1-q_edges[::-1]
gc50=counts50[::-1]
support=(g_edges[:-1]>=-0.5)&(g_edges[1:]<=3e-4)
N50=int(gc50[support].sum())
assert N50==373608
g50,p50d,c50=merge_counts(g_edges,gc50,N50,MIN_COUNT)
G50,P50=fit16(g50,p50d,c50)

raw400=np.load(SCRATCH/'pair_distribution_400k.npz')
q400=np.asarray(raw400['q'],float)
counts400,_=np.histogram(q400,bins=q_edges)
gc400=counts400[::-1]
N400pairs=int(gc400[support].sum())
g400,p400d,c400=merge_counts(g_edges,gc400,N400pairs,MIN_COUNT)
G400,P400=fit16(g400,p400d,c400)

meta400=json.loads((SCRATCH/'metadata.json').read_text())
contact50=149326/373608
contact400=float(meta400['contact_fraction'])

common=np.linspace(-0.5,3e-4,200000)
p50i=np.interp(common,G50,P50)
p400i=np.interp(common,G400,P400)
l1=float(np.trapezoid(np.abs(p400i-p50i),common))
cdf50=np.r_[0,cumulative_trapezoid(p50i,common)]
cdf400=np.r_[0,cumulative_trapezoid(p400i,common)]
ks=float(np.max(np.abs(cdf400-cdf50)))

# Raw count comparison on identical source bins, scaled to 50k-particle basis.
scale=50000/400000
scaled400=counts400*scale
mask=(counts50+scaled400)>0
count_l1=float(np.sum(np.abs(scaled400-counts50))/np.sum(counts50))

summary=dict(
    comparison='Independent 400k realization versus accepted 50k reference at matched phi0',
    particle_count_50k=50000,particle_count_400k=400000,raw_count_scale_400k_to_50k=scale,
    retained_pairs_50k=N50,retained_pairs_400k=N400pairs,
    measured_contact_fraction_50k=contact50,measured_contact_fraction_400k=contact400,
    measured_contact_fraction_difference=contact400-contact50,
    fit_L1_probability_density=l1,fit_CDF_max_difference=ks,
    scaled_fixed_bin_count_relative_L1=count_l1,
    fit50=diagnostics(G50,P50),fit400=diagnostics(G400,P400),
    canonical_fit=dict(minimum_bin_count=MIN_COUNT,x0=X0,smoothing='16x',s_parameter_multiplier=SMOOTH_MULT,C2=True,single_signed_domain=True)
)
(OUT/'summary.json').write_text(json.dumps(summary,indent=2))

# Presentation-format normalized-density comparison.
plt.rcParams.update({'figure.facecolor':'white','axes.facecolor':'white','savefig.facecolor':'white',
'font.family':'serif','font.serif':['Times New Roman','Times','DejaVu Serif'],'mathtext.fontset':'dejavuserif',
'font.size':9.0,'axes.labelsize':10.2,'xtick.labelsize':8.2,'ytick.labelsize':8.7,'legend.fontsize':7.0,
'axes.linewidth':0.85,'xtick.major.width':0.85,'ytick.major.width':0.85,'xtick.minor.width':0.65,'ytick.minor.width':0.65,
'xtick.direction':'in','ytick.direction':'in','lines.linewidth':1.55,'pdf.fonttype':42,'ps.fonttype':42})
fig,ax=plt.subplots(figsize=(3.3,3.3))
ax.plot(g50,p50d,ls='none',marker='o',mfc='none',ms=2.7,label='50k discrete',zorder=4)
ax.plot(g400,p400d,ls='none',marker='s',mfc='none',ms=2.5,label='400k discrete',zorder=4)
ax.plot(G50,P50,ls='-',lw=1.65,label=r'50k official 16x fit',zorder=3)
ax.plot(G400,P400,ls='--',lw=1.55,label=r'400k 16x fit',zorder=2)
ax.axvline(0,color='.45',ls=':',lw=.85,zorder=1)
ax.set_xscale('symlog',linthresh=1e-6,linscale=1,base=10); ax.set_yscale('log'); ax.set_xlim(-0.5,3e-4)
vis=np.r_[p50d,p400d,P50,P400]; ax.set_ylim(max(1e-2,vis.min()*.7),vis.max()*1.35)
ticks=[-1e-1,-1e-3,-1e-5,0,1e-5,1e-4]; labels=[r'$-10^{-1}$',r'$-10^{-3}$',r'$-10^{-5}$','0',r'$10^{-5}$',r'$10^{-4}$']
ax.xaxis.set_major_locator(FixedLocator(ticks)); ax.xaxis.set_major_formatter(FixedFormatter(labels)); ax.xaxis.set_minor_formatter(NullFormatter())
ax.set_xlabel(r'$\gamma/(2R)$'); ax.set_ylabel(r'$P\!\left(\gamma/(2R)\right)$')
ax.tick_params(which='major',direction='in',top=True,right=True,length=4); ax.tick_params(which='minor',direction='in',top=True,right=True,length=2.5)
ax.grid(False)
for s in ax.spines.values(): s.set_visible(True); s.set_linewidth(.85)
ax.legend(frameon=False,loc='lower left',bbox_to_anchor=(.02,.12),borderaxespad=0)
fig.tight_layout(pad=.25)
fig.savefig(OUT/'Pgamma_50k_vs_400k_presentation.pdf',bbox_inches='tight',pad_inches=.02)
fig.savefig(OUT/'Pgamma_50k_vs_400k_presentation.png',dpi=300,bbox_inches='tight',pad_inches=.02)
plt.close(fig)

# Fixed-bin raw-count comparison, with 400k scaled by 1/8.
centers=0.5*(q_edges[:-1]+q_edges[1:])
np.savetxt(OUT/'scaled_fixed_bin_counts.csv',np.c_[centers,counts50,scaled400],delimiter=',',
           header='q_center,count_50k,count_400k_scaled_to_50k',comments='')
print(json.dumps(summary,indent=2))

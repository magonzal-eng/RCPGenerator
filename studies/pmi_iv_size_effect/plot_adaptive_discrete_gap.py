from pathlib import Path
import json, math, csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FixedFormatter, NullFormatter

ROOT=Path('/scratch/gautschi/gonza226/pmi_nearcontact_monotone_gap_atom_M0_20261002')
summary=json.loads((ROOT/'nearcontact_monotone_gap_atom_summary.json').read_text())
fit=pd.read_csv(ROOT/'nearcontact_monotone_gap_density.csv')
xfit=fit['x_gamma_over_2R'].to_numpy(float)
pfit=fit['p_gap_conditional'].to_numpy(float)
x_cut=float(summary['gap_fit']['near_contact_monotonicity_cut_x'])

paths={
 'raw50k':Path('/scratch/gautschi/gonza226/pmi_raw50k_original_20261002/pair_distribution_50k_raw_original.npz'),
 'tol5e-5':Path('/scratch/gautschi/gonza226/pmi_raw50k_tightov_20261002/pair_distribution_50k_tighter_overlap.npz'),
 'tol5e-6':Path('/scratch/gautschi/gonza226/pmi_raw50k_tol5e-6_20261002/pair_distribution_50k_tol5e-6.npz'),
 'raw400k':Path('/scratch/gautschi/gonza226/pmi_400k_raw_noadjust_20261002/pair_distribution_400k_as_generated.npz'),
}
mag=np.geomspace(1e-10,0.5,620)
edges=np.r_[-mag[::-1],0.0]

def adaptive_display(sample,min_count=50):
    c,_=np.histogram(sample,bins=edges)
    nz=np.flatnonzero(c)
    lo=int(nz[0]); hi=int(nz[-1])
    total=int(c[lo:hi+1].sum())
    # Merge from contact outward to preserve near-contact resolution.
    groups=[]; g_hi=hi; acc=0
    for j in range(hi,lo-1,-1):
        acc += int(c[j])
        if acc>=min_count:
            groups.append((j,g_hi,acc))
            g_hi=j-1; acc=0
    if acc>0:
        if groups:
            a,b,n=groups[-1]
            groups[-1]=(lo,b,n+acc)
        else:
            groups=[(lo,hi,acc)]
    groups=sorted(groups,key=lambda z:z[0])
    rows=[]
    for a,b,n in groups:
        e0=float(edges[a]); e1=float(edges[b+1])
        xc=-math.sqrt(abs(e0*e1)) if e1<0 else e0/2
        density=n/(total*(e1-e0))
        rows.append((xc,density,n,e0,e1))
    return rows

all_rows=[]
display={}
for name,p in paths.items():
    q=np.asarray(np.load(p)['q'],float)
    gaps=(1.0-q)
    gaps=gaps[gaps<0]
    rows=adaptive_display(gaps,50)
    display[name]=rows
    for xc,d,n,e0,e1 in rows:
        all_rows.append((name,xc,d,n,e0,e1))

with open(ROOT/'adaptive_discrete_gap_points.csv','w',newline='') as f:
    w=csv.writer(f)
    w.writerow(['dataset','x_center','conditional_gap_density','actual_count','bin_left','bin_right'])
    w.writerows(all_rows)

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

labels={'raw50k':'raw 50k','tol5e-5':r'50k $5\times10^{-5}$','tol5e-6':r'50k $5\times10^{-6}$','raw400k':'raw 400k'}
markers={'raw50k':'o','tol5e-5':'s','tol5e-6':'^','raw400k':'d'}

fig,ax=plt.subplots(figsize=(6.8,3.5))
for name in ['raw50k','tol5e-5','tol5e-6','raw400k']:
    a=np.asarray(display[name],float)
    ax.plot(a[:,0],a[:,1],ls='none',marker=markers[name],mfc='none',ms=3.0,label=labels[name]+' adaptive discrete')
ax.plot(xfit,pfit,lw=2.0,label='representative constrained fit')
ax.axvline(x_cut,ls='--',lw=.9,label=rf'monotone branch onset $x_c={x_cut:.2e}$')
setup(ax,r'$p_g(x\mid x<0)$')
ax.set_yscale('log'); vv=pfit[pfit>0]; ax.set_ylim(max(1e-3,float(vv.min())*.4),float(max(vv.max(),max(np.max(np.asarray(display[n])[:,1]) for n in display)))*1.7)
ax.legend(frameon=False,ncol=2,loc='best')
fig.tight_layout(pad=.25)
fig.savefig(ROOT/'Nearcontact_monotone_gap_fit_vs_adaptive_discrete_Rodrigues.png',dpi=300,bbox_inches='tight',pad_inches=.02)
fig.savefig(ROOT/'Nearcontact_monotone_gap_fit_vs_adaptive_discrete_Rodrigues.pdf',bbox_inches='tight',pad_inches=.02)
plt.close(fig)

fig,ax=plt.subplots(figsize=(3.3,3.0))
xmin=min(-2e-4,5*x_cut)
mask=xfit>=xmin
ax.plot(xfit[mask],pfit[mask],lw=1.8,label='constrained fit')
for name in ['raw50k','tol5e-5','tol5e-6']:
    a=np.asarray(display[name],float)
    m=a[:,0]>=xmin
    ax.plot(a[m,0],a[m,1],ls='none',marker=markers[name],mfc='none',ms=3.0,label=labels[name])
ax.axvline(x_cut,ls='--',lw=.9)
setup(ax,r'$p_g(x\mid x<0)$',xlim=(xmin,1e-8))
ax.set_yscale('log'); ax.legend(frameon=False,loc='best')
fig.tight_layout(pad=.25)
fig.savefig(ROOT/'Nearcontact_monotone_gap_zoom_adaptive_Rodrigues.png',dpi=300,bbox_inches='tight',pad_inches=.02)
fig.savefig(ROOT/'Nearcontact_monotone_gap_zoom_adaptive_Rodrigues.pdf',bbox_inches='tight',pad_inches=.02)
plt.close(fig)

print('ADAPTIVE_DISCRETE_PLOTS_DONE',flush=True)

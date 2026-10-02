from pathlib import Path
import json, time, hashlib, os
import numpy as np
from scipy.spatial import cKDTree
import rcpgenerator

N=400_000
SEED=20261001
PHI_INIT=0.05
PHI_TARGET=0.6451654159495944
DIAMETER_M=0.00025
QMAX=1.5
THREADS=int(os.environ.get('RCP_THREADS','1'))
OUT=Path('/scratch/gautschi/gonza226/pmi_400k_size_effect_20261001')
OUT.mkdir(parents=True, exist_ok=True)

def main():
    rcpgenerator.set_num_threads(THREADS)
    t0=time.time()
    p=rcpgenerator.Packing(phi=PHI_INIT,N=N,Ndim=3,box=[1,1,1],walls=[0,0,0],dist={'type':'mono','d':1.0},neighbor_max=0,seed=SEED)
    p.pack(verbose=True,progress_interval=5000)
    raw_d=np.asarray(p.diameters,float)
    factor=DIAMETER_M/raw_d.mean()
    box=np.asarray(p.box,float)*factor
    x=np.asarray(p.positions,float)*factor
    x%=box
    nearest=cKDTree(x,boxsize=box).query(x,k=2,workers=THREADS)[0][:,1]
    dilation=max(1.0,DIAMETER_M/nearest.min())*(1+2e-14)
    box*=dilation; x*=dilation
    phi_rcp=N*np.pi/6*DIAMETER_M**3/np.prod(box)
    lam=(phi_rcp/PHI_TARGET)**(1/3)
    box*=lam; x*=lam
    phi=N*np.pi/6*DIAMETER_M**3/np.prod(box)
    assert abs(phi-PHI_TARGET)<5e-13
    tree=cKDTree(x,boxsize=box)
    pairs=tree.query_pairs(QMAX*DIAMETER_M,output_type='ndarray')
    dv=x[pairs[:,1]]-x[pairs[:,0]]
    dv-=box*np.rint(dv/box)
    r=np.linalg.norm(dv,axis=1)
    q=r/DIAMETER_M
    gamma=1.0-q
    contacts=int(np.count_nonzero(gamma>0))
    np.savez_compressed(OUT/'packing_400k.npz',positions_m=x,box_m=box,diameter_m=DIAMETER_M,phi=phi,seed=SEED)
    np.savez_compressed(OUT/'pair_distribution_400k.npz',q=q,gamma_over_2R=gamma,pairs=pairs)
    meta=dict(N=N,seed=SEED,initial_phi=PHI_INIT,target_phi=PHI_TARGET,generated_phi_before_target=phi_rcp,final_phi=phi,diameter_m=DIAMETER_M,
              box_m=box.tolist(),box_ratio_to_50k_equivalent=float((N/50000)**(1/3)),qmax=QMAX,pair_count=int(len(q)),contact_count=contacts,
              contact_fraction=contacts/len(q),threads=THREADS,seconds=time.time()-t0,generator_commit='974d703a655b337b93379dbcd1a6eef47386d674',
              interpretation='Independent 400k periodic cubic realization; homogeneous scaling to accepted PMI phi0; not a tiled copy.')
    (OUT/'metadata.json').write_text(json.dumps(meta,indent=2))
    print(json.dumps(meta,indent=2),flush=True)
if __name__=='__main__': main()

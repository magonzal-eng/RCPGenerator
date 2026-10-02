from pathlib import Path
import json, time, os
import numpy as np
from scipy.spatial import cKDTree
import rcpgenerator

N=50_000
SEED=20260922
PHI_INIT=0.05
QMAX=1.5
EXPECTED_OV_THRESH=5e-5
ROOT=Path('/scratch/gautschi/gonza226/pmi_raw50k_tightov_20261002')
ROOT.mkdir(parents=True,exist_ok=True)

env_tol=float(os.environ.get('RCP_OSC_OV_THRESH','nan'))
if not np.isfinite(env_tol) or abs(env_tol-EXPECTED_OV_THRESH)>1e-15:
    raise RuntimeError(f'RCP_OSC_OV_THRESH={env_tol!r}, expected {EXPECTED_OV_THRESH}')

rcpgenerator.set_num_threads(1)
t0=time.time()
p=rcpgenerator.Packing(phi=PHI_INIT,N=N,Ndim=3,box=[1,1,1],walls=[0,0,0],
                       dist={'type':'mono','d':1.0},neighbor_max=0,seed=SEED)
p.pack(verbose=True,progress_interval=2000)

x=np.asarray(p.positions,float).copy()
box=np.asarray(p.box,float).copy()
diam=np.asarray(p.diameters,float).copy()
x%=box
d0=float(np.mean(diam))
phi=float(np.sum(np.pi/6.0*diam**3)/np.prod(box))

tree=cKDTree(x,boxsize=box)
pairs=tree.query_pairs(QMAX*d0,output_type='ndarray')
dv=x[pairs[:,1]]-x[pairs[:,0]]
dv-=box*np.rint(dv/box)
q=np.linalg.norm(dv,axis=1)/d0
gamma=1.0-q
contacts=int(np.count_nonzero(q<1.0))

np.savez_compressed(ROOT/'packing_50k_tighter_overlap.npz',
                    positions=x,box=box,diameters=diam,phi=phi,seed=SEED)
np.savez_compressed(ROOT/'pair_distribution_50k_tighter_overlap.npz',
                    q=q,gamma_over_2R=gamma)

meta={
 'N':N,
 'seed':SEED,
 'generator_commit':'974d703a655b337b93379dbcd1a6eef47386d674',
 'generator_initial_phi':PHI_INIT,
 'RCP_OSC_OV_THRESH':env_tol,
 'baseline_RCP_OSC_OV_THRESH':5e-4,
 'tolerance_ratio':env_tol/5e-4,
 'phi_as_generated':phi,
 'box':box.tolist(),
 'mean_diameter':d0,
 'pair_count_q_le_1p5':int(len(q)),
 'contact_count_strict_q_lt_1':contacts,
 'contact_fraction_strict':contacts/len(q),
 'generator_reported_steps':int(p.steps),
 'generator_reported_phi_final':float(p.phi_final),
 'generator_reported_force_magnitude':float(p.force_magnitude),
 'generator_reported_max_fractional_overlap':float(p.max_min_dist),
 'preparation':'RCPGenerator pack() output exactly; no dilation, density matching, preload, affine rescaling, or mechanical relaxation',
 'seconds':time.time()-t0
}
(ROOT/'metadata.json').write_text(json.dumps(meta,indent=2))
print('TIGHT50K_METADATA='+json.dumps(meta,sort_keys=True),flush=True)
print('TIGHT50K_RESULT_DIR='+str(ROOT),flush=True)

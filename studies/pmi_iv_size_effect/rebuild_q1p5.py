from pathlib import Path
import json, numpy as np
from scipy.spatial import cKDTree
ROOT=Path('/scratch/gautschi/gonza226/pmi_400k_size_effect_20261001/hertz_relaxed')
with np.load(ROOT/'packing_400k_hertz_relaxed.npz') as z:
    x=np.asarray(z['positions_over_d'],float)
    cell=np.asarray(z['cell_over_d'],float)
lengths=np.diag(cell)
tree=cKDTree(x%lengths,boxsize=lengths)
pairs=tree.query_pairs(1.5,output_type='ndarray')
dv=x[pairs[:,1]]-x[pairs[:,0]]
dv-=lengths*np.rint(dv/lengths)
q=np.linalg.norm(dv,axis=1)
keep=q<=1.5
q=q[keep]
out=ROOT/'pair_distribution_400k_hertz_relaxed_q1p5.npz'
np.savez_compressed(out,q=q,gamma_over_2R=1-q)
contacts=int(np.count_nonzero(q<1))
receipt={'pair_count_q_le_1p5':int(len(q)),'contact_count':contacts,'contact_fraction':contacts/len(q),'q_min':float(q.min()),'q_max':float(q.max())}
(ROOT/'pair_reservoir_q1p5_receipt.json').write_text(json.dumps(receipt,indent=2))
print('PMI400K_RELAXED_Q1P5='+json.dumps(receipt,sort_keys=True),flush=True)

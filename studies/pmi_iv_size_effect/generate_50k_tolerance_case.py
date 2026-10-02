from pathlib import Path
import json, time, os, math
import numpy as np
from scipy.spatial import cKDTree
import rcpgenerator

N = 50_000
SEED = 20260922
PHI_INIT = 0.05
QMAX = 1.5
COMMIT = "974d703a655b337b93379dbcd1a6eef47386d674"

tol = float(os.environ["RCP_OSC_OV_THRESH"])
outdir = Path(os.environ["PMI_TOL_OUTDIR"])
tag = os.environ["PMI_TOL_TAG"]
outdir.mkdir(parents=True, exist_ok=True)

if not (tol > 0 and np.isfinite(tol)):
    raise RuntimeError(f"invalid RCP_OSC_OV_THRESH={tol!r}")

rcpgenerator.set_num_threads(1)
t0 = time.time()

p = rcpgenerator.Packing(
    phi=PHI_INIT, N=N, Ndim=3, box=[1,1,1], walls=[0,0,0],
    dist={"type":"mono","d":1.0}, neighbor_max=0, seed=SEED
)
p.pack(verbose=True, progress_interval=2000)

x = np.asarray(p.positions, float).copy()
box = np.asarray(p.box, float).copy()
diam = np.asarray(p.diameters, float).copy()
x %= box
d0 = float(np.mean(diam))
phi = float(np.sum(np.pi/6.0 * diam**3) / np.prod(box))

tree = cKDTree(x, boxsize=box)
pairs = tree.query_pairs(QMAX*d0, output_type="ndarray")
dv = x[pairs[:,1]] - x[pairs[:,0]]
dv -= box * np.rint(dv/box)
q = np.linalg.norm(dv, axis=1) / d0
gamma = 1.0 - q
contacts = int(np.count_nonzero(q < 1.0))
max_overlap_rebuilt = float(max(0.0, np.max(gamma))) if len(gamma) else 0.0

np.savez_compressed(
    outdir / f"packing_50k_{tag}.npz",
    positions=x, box=box, diameters=diam, phi=phi, seed=SEED
)
np.savez_compressed(
    outdir / f"pair_distribution_50k_{tag}.npz",
    q=q, gamma_over_2R=gamma
)

meta = {
    "tag": tag,
    "N": N,
    "seed": SEED,
    "generator_commit": COMMIT,
    "generator_initial_phi": PHI_INIT,
    "RCP_OSC_OV_THRESH": tol,
    "phi_as_generated": phi,
    "box": box.tolist(),
    "mean_diameter": d0,
    "pair_count_q_le_1p5": int(len(q)),
    "contact_count_strict_q_lt_1": contacts,
    "contact_fraction_strict": contacts / len(q),
    "generator_reported_steps": int(p.steps),
    "generator_reported_phi_final": float(p.phi_final),
    "generator_reported_force_magnitude": float(p.force_magnitude),
    "generator_reported_max_fractional_overlap": float(p.max_min_dist),
    "rebuilt_max_fractional_overlap": max_overlap_rebuilt,
    "threshold_satisfied_generator_report": bool(float(p.max_min_dist) <= tol),
    "threshold_satisfied_rebuilt_pairs": bool(max_overlap_rebuilt <= tol),
    "preparation": "RCPGenerator pack() endpoint exactly; no dilation, density matching, preload, affine rescaling, or mechanical relaxation",
    "seconds": time.time() - t0,
}
(outdir / "metadata.json").write_text(json.dumps(meta, indent=2))
print("PMI_TOL_METADATA=" + json.dumps(meta, sort_keys=True), flush=True)
print("PMI_TOL_RESULT_DIR=" + str(outdir), flush=True)

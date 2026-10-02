from pathlib import Path
import hashlib, json, sys, time
import numpy as np

ROOT = Path("/home/gonza226/multiparticle_die_solver/pmi_hertz_equilibrium_20260923c")
sys.path.insert(0, str(ROOT))
from practical_relax import relax, POLICY
from mechanics import candidates, evaluate

IN = Path("/scratch/gautschi/gonza226/pmi_400k_size_effect_20261001/packing_400k.npz")
OUT = Path("/scratch/gautschi/gonza226/pmi_400k_size_effect_20261001/hertz_relaxed")
OUT.mkdir(parents=True, exist_ok=True)

with np.load(IN) as z:
    x_m = np.asarray(z["positions_m"], float)
    box_m = np.asarray(z["box_m"], float)
    d = float(z["diameter_m"])
    phi = float(z["phi"])

x = np.ascontiguousarray(x_m / d)
cell = np.diag(box_m / d)
if len(x) != 400000:
    raise RuntimeError(f"expected 400000 particles, got {len(x)}")
phi_check = len(x) * np.pi / 6 / np.linalg.det(cell)
if abs(phi_check - phi) > 5e-13:
    raise RuntimeError((phi_check, phi))

initial_graph = candidates(x, cell)
_, _, initial = evaluate(x, initial_graph, jacobian=False)
t0 = time.time()
xr, state = relax(x, cell, OUT / "diagnostics", 0)
final_graph = candidates(xr, cell)
_, _, final = evaluate(xr, final_graph, jacobian=False)

np.savez_compressed(
    OUT / "packing_400k_hertz_relaxed.npz",
    positions_m=xr*d, box_m=box_m, diameter_m=d, phi=phi,
    positions_over_d=xr, cell_over_d=cell,
)
pairs = final_graph.pairs
branch = xr[pairs[:,1]] - xr[pairs[:,0]] + final_graph.images @ cell.T
q = np.linalg.norm(branch, axis=1)
q = q[q <= 1.5]
np.savez_compressed(
    OUT / "pair_distribution_400k_hertz_relaxed.npz",
    q=q, gamma_over_2R=1-q,
)
receipt = dict(
    source_packing=str(IN),
    source_sha256=hashlib.sha256(IN.read_bytes()).hexdigest(),
    N=len(xr), phi=phi, diameter_m=d, box_m=box_m.tolist(),
    initial_contacts=int(initial["contacts"]),
    initial_force_ratio=float(initial["force_ratio"]),
    initial_max_overlap=float(initial["max_overlap"]),
    initial_energy=float(initial["energy"]),
    final_contacts=int(final["contacts"]),
    final_force_ratio=float(final["force_ratio"]),
    final_max_overlap=float(final["max_overlap"]),
    final_energy=float(final["energy"]),
    retained_pairs_q_le_1p5=int(len(q)),
    relaxation_state=state, policy=POLICY,
    seconds_total=time.time()-t0,
    interpretation="Independent 400k packing relaxed at fixed periodic cell with revision-c practical Hertz endpoint policy before size-effect comparison.",
)
(OUT/"relaxation_receipt.json").write_text(
    json.dumps(receipt, indent=2, default=lambda o:o.tolist() if hasattr(o,"tolist") else o)
)
compact={k:v for k,v in receipt.items() if k not in ("relaxation_state","policy")}
print("PMI400K_HERTZ_RELAX_RECEIPT="+json.dumps(compact,sort_keys=True),flush=True)

import os
import shutil
import subprocess
import torch

from src.feature_extraction.feature_extraction import create_features_from_a3m
from src.geometry.residue_constants import restypes
from src.model.model import Model
from src.model.utils import load_openfold_weights, to_modelcif

device = "mps" if torch.backends.mps.is_available() else "cpu"

model = Model()
openfold_weigths = load_openfold_weights('data/openfold_params/finetuning_2.pt')

model.load_state_dict(openfold_weigths)
model.to(device)

single_cycle_batches = []
for i in range(4):
    single_cycle_batch = create_features_from_a3m('data/alignment_tautomerase.a3m')
    single_cycle_batches.append(single_cycle_batch)

batch = {
    key: torch.stack([single_batch[key] for single_batch in single_cycle_batches], dim=-1)
    for key in single_cycle_batches[0].keys()
}

for key, value in batch.items():
    batch[key] = value.to(device)

model.eval()
with torch.no_grad():
    outputs = model(batch)

atom_positions = outputs['final_positions'][..., -1]
atom_mask = outputs['position_mask'][..., -1]
seq_inds = batch['target_feat'].cpu()[..., -1].argmax(dim=-1).numpy()
seq = ''.join([restypes[ind] for ind in seq_inds])

cif_str = to_modelcif(atom_positions, atom_mask, seq)

output_dir = "data"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "predicted_structure.cif")

with open(output_path, "w") as f:
    f.write(cif_str)

print(f"Structure saved successfully to {output_path}")

chimerax_path = shutil.which("chimerax")

if not chimerax_path:
    default_mac_path = "/Applications/ChimeraX-1.12.app/Contents/MacOS/ChimeraX"
    if os.path.exists(default_mac_path):
        chimerax_path = default_mac_path

if chimerax_path:
    print(f"Launching ChimeraX via: {chimerax_path}")
    subprocess.Popen([
        chimerax_path, 
        "--cmd", f"open {output_path}",
        "--cmd", "preset cartoon", 
        "--cmd", "rainbow",
        "--cmd", "zoom"
    ])
else:
    print(f"\n[Warning] ChimeraX executable could not be found automatically.")
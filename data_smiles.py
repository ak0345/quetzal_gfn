"""
geom_smiles.py -- Download GEOM-Drugs, extract SMILES only, clean, split.

Produces a clean, canonical, deduplicated SMILES list matching the base model's
TRAIN split -- ready to use as an in-distribution reference.

    pip install msgpack rdkit numpy tqdm

Outputs in data/:
    geom_drugs_smiles.txt      all raw SMILES (msgpack keys, in dataset order)
    geom_train_smiles.txt      cleaned + canonical + deduped TRAIN split  <-- USE THIS
    geom_val_smiles.txt        cleaned val split
    geom_test_smiles.txt       cleaned test split
"""

import os
import subprocess

import numpy as np
import msgpack
from tqdm import tqdm
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors

RDLogger.DisableLog("rdApp.*")            # silence parse spam

# ------------------------------------------------------------------ config
DRUGS_URL = "https://dataverse.harvard.edu/api/access/datafile/4360331"
PERM_URL = "https://github.com/ehoogeboom/e3_diffusion_for_molecules/raw/fce07d701a2d2340f3522df588832c2c0f7e044a/data/geom/geom_permutation.npy"



ROOT = "data"
os.makedirs(ROOT, exist_ok=True)
ARCHIVE   = os.path.join(ROOT, "drugs_crude.tar")
MSGPACK   = os.path.join(ROOT, "drugs_crude.msgpack")
PERM      = os.path.join(ROOT, "geom_permutation.npy")
RAW_SMI   = os.path.join(ROOT, "geom_drugs_smiles.txt")


# ------------------------------------------------------------------ download
def download():
    if not os.path.exists(MSGPACK):
        if not os.path.exists(ARCHIVE):
            print("Downloading GEOM drugs (~7GB)...")
            subprocess.run(["curl", "-L", "--retry", "5", "--max-time", "3600",
                "-o", ARCHIVE, DRUGS_URL], check=True)
        print("Extracting...")
        subprocess.run(["tar", "-xf", ARCHIVE, "-C", ROOT], check=True)
        # the msgpack may extract under a different name/subfolder; find it
        if not os.path.exists(MSGPACK):
            hits = subprocess.run(["find", ROOT, "-name", "*.msgpack"],
                                  capture_output=True, text=True).stdout.split()
            if hits:
                os.rename(hits[0], MSGPACK)
            else:
                raise FileNotFoundError("no .msgpack found after extraction")
    if not os.path.exists(PERM):
        print("Downloading split permutation...")
        subprocess.run(["curl", "-L", "-o", PERM, PERM_URL], check=True)


# ------------------------------------------------------------------ extract
def extract_raw_smiles():
    """SMILES are the msgpack keys. No conformer processing.

    IMPORTANT: order matters. The split permutation indexes molecules in the
    order they appear here, so we must NOT reorder or dedupe before splitting.
    """
    if os.path.exists(RAW_SMI):
        smiles = open(RAW_SMI).read().splitlines()
        print(f"  {len(smiles)} raw SMILES already extracted")
        return smiles
    smiles = []
    for i, chunk in enumerate(msgpack.Unpacker(open(MSGPACK, "rb"))):
        print(f"  unpacking chunk {i}...")
        smiles.extend(chunk.keys())
    with open(RAW_SMI, "w") as f:
        f.write("\n".join(smiles))
    print(f"  extracted {len(smiles)} raw SMILES")
    return smiles


# ------------------------------------------------------------------ clean
# GEOM-Drugs atom set used by the base model (Hoogeboom/Vignac EDM).
ALLOWED_ATOMS = {"H", "B", "C", "N", "O", "F", "Al", "Si", "P", "S",
                 "Cl", "As", "Br", "I", "Hg", "Bi"}


def clean_one(smi):
    """Canonicalize and filter one SMILES. Returns canonical SMILES or None.

    Filters, matched to how GEOM generative models treat the data:
      - must parse and sanitize in RDKit
      - single fragment (drop salts / mixtures -- '.' in SMILES)
      - only the GEOM atom set
      - drop charged species (EDM data is neutral)
    Adjust these to match the base model's preprocessing where it differs;
    an in-distribution claim is only as honest as the filter matching.
    """
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    if "." in smi:                                    # salt / mixture
        return None
    if any(a.GetSymbol() not in ALLOWED_ATOMS for a in mol.GetAtoms()):
        return None
    if Chem.GetFormalCharge(mol) != 0:                # neutral only
        return None
    return Chem.MolToSmiles(mol)                      # canonical form


def clean_list(smiles):
    """Clean a list, PRESERVING ORDER and length via None placeholders.

    We keep None for failures so indices still line up with the split
    permutation. Dedup happens AFTER splitting, within each split.
    """
    out = []
    for s in tqdm(smiles, desc="cleaning"):
        out.append(clean_one(s))
    n_ok = sum(x is not None for x in out)
    print(f"  {n_ok}/{len(out)} passed cleaning ({100*n_ok/len(out):.1f}%)")
    return out


# ------------------------------------------------------------------ split
def split_and_save(clean_smiles):
    """Apply the base model's exact split, then dedupe within each split.

    The permutation order is (val, test, train) with 10%/10%/80% -- IDENTICAL
    to the base model's load_split_data(), so geom_train_smiles.txt is precisely
    the set the base model trained on.
    """
    perm = np.load(PERM)
    n = len(perm)
    assert n == len(clean_smiles), (
        f"permutation length {n} != #molecules {len(clean_smiles)}; "
        "did you reorder or dedupe before splitting? Don't.")

    val_split = int(n * 0.1)
    test_split = val_split + int(n * 0.1)
    val_idx, test_idx, train_idx = np.split(perm, [val_split, test_split])

    arr = np.array(clean_smiles, dtype=object)
    for name, idx in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
        sel = [arr[i] for i in idx if arr[i] is not None]     # drop failures
        deduped = list(dict.fromkeys(sel))                    # order-preserving unique
        path = os.path.join(ROOT, f"geom_{name}_smiles.txt")
        with open(path, "w") as f:
            f.write("\n".join(deduped))
        print(f"  {name}: {len(deduped)} unique clean SMILES -> {path}")


# ------------------------------------------------------------------ driver
if __name__ == "__main__":
    #print("1. download")
    #download()

    print("2. extract raw SMILES")
    raw = extract_raw_smiles()

    print("3. clean (canonicalize + filter)")
    cleaned = clean_list(raw)

    print("4. split (matches base model) + dedupe")
    split_and_save(cleaned)

    print("\nDone. Use data/geom_train_smiles.txt as your in-distribution reference.")
    print("Feed it straight into eval_rewards.py's score_file().")
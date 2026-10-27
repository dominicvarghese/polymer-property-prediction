"""
Polymer-specific SMILES and Molecular Graph Preprocessing in RDKit and PyTorch Geometric.
Implements star-atom heavy-atom substitution, programmatic oligomer extension (n=3),
periodic inter-monomer edges, central repeat-unit masks, RWSE, and physics priors.
"""
import math
import numpy as np
import torch
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, Descriptors, GraphDescriptors, Lipinski, MolSurf
from torch_geometric.data import Data

RDLogger.DisableLog("rdApp.*")

ALLOWED_ATOMS = [6, 7, 8, 9, 14, 15, 16, 17, 35, 53, 5, 32, 50, 34, 1]
ATOM_TO_IDX = {z: i for i, z in enumerate(ALLOWED_ATOMS)}
ELECTRONEGATIVITY = {1: 2.20, 5: 2.04, 6: 2.55, 7: 3.04, 8: 3.44, 9: 3.98, 14: 1.90, 15: 2.19, 16: 2.58, 17: 3.16, 32: 2.01, 34: 2.55, 35: 2.96, 50: 1.96, 53: 2.66, 0: 2.55}
HYBRIDIZATIONS = [Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2, Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D, Chem.rdchem.HybridizationType.SP3D2]
BOND_TYPES = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC]
STEREO_TYPES = [Chem.rdchem.BondStereo.STEREONONE, Chem.rdchem.BondStereo.STEREOANY, Chem.rdchem.BondStereo.STEREOZ, Chem.rdchem.BondStereo.STEREOE]
PTABLE = Chem.GetPeriodicTable()


def _find_attachment_sites(mol):
    sites = []
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 0 and len(atom.GetBonds()) >= 1:
            b = atom.GetBonds()[0]
            sites.append((atom.GetIdx(), b.GetOtherAtomIdx(atom.GetIdx()), b.GetBondType()))
    return sites


def _select_primary_pair(mol, sites):
    if len(sites) == 2:
        return sites[0], sites[1], []
    dmat = Chem.GetDistanceMatrix(mol)
    best_pair, best_d = (0, 1), -1.0
    for i in range(len(sites)):
        for j in range(i + 1, len(sites)):
            d = dmat[sites[i][1], sites[j][1]]
            if d > best_d:
                best_d, best_pair = d, (i, j)
    i, j = best_pair
    return sites[i], sites[j], [sites[k] for k in range(len(sites)) if k not in (i, j)]


def _compute_backbone_mask(mol, end_atoms):
    n_atoms = mol.GetNumAtoms()
    is_bb = np.zeros(n_atoms, dtype=np.float32)
    bb_dist = np.zeros(n_atoms, dtype=np.float32)
    if len(end_atoms) < 2 or n_atoms == 0:
        return is_bb, bb_dist, 1.0
    u, v = int(end_atoms[0]), int(end_atoms[1])
    if u >= n_atoms or v >= n_atoms:
        return is_bb, bb_dist, 1.0
    try:
        path = Chem.rdmolops.GetShortestPath(mol, u, v)
        if len(path) == 0:
            path = (u, v)
        span = float(max(len(path) - 1, 1))
    except Exception:
        path, span = (u, v), 1.0

    bb_set = set(path)
    try:
        for ring in mol.GetRingInfo().AtomRings():
            if len(bb_set.intersection(ring)) >= 2:
                bb_set.update(ring)
    except Exception:
        pass
    bb_indices = sorted(bb_set)
    for idx in bb_indices:
        is_bb[idx] = 1.0
    if len(bb_indices) > 0:
        try:
            dmat = Chem.GetDistanceMatrix(mol)
            min_d = dmat[:, bb_indices].min(axis=1)
            bb_dist = (np.where(min_d > 1e5, 0.0, min_d) / 10.0).astype(np.float32)
        except Exception:
            pass
    return is_bb, bb_dist, max(span, 1.0)


def build_polymer_mol_representation(smiles, mode="oligomer_3_periodic", cap_atomic_num=6):
    base_mol = Chem.MolFromSmiles(smiles)
    sites = _find_attachment_sites(base_mol)

    if mode == "monomer_raw" or len(sites) == 0:
        n_atoms = base_mol.GetNumAtoms()
        is_att = np.zeros(n_atoms, dtype=np.float32)
        is_cap = np.zeros(n_atoms, dtype=np.float32)
        for s_idx, nbr_idx, _ in sites:
            is_att[nbr_idx] = 1.0
            is_cap[s_idx] = 1.0
        is_bb, bb_dist, span = _compute_backbone_mask(base_mol, [s[1] for s in sites[:2]] if len(sites) >= 2 else [])
        return base_mol, {"is_attachment_neighbor": is_att, "is_cap_atom": is_cap, "is_on_backbone": is_bb, "backbone_dist": bb_dist, "inter_monomer_bonds": [], "periodic_pair": None, "topo_span": span}

    if mode == "monomer_capped" or len(sites) == 1:
        rw = Chem.RWMol(base_mol)
        n_atoms = rw.GetNumAtoms()
        is_att = np.zeros(n_atoms, dtype=np.float32)
        is_cap = np.zeros(n_atoms, dtype=np.float32)
        for s_idx, nbr_idx, _ in sites:
            rw.GetAtomWithIdx(s_idx).SetAtomicNum(cap_atomic_num)
            rw.GetAtomWithIdx(s_idx).SetIsotope(0)
            is_att[nbr_idx] = 1.0
            is_cap[s_idx] = 1.0
        mol = rw.GetMol()
        try:
            Chem.SanitizeMol(mol)
        except Exception:
            pass
        ends = [_select_primary_pair(base_mol, sites)[0][1], _select_primary_pair(base_mol, sites)[1][1]] if len(sites) >= 2 else []
        is_bb, bb_dist, span = _compute_backbone_mask(mol, ends)
        return mol, {"is_attachment_neighbor": is_att, "is_cap_atom": is_cap, "is_on_backbone": is_bb, "backbone_dist": bb_dist, "inter_monomer_bonds": [], "periodic_pair": None, "topo_span": span}

    n_repeats = 3
    use_periodic = "periodic" in mode
    h_site, t_site, rem_sites = _select_primary_pair(base_mol, sites)
    _, _, monomer_span = _compute_backbone_mask(base_mol, [h_site[1], t_site[1]])

    template_rw = Chem.RWMol(base_mol)
    for s_idx, _, _ in rem_sites:
        template_rw.GetAtomWithIdx(s_idx).SetAtomicNum(cap_atomic_num)
        template_rw.GetAtomWithIdx(s_idx).SetIsotope(0)

    h_star, h_nbr, h_btype = h_site
    t_star, t_nbr, t_btype = t_site
    combined = template_rw.GetMol()
    n_unit = combined.GetNumAtoms()
    rw_olig = Chem.RWMol(combined)
    for _ in range(n_repeats - 1):
        rw_olig.InsertMol(combined)

    stars_to_remove, stars_to_cap, att_neighbors, extra_caps, inter_bonds_orig = set(), set(), set(), set(), []
    for k in range(n_repeats):
        offset = k * n_unit
        att_neighbors.update([offset + h_nbr, offset + t_nbr])
        for s_idx, nbr_idx, _ in rem_sites:
            att_neighbors.add(offset + nbr_idx)
            extra_caps.add(offset + s_idx)
        if k == 0:
            stars_to_cap.add(offset + h_star)
        else:
            stars_to_remove.add(offset + h_star)
        if k == n_repeats - 1:
            stars_to_cap.add(offset + t_star)
        else:
            stars_to_remove.add(offset + t_star)
            u, v = offset + t_nbr, (k + 1) * n_unit + h_nbr
            btype = t_btype if t_btype == h_btype else Chem.rdchem.BondType.SINGLE
            if rw_olig.GetBondBetweenAtoms(u, v) is None:
                rw_olig.AddBond(u, v, btype)
            inter_bonds_orig.append((u, v))

    for idx in stars_to_cap:
        rw_olig.GetAtomWithIdx(idx).SetAtomicNum(cap_atomic_num)
        rw_olig.GetAtomWithIdx(idx).SetIsotope(0)

    keep_indices = [i for i in range(rw_olig.GetNumAtoms()) if i not in stars_to_remove]
    old_to_new = {old_i: new_i for new_i, old_i in enumerate(keep_indices)}
    for idx in sorted(stars_to_remove, reverse=True):
        rw_olig.RemoveAtom(idx)

    olig_mol = rw_olig.GetMol()
    try:
        Chem.SanitizeMol(olig_mol)
    except Exception:
        try:
            Chem.SanitizeMol(olig_mol, Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE)
        except Exception:
            pass

    n_final = olig_mol.GetNumAtoms()
    is_att = np.zeros(n_final, dtype=np.float32)
    is_cap = np.zeros(n_final, dtype=np.float32)
    for old_i in att_neighbors:
        if old_i in old_to_new:
            is_att[old_to_new[old_i]] = 1.0
    for old_i in stars_to_cap | extra_caps:
        if old_i in old_to_new:
            is_cap[old_to_new[old_i]] = 1.0
    inter_bonds = [(old_to_new[u], old_to_new[v]) for (u, v) in inter_bonds_orig if u in old_to_new and v in old_to_new]
    head_end, tail_end = old_to_new[h_nbr], old_to_new[(n_repeats - 1) * n_unit + t_nbr]
    is_bb, bb_dist, _ = _compute_backbone_mask(olig_mol, [head_end, tail_end])

    return olig_mol, {
        "is_attachment_neighbor": is_att,
        "is_cap_atom": is_cap,
        "is_on_backbone": is_bb,
        "backbone_dist": bb_dist,
        "inter_monomer_bonds": inter_bonds,
        "periodic_pair": (head_end, tail_end) if use_periodic else None,
        "topo_span": monomer_span,
    }


def compute_rwse(edge_index, num_nodes, walk_lengths=8):
    if num_nodes == 0 or edge_index.numel() == 0:
        return torch.zeros((num_nodes, walk_lengths), dtype=torch.float32)
    adj = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)
    adj[edge_index[0], edge_index[1]] = 1.0
    rw = adj / adj.sum(dim=1, keepdim=True).clamp(min=1.0)
    rw_p = rw.clone()
    pe = []
    for _ in range(walk_lengths):
        pe.append(torch.diagonal(rw_p))
        rw_p = rw_p @ rw
    return torch.stack(pe, dim=-1)

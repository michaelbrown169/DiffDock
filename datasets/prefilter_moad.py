#!/usr/bin/env python3
"""
Pre-filter MOAD dataset to identify complexes that will successfully load.
This avoids wasting time on complexes that will fail during expensive processing.
"""

import os
import pickle
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, '/media/mike/mass_drive/Documents/diffdock_adaption/DiffDock')

try:
    import torch
    from Bio import PDB
    from rdkit import Chem
    DEPS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Some dependencies not available: {e}")
    DEPS_AVAILABLE = False


def check_esm_sequence_availability(protein_id, esm_embeddings_path, esm_sequences_path):
    """Check if ESM embeddings and sequence are available for a protein."""
    try:
        # Load ESM data
        if not os.path.exists(esm_embeddings_path) or not os.path.exists(esm_sequences_path):
            return False
            
        esm_embeddings = torch.load(esm_embeddings_path, map_location='cpu')
        
        with open(esm_sequences_path, 'r') as f:
            sequences_data = f.read()
            
        # Check if protein_id exists in ESM data
        return protein_id in esm_embeddings and protein_id in sequences_data
        
    except Exception:
        return False


def check_protein_file_validity(protein_path):
    """Check if protein PDB file exists and is readable."""
    try:
        if not os.path.exists(protein_path):
            return False
            
        parser = PDB.PDBParser(QUIET=True)
        structure = parser.get_structure('protein', protein_path)
        
        # Check if structure has at least one chain with atoms
        atom_count = sum(1 for atom in structure.get_atoms())
        return atom_count > 0
        
    except Exception:
        return False


def check_ligand_file_validity(ligand_path):
    """Check if ligand file exists and is readable."""
    try:
        if not os.path.exists(ligand_path):
            return False
            
        if ligand_path.endswith('.mol2'):
            mol = Chem.MolFromMol2File(ligand_path, sanitize=False, removeHs=False)
        elif ligand_path.endswith('.pdb'):
            mol = Chem.MolFromPDBFile(ligand_path, sanitize=False, removeHs=False)
        else:
            return False
            
        return mol is not None and mol.GetNumAtoms() > 0
        
    except Exception:
        return False


def prefilter_moad_complexes(moad_dir, esm_embeddings_path, esm_sequences_path, 
                           output_path='data/moad_prefiltered_complexes.pkl', 
                           max_complexes=None):
    """
    Pre-filter MOAD complexes to identify which ones will successfully load.
    
    Returns:
        dict: {complex_name: {'protein_id': str, 'ligand_path': str, 'protein_path': str, 'valid': bool}}
    """
    
    print("🔍 Pre-filtering MOAD complexes...")
    
    # Get all ligand files
    ligand_dir = os.path.join(moad_dir, 'pdb_superligand')
    protein_dir = os.path.join(moad_dir, 'pdb_protein')
    
    ligand_files = []
    for ext in ['*.pdb', '*.mol2']:
        ligand_files.extend(Path(ligand_dir).glob(ext))
    
    if max_complexes:
        ligand_files = ligand_files[:max_complexes]
    
    print(f"Found {len(ligand_files)} ligand files to check")
    
    valid_complexes = []
    invalid_reasons = {}
    
    for ligand_file in tqdm(ligand_files, desc="Checking complexes"):
        complex_name = ligand_file.stem
        protein_id = complex_name[:6]  # First 6 characters are PDB ID
        
        protein_path = os.path.join(protein_dir, f"{protein_id}.pdb")
        
        # Check all requirements
        checks = {
            'ligand_file_valid': check_ligand_file_validity(str(ligand_file)),
            'protein_file_exists': os.path.exists(protein_path),
            'protein_file_valid': False,
            'esm_available': False
        }
        
        if checks['protein_file_exists']:
            checks['protein_file_valid'] = check_protein_file_validity(protein_path)
            
        if checks['protein_file_valid']:
            checks['esm_available'] = check_esm_sequence_availability(
                protein_id, esm_embeddings_path, esm_sequences_path)
        
        # Complex is valid if all checks pass
        is_valid = all(checks.values())
        
        complex_info = {
            'complex_name': complex_name,
            'protein_id': protein_id,
            'ligand_path': str(ligand_file),
            'protein_path': protein_path,
            'valid': is_valid,
            'checks': checks
        }
        
        if is_valid:
            valid_complexes.append(complex_info)
        else:
            # Record why it failed
            failed_checks = [k for k, v in checks.items() if not v]
            invalid_reasons[complex_name] = failed_checks
    
    print(f"\n✅ Valid complexes: {len(valid_complexes)}")
    print(f"❌ Invalid complexes: {len(ligand_files) - len(valid_complexes)}")
    
    # Show failure statistics
    if invalid_reasons:
        failure_stats = {}
        for reasons in invalid_reasons.values():
            for reason in reasons:
                failure_stats[reason] = failure_stats.get(reason, 0) + 1
        
        print("\n📊 Failure reasons:")
        for reason, count in sorted(failure_stats.items(), key=lambda x: x[1], reverse=True):
            print(f"  {reason}: {count} complexes")
    
    # Save results
    results = {
        'valid_complexes': valid_complexes,
        'invalid_reasons': invalid_reasons,
        'stats': {
            'total_checked': len(ligand_files),
            'valid': len(valid_complexes),
            'invalid': len(ligand_files) - len(valid_complexes),
            'success_rate': len(valid_complexes) / len(ligand_files) if ligand_files else 0
        }
    }
    
    with open(output_path, 'wb') as f:
        pickle.dump(results, f)
    
    print(f"\n💾 Results saved to {output_path}")
    print(f"🎯 Success rate: {results['stats']['success_rate']:.1%}")
    
    return results


if __name__ == "__main__":
    # Example usage
    results = prefilter_moad_complexes(
        moad_dir='data/BindingMOAD_2020_ab_processed_biounit',
        esm_embeddings_path='data/moad_esm2_embeddings.pt',
        esm_sequences_path='data/moad_sequences.txt',
        max_complexes=1000  # Test with first 1000 complexes
    )
    
    print(f"\nFound {len(results['valid_complexes'])} valid complexes out of {results['stats']['total_checked']} checked")
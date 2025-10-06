#!/usr/bin/env python3
"""
Create a filtered whitelist of MOAD complexes based on file existence.
This allows the dataset loader to skip problematic complexes early.
"""

import os
from pathlib import Path
import pickle


def create_moad_whitelist(moad_dir, max_complexes=None):
    """
    Create a whitelist of complexes where both protein and ligand files exist.
    
    Args:
        moad_dir: Path to MOAD dataset
        max_complexes: Limit number of complexes to check (None for all)
        
    Returns:
        Set of valid complex names
    """
    
    print("🔍 Creating MOAD whitelist based on file existence...")
    
    ligand_dir = os.path.join(moad_dir, 'pdb_superligand')
    protein_dir = os.path.join(moad_dir, 'pdb_protein')
    
    # Get all ligand files
    ligand_files = list(Path(ligand_dir).glob('*.pdb'))
    
    if max_complexes:
        ligand_files = ligand_files[:max_complexes]
    
    print(f"Found {len(ligand_files)} ligand files to check")
    
    valid_complexes = set()
    total_checked = 0
    
    # Group ligands by protein to reduce redundant protein file checks
    protein_to_ligands = {}
    for ligand_file in ligand_files:
        complex_name = ligand_file.stem
        # Extract protein ID from complex name (e.g., "10gs_1_superlig_0" -> "10gs_1")
        parts = complex_name.split('_')
        if len(parts) >= 4:  # Should be: pdb_id, chain_id, "superlig", ligand_id
            protein_id = f"{parts[0]}_{parts[1]}"
            if protein_id not in protein_to_ligands:
                protein_to_ligands[protein_id] = []
            protein_to_ligands[protein_id].append(complex_name)
    
    print(f"Found {len(protein_to_ligands)} unique proteins to check")
    
    # Check each protein once
    for protein_id, ligand_names in protein_to_ligands.items():
        total_checked += 1
        if total_checked % 1000 == 0:
            print(f"Checked {total_checked}/{len(protein_to_ligands)} proteins...")
            
        protein_path = os.path.join(protein_dir, f"{protein_id}_protein.pdb")
        
        if os.path.exists(protein_path):
            # If protein exists, add all its ligands to valid set
            for ligand_name in ligand_names:
                ligand_path = os.path.join(ligand_dir, f"{ligand_name}.pdb")
                if os.path.exists(ligand_path):
                    valid_complexes.add(ligand_name)
    
    print(f"\n✅ Valid complexes: {len(valid_complexes)}")
    print(f"❌ Invalid complexes: {len(ligand_files) - len(valid_complexes)}")
    print(f"🎯 Success rate: {len(valid_complexes) / len(ligand_files) * 100:.1f}%")
    
    return valid_complexes


def save_moad_whitelist(moad_dir, output_path='data/moad_whitelist.pkl', max_complexes=None):
    """Save the whitelist to a file for use in training."""
    
    valid_complexes = create_moad_whitelist(moad_dir, max_complexes)
    
    # Save as both pickle and text
    whitelist_data = {
        'valid_complexes': sorted(list(valid_complexes)),
        'total_valid': len(valid_complexes),
        'created_with_max_complexes': max_complexes
    }
    
    with open(output_path, 'wb') as f:
        pickle.dump(whitelist_data, f)
    
    # Also save as plain text for easy inspection
    txt_path = output_path.replace('.pkl', '.txt')
    with open(txt_path, 'w') as f:
        for complex_name in sorted(valid_complexes):
            f.write(f"{complex_name}\n")
    
    print(f"\n💾 Saved whitelist to {output_path}")
    print(f"💾 Saved text version to {txt_path}")
    
    return whitelist_data


if __name__ == "__main__":
    # Create whitelist for full dataset
    whitelist = save_moad_whitelist(
        moad_dir='data/BindingMOAD_2020_ab_processed_biounit',
        output_path='data/moad_whitelist.pkl'
    )
    
    print(f"\nWhitelist contains {whitelist['total_valid']} valid complexes")
    print("You can now use this whitelist to speed up training by pre-filtering complexes!")
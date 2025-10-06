#!/usr/bin/env python3
"""
Create a more targeted whitelist that only includes complexes from train/val splits
"""

import pickle
import os

def create_targeted_whitelist():
    # Load original whitelist
    with open('data/moad_whitelist.pkl', 'rb') as f:
        whitelist_data = pickle.load(f)
        
    if isinstance(whitelist_data, dict) and 'valid_complexes' in whitelist_data:
        all_valid_complexes = set(whitelist_data['valid_complexes'])
    else:
        all_valid_complexes = set(whitelist_data)
    
    print(f"Original whitelist: {len(all_valid_complexes)} complexes")
    
    # Load train and val splits
    with open('data/splits/timesplit_no_lig_overlap_train', 'r') as f:
        train_pdbs = set(line.strip() for line in f)
    
    with open('data/splits/timesplit_no_lig_overlap_val', 'r') as f:
        val_pdbs = set(line.strip() for line in f)
    
    print(f"Train PDbs: {len(train_pdbs)}")
    print(f"Val PDBs: {len(val_pdbs)}")
    
    # Filter whitelist to only include complexes from train/val splits
    train_complexes = {complex_name for complex_name in all_valid_complexes 
                      if any(complex_name.startswith(pdb + '_') for pdb in train_pdbs)}
    
    val_complexes = {complex_name for complex_name in all_valid_complexes 
                    if any(complex_name.startswith(pdb + '_') for pdb in val_pdbs)}
    
    targeted_complexes = train_complexes | val_complexes
    
    print(f"Train complexes in whitelist: {len(train_complexes)}")  
    print(f"Val complexes in whitelist: {len(val_complexes)}")
    print(f"Total targeted complexes: {len(targeted_complexes)}")
    
    # Save targeted whitelist
    targeted_data = {
        'valid_complexes': list(targeted_complexes),
        'train_complexes': list(train_complexes),
        'val_complexes': list(val_complexes),
        'total_valid': len(targeted_complexes),
        'created_with_splits': True
    }
    
    with open('data/moad_targeted_whitelist.pkl', 'wb') as f:
        pickle.dump(targeted_data, f)
    
    # Also save text version
    with open('data/moad_targeted_whitelist.txt', 'w') as f:
        for complex_name in sorted(targeted_complexes):
            f.write(f"{complex_name}\n")
    
    print(f"Saved targeted whitelist to data/moad_targeted_whitelist.pkl")
    print(f"Saved text version to data/moad_targeted_whitelist.txt")
    
    # Show some examples
    print(f"\nSample train complexes: {sorted(train_complexes)[:5]}")
    print(f"Sample val complexes: {sorted(val_complexes)[:5]}")

if __name__ == "__main__":
    create_targeted_whitelist()
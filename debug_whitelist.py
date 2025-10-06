#!/usr/bin/env python3
"""
Debug whitelist integration to see what complexes are being filtered and why
"""

import os
import torch
import pickle
from functools import partial
from datasets.moad import MOAD
from datasets.loader import construct_loader
from utils.parsing import parse_train_args
from utils.diffusion_utils import t_to_sigma as t_to_sigma_compl

def debug_whitelist():
    args = parse_train_args()
    args.dataset = "moad"
    args.moad_dir = "data/BindingMOAD_2020_ab_processed_biounit/"
    args.limit_complexes = 20  # Small number for debugging
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load whitelist
    whitelist_path = "data/moad_whitelist.pkl"
    with open(whitelist_path, 'rb') as f:
        whitelist_data = pickle.load(f)
        # Extract the actual list of valid complexes
        if isinstance(whitelist_data, dict) and 'valid_complexes' in whitelist_data:
            whitelist_set = set(whitelist_data['valid_complexes'])
        else:
            whitelist_set = set(whitelist_data)  # Fallback for direct list/set
    print(f"Loaded whitelist with {len(whitelist_set)} valid complexes")
    print(f"Sample whitelist entries: {list(whitelist_set)[:5]}")
    
    # Set up args
    if not hasattr(args, 'matching'): args.matching = True
    if not hasattr(args, 'all_atoms'): args.all_atoms = False
    if not hasattr(args, 'sampling_alpha'): args.sampling_alpha = 1
    if not hasattr(args, 'sampling_beta'): args.sampling_beta = 1
    if not hasattr(args, 'no_torsion'): args.no_torsion = False
    if not hasattr(args, 'crop_beyond'): args.crop_beyond = 10
    
    # Create t_to_sigma function
    t_to_sigma = partial(t_to_sigma_compl, args=args)
    
    # Debug version of preprocessing_ligands that shows what's being filtered
    original_preprocessing_ligands = MOAD.preprocessing_ligands
    
    def debug_preprocessing_ligands(self):
        print(f"\n=== DEBUG preprocessing_ligands CALLED ===")
        print(f"Self type: {type(self)}")
        print(f"Self: {self}")
        
        # Call original first
        original_preprocessing_ligands(self)
        
        print(f"After original preprocessing:")
        if hasattr(self, 'cluster_to_ligands'):
            print(f"  Total clusters: {len(self.cluster_to_ligands)}")
            total_ligands = sum(len(ligands) for ligands in self.cluster_to_ligands.values())
            print(f"  Total ligands: {total_ligands}")
            
            # Show sample clusters and ligands
            sample_clusters = list(self.cluster_to_ligands.keys())[:3]
            for cluster in sample_clusters:
                ligands = self.cluster_to_ligands[cluster]
                print(f"  Cluster {cluster}: {len(ligands)} ligands")
                print(f"    Sample ligands: {ligands[:3]}")
                # Check if these ligands are in whitelist
                in_whitelist = [lig for lig in ligands[:3] if lig in whitelist_set]
                print(f"    In whitelist: {in_whitelist}")
        
        # Apply whitelist filtering
        if hasattr(self, 'cluster_to_ligands') and len(whitelist_set) > 0:
            print(f"\nApplying whitelist filtering...")
            filtered_cluster_to_ligands = {}
            total_filtered_ligands = 0
            
            for cluster_name, ligands in self.cluster_to_ligands.items():
                valid_ligands = [lig for lig in ligands if lig in whitelist_set]
                if valid_ligands:
                    filtered_cluster_to_ligands[cluster_name] = valid_ligands
                    total_filtered_ligands += len(valid_ligands)
            
            print(f"After whitelist filtering:")
            print(f"  Clusters: {len(self.cluster_to_ligands)} -> {len(filtered_cluster_to_ligands)}")
            print(f"  Ligands: {sum(len(ligands) for ligands in self.cluster_to_ligands.values())} -> {total_filtered_ligands}")
            
            self.cluster_to_ligands = filtered_cluster_to_ligands
        else:
            print(f"No whitelist filtering applied (no cluster_to_ligands or empty whitelist)")
    
    # Apply the debug patch
    print(f"Patching MOAD.preprocessing_ligands...")
    print(f"Original method: {original_preprocessing_ligands}")
    print(f"New method: {debug_preprocessing_ligands}")
    MOAD.preprocessing_ligands = debug_preprocessing_ligands
    print(f"After patching: {MOAD.preprocessing_ligands}")
    
    try:
        print(f"\nCreating train loader...")
        train_loader, val_loader, val_dataset2 = construct_loader(args, t_to_sigma, device)
        print(f"Train loader batches: {len(train_loader)}")
        print(f"Val loader batches: {len(val_loader)}")
        
    finally:
        # Restore original method
        MOAD.preprocessing_ligands = original_preprocessing_ligands

if __name__ == "__main__":
    debug_whitelist()
#!/usr/bin/env python3
"""
Quick test to see train vs val data distribution with targeted whitelist
"""

import torch
import pickle
from functools import partial
from datasets.moad import MOAD
from datasets.loader import construct_loader
from utils.parsing import parse_train_args
from utils.diffusion_utils import t_to_sigma as t_to_sigma_compl

def test_data_splits():
    args = parse_train_args()
    args.dataset = "moad"
    args.moad_dir = "data/BindingMOAD_2020_ab_processed_biounit/"
    args.limit_complexes = 100  # Small test
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load targeted whitelist 
    with open('data/moad_targeted_whitelist.pkl', 'rb') as f:
        whitelist_data = pickle.load(f)
        train_complexes = set(whitelist_data['train_complexes'])
        val_complexes = set(whitelist_data['val_complexes'])
        all_complexes = set(whitelist_data['valid_complexes'])
    
    print(f"Whitelist stats:")
    print(f"  Total complexes: {len(all_complexes)}")
    print(f"  Train complexes: {len(train_complexes)}")
    print(f"  Val complexes: {len(val_complexes)}")
    
    # Set up args
    if not hasattr(args, 'matching'): args.matching = True
    if not hasattr(args, 'all_atoms'): args.all_atoms = False
    if not hasattr(args, 'sampling_alpha'): args.sampling_alpha = 1
    if not hasattr(args, 'sampling_beta'): args.sampling_beta = 1
    if not hasattr(args, 'no_torsion'): args.no_torsion = False
    if not hasattr(args, 'crop_beyond'): args.crop_beyond = 10
    
    # Create t_to_sigma function
    t_to_sigma = partial(t_to_sigma_compl, args=args)
    
    # Patch MOAD to use targeted whitelist
    original_preprocessing_ligands = MOAD.preprocessing_ligands
    
    def targeted_preprocessing_ligands(self):
        original_preprocessing_ligands(self)
        
        if hasattr(self, 'cluster_to_ligands') and len(all_complexes) > 0:
            print(f"\nFiltering for {self.split} split...")
            if self.split == 'train':
                target_complexes = train_complexes
            elif self.split == 'val':
                target_complexes = val_complexes
            else:
                target_complexes = all_complexes
            
            original_clusters = len(self.cluster_to_ligands)
            original_ligands = sum(len(ligands) for ligands in self.cluster_to_ligands.values())
            
            filtered_cluster_to_ligands = {}
            for cluster_name, ligands in self.cluster_to_ligands.items():
                valid_ligands = [lig for lig in ligands if lig in target_complexes]
                if valid_ligands:
                    filtered_cluster_to_ligands[cluster_name] = valid_ligands
            
            filtered_clusters = len(filtered_cluster_to_ligands)
            filtered_ligands = sum(len(ligands) for ligands in filtered_cluster_to_ligands.values())
            
            print(f"  Before: {original_clusters} clusters, {original_ligands} ligands")
            print(f"  After: {filtered_clusters} clusters, {filtered_ligands} ligands")
            
            self.cluster_to_ligands = filtered_cluster_to_ligands
    
    MOAD.preprocessing_ligands = targeted_preprocessing_ligands
    
    try:
        print(f"\nCreating loaders...")
        train_loader, val_loader, val_dataset2 = construct_loader(args, t_to_sigma, device)
        
        print(f"\nResults:")
        print(f"  Train batches: {len(train_loader)}")
        print(f"  Val batches: {len(val_loader)}")
        
        if len(train_loader) > 0:
            print(f"  Train batch size: {len(next(iter(train_loader)))}")
        if len(val_loader) > 0:
            print(f"  Val batch size: {len(next(iter(val_loader)))}")
        else:
            print(f"  No validation data available")
        
    finally:
        MOAD.preprocessing_ligands = original_preprocessing_ligands

if __name__ == "__main__":
    test_data_splits()
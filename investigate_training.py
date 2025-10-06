#!/usr/bin/env python3
"""
Investigate what's actually happening in our training - check data sizes and batch contents
"""

import torch
import pickle
from functools import partial
from datasets.moad import MOAD
from datasets.loader import construct_loader
from utils.parsing import parse_train_args
from utils.diffusion_utils import t_to_sigma as t_to_sigma_compl

def investigate_training_data():
    args = parse_train_args()
    args.dataset = "moad"
    args.moad_dir = "data/BindingMOAD_2020_ab_processed_biounit/"
    args.limit_complexes = 50
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load targeted whitelist 
    with open('data/moad_targeted_whitelist.pkl', 'rb') as f:
        whitelist_data = pickle.load(f)
        whitelist_set = set(whitelist_data['valid_complexes'])
    
    print(f"Loaded whitelist with {len(whitelist_set)} valid complexes")
    
    # Set up args
    if not hasattr(args, 'matching'): args.matching = True
    if not hasattr(args, 'all_atoms'): args.all_atoms = False
    if not hasattr(args, 'sampling_alpha'): args.sampling_alpha = 1
    if not hasattr(args, 'sampling_beta'): args.sampling_beta = 1
    if not hasattr(args, 'no_torsion'): args.no_torsion = False
    if not hasattr(args, 'crop_beyond'): args.crop_beyond = 10
    
    # Create t_to_sigma function
    t_to_sigma = partial(t_to_sigma_compl, args=args)
    
    # Patch MOAD to debug what's happening
    original_preprocessing_ligands = MOAD.preprocessing_ligands
    
    def debug_preprocessing_ligands(self):
        print(f"\n=== DEBUG preprocessing_ligands for {self.split} ===")
        
        # Call original
        original_preprocessing_ligands(self)
        
        if hasattr(self, 'cluster_to_ligands'):
            original_clusters = len(self.cluster_to_ligands)
            original_ligands = sum(len(ligands) for ligands in self.cluster_to_ligands.values())
            print(f"Original: {original_clusters} clusters, {original_ligands} ligands")
            
            # Apply whitelist filtering
            filtered_cluster_to_ligands = {}
            for cluster_name, ligands in self.cluster_to_ligands.items():
                valid_ligands = [lig for lig in ligands if lig in whitelist_set]
                if valid_ligands:
                    filtered_cluster_to_ligands[cluster_name] = valid_ligands
            
            filtered_clusters = len(filtered_cluster_to_ligands)
            filtered_ligands = sum(len(ligands) for ligands in filtered_cluster_to_ligands.values())
            
            print(f"After whitelist: {filtered_clusters} clusters, {filtered_ligands} ligands")
            self.cluster_to_ligands = filtered_cluster_to_ligands
    
    MOAD.preprocessing_ligands = debug_preprocessing_ligands
    
    try:
        print(f"\nCreating loaders...")
        train_loader, val_loader, val_dataset2 = construct_loader(args, t_to_sigma, device)
        
        print(f"\n=== LOADER RESULTS ===")
        print(f"Train loader batches: {len(train_loader)}")
        print(f"Val loader batches: {len(val_loader)}")
        
        # Check actual batch content
        if len(train_loader) > 0:
            print(f"\n=== INSPECTING FIRST BATCH ===")
            first_batch = next(iter(train_loader))
            print(f"Batch type: {type(first_batch)}")
            
            if isinstance(first_batch, (list, tuple)):
                print(f"Batch length: {len(first_batch)}")
                for i, item in enumerate(first_batch):
                    if hasattr(item, 'shape'):
                        print(f"  Item {i}: {type(item)} shape {item.shape}")
                    elif hasattr(item, '__len__'):
                        print(f"  Item {i}: {type(item)} len {len(item)}")
                    else:
                        print(f"  Item {i}: {type(item)}")
            elif hasattr(first_batch, 'batch'):
                print(f"PyG batch - num graphs: {first_batch.batch.max().item() + 1}")
                print(f"Total nodes: {first_batch.x.shape[0] if hasattr(first_batch, 'x') else 'N/A'}")
            
            # Try to see how many samples per batch
            print(f"\n=== DATASET SIZES ===")
            if hasattr(train_loader, 'dataset'):
                print(f"Train dataset length: {len(train_loader.dataset)}")
            if hasattr(val_loader, 'dataset'):
                print(f"Val dataset length: {len(val_loader.dataset)}")
        
        # Check batch size settings
        print(f"\n=== BATCH SIZE INFO ===")
        print(f"Train loader batch size: {getattr(train_loader, 'batch_size', 'Unknown')}")
        if hasattr(args, 'batch_size'):
            print(f"Args batch size: {args.batch_size}")
        
    finally:
        MOAD.preprocessing_ligands = original_preprocessing_ligands

if __name__ == "__main__":
    investigate_training_data()
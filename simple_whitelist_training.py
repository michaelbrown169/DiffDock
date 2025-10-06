#!/usr/bin/env python3
"""
Simple whitelist training test without class replacement
"""

import os
import torch
from functools import partial
from datasets.moad import MOAD
from datasets.loader import construct_loader
from utils.parsing import parse_train_args
from utils.utils import get_model, get_optimizer_and_scheduler
from utils.diffusion_utils import t_to_sigma as t_to_sigma_compl
from utils.training import train_epoch, test_epoch, loss_function

def main():
    args = parse_train_args()
    args.dataset = "moad"
    args.moad_dir = "data/BindingMOAD_2020_ab_processed_biounit/"
    
    # Enable unroll_clusters to get individual complex training instead of cluster-based
    args.unroll_clusters = True
    
    # Increase multiplicity to get more training samples per epoch
    args.train_multiplicity = 5
    args.val_multiplicity = 1
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load targeted whitelist (only train/val splits)
    whitelist_path = "data/moad_targeted_whitelist.pkl"
    import pickle
    with open(whitelist_path, 'rb') as f:
        whitelist_data = pickle.load(f)
        # Extract the actual list of valid complexes
        if isinstance(whitelist_data, dict) and 'valid_complexes' in whitelist_data:
            whitelist_set = set(whitelist_data['valid_complexes'])
        else:
            whitelist_set = set(whitelist_data)  # Fallback for direct list/set
    print(f"Loaded whitelist with {len(whitelist_set)} valid complexes")
    
    # Set up basic args
    if not hasattr(args, 'matching'): args.matching = True
    if not hasattr(args, 'all_atoms'): args.all_atoms = False
    if not hasattr(args, 'sampling_alpha'): args.sampling_alpha = 1
    if not hasattr(args, 'sampling_beta'): args.sampling_beta = 1
    if not hasattr(args, 'no_torsion'): args.no_torsion = False
    if not hasattr(args, 'crop_beyond'): args.crop_beyond = 10
    if not hasattr(args, 'tr_weight'): args.tr_weight = 1.0
    if not hasattr(args, 'rot_weight'): args.rot_weight = 1.0  
    if not hasattr(args, 'tor_weight'): args.tor_weight = 1.0
    if not hasattr(args, 'backbone_loss_weight'): args.backbone_loss_weight = 1.0
    if not hasattr(args, 'sidechain_loss_weight'): args.sidechain_loss_weight = 1.0
    
    # Create t_to_sigma function
    t_to_sigma = partial(t_to_sigma_compl, args=args)
    
    # Patch MOAD preprocessing_ligands method to use whitelist
    original_preprocessing_ligands = MOAD.preprocessing_ligands
    
    def whitelist_preprocessing_ligands(self):
        # Call original first
        original_preprocessing_ligands(self)
        
        # Apply whitelist filtering
        if hasattr(self, 'cluster_to_ligands') and len(whitelist_set) > 0:
            filtered_cluster_to_ligands = {}
            for cluster_name, ligands in self.cluster_to_ligands.items():
                valid_ligands = [lig for lig in ligands if lig in whitelist_set]
                if valid_ligands:
                    filtered_cluster_to_ligands[cluster_name] = valid_ligands
            
            print(f"🔄 Whitelist filtering: {len(self.cluster_to_ligands)} -> {len(filtered_cluster_to_ligands)} clusters")
            self.cluster_to_ligands = filtered_cluster_to_ligands
    
    # Apply the patch
    MOAD.preprocessing_ligands = whitelist_preprocessing_ligands
    
    try:
        train_loader, val_loader, val_dataset2 = construct_loader(args, t_to_sigma, device)
        print(f"✅ Train loader batches: {len(train_loader)}")
        print(f"✅ Val loader batches: {len(val_loader)}")
        
        if len(train_loader) == 0:
            print("ERROR: No training samples found!")
            return
        
        # Create model
        model = get_model(args, device, t_to_sigma=t_to_sigma)
        print(f"✅ Model created with {sum(p.numel() for p in model.parameters())} parameters")
        
        # Optimizer and loss function
        optimizer, scheduler = get_optimizer_and_scheduler(args, model)
        loss_fn = partial(loss_function, tr_weight=args.tr_weight, rot_weight=args.rot_weight,
                         tor_weight=args.tor_weight, no_torsion=args.no_torsion)
        
        print(f"Starting training for {args.n_epochs} epochs...")
        
        # Training loop
        for epoch in range(args.n_epochs):
            print(f"\n=== Epoch {epoch+1}/{args.n_epochs} ===")
            
            # Training
            train_losses = train_epoch(model, train_loader, optimizer, device, t_to_sigma, loss_fn, None)
            
            # Validation
            val_losses = test_epoch(model, val_loader, device, t_to_sigma, loss_fn, None) if len(val_loader) > 0 else {}
            
            # Print results
            train_loss = sum(train_losses.values()) if train_losses else 0
            val_loss = sum(val_losses.values()) if val_losses else 0
            
            print(f"✅ Epoch {epoch+1} completed!")
            print(f"   Train loss: {train_loss:.6f}")
            if val_losses:
                print(f"   Val loss: {val_loss:.6f}")
                
        print("\n🎉 Whitelist training completed successfully!")
        
    finally:
        # Restore original method
        MOAD.preprocessing_ligands = original_preprocessing_ligands

if __name__ == "__main__":
    main()
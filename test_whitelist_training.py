#!/usr/bin/env python3

"""
Test training with whitelist integration
Uses the pre-computed whitelist to only process valid complexes
"""

import os
import argparse
import yaml
import torch
from functools import partial
from datasets.moad import MOAD
from datasets.loader import construct_loader  
from torch_geometric.loader import DataLoader
from utils.training import train_epoch, test_epoch
from utils.parsing import parse_train_args
from utils.utils import get_optimizer_and_scheduler, get_model
from utils.diffusion_utils import t_to_sigma as t_to_sigma_compl
import wandb
from tqdm import tqdm

def load_whitelist(whitelist_path="moad_valid_complexes.txt"):
    """Load the pre-computed whitelist of valid complexes"""
    if not os.path.exists(whitelist_path):
        print(f"Warning: Whitelist not found at {whitelist_path}")
        return set()
    
    with open(whitelist_path, 'r') as f:
        whitelist = set(line.strip() for line in f if line.strip())
    
    print(f"Loaded whitelist with {len(whitelist)} valid complexes")
    return whitelist

class WhitelistMOAD(MOAD):
    """MOAD dataset that only processes complexes from the whitelist"""
    
    def __init__(self, whitelist_path=None, *args, **kwargs):
        self.whitelist = load_whitelist(whitelist_path) if whitelist_path else set()
        super().__init__(*args, **kwargs)
    
    def preprocessing_ligands(self):
        # Use the whitelist loaded in __init__
        whitelist_set = self.whitelist
        if whitelist_set:
            print(f"✅ Using whitelist with {len(whitelist_set)} valid complexes")
        else:
            print(f"⚠️  No whitelist available, processing all complexes")
        
        # Filter self.cluster_to_ligands to only include complexes in whitelist
        if whitelist_set:
            print(f"🔄 Filtering clusters and complexes using whitelist...")
            
            # First, filter split_clusters to only include clusters that exist in cluster_to_ligands
            existing_clusters = [c for c in self.split_clusters if c in self.cluster_to_ligands]
            if len(existing_clusters) != len(self.split_clusters):
                print(f"⚠️  {len(self.split_clusters) - len(existing_clusters)} clusters from split not found in cluster_to_ligands")
                self.split_clusters = existing_clusters
            
            # Get all complex names from existing clusters in split
            all_complex_names = set()
            for cluster in self.split_clusters:
                all_complex_names.update(self.cluster_to_ligands[cluster])
            
            # Filter to only keep whitelisted complexes
            whitelisted_complex_names = all_complex_names & whitelist_set
            print(f"📊 Complex overlap: {len(whitelisted_complex_names)}/{len(all_complex_names)} complexes are in whitelist")
            
            # Rebuild cluster_to_ligands with only whitelisted complexes
            original_cluster_to_ligands = self.cluster_to_ligands.copy()
            self.cluster_to_ligands = {}
            
            for cluster, complexes in original_cluster_to_ligands.items():
                # Only keep complexes that are in the whitelist
                filtered_complexes = [c for c in complexes if c in whitelist_set]
                if filtered_complexes:  # Only keep clusters that still have valid complexes
                    self.cluster_to_ligands[cluster] = filtered_complexes
            
            # Update split_clusters to only include clusters that still have complexes
            self.split_clusters = [c for c in self.split_clusters if c in self.cluster_to_ligands]
            
            print(f"🎯 After whitelist filtering:")
            print(f"   - Clusters in split: {len(self.split_clusters)}")
            print(f"   - Total valid complexes: {sum(len(complexes) for complexes in self.cluster_to_ligands.values())}")
        
        # Call the original preprocessing_ligands
        super().preprocessing_ligands()

def main():
    args = parse_train_args()
    
    # Force MOAD dataset
    args.dataset = "moad"
    args.moad_dir = "data/BindingMOAD_2020_ab_processed_biounit/"
    
    if args.config:
        config_dict = yaml.load(args.config, Loader=yaml.FullLoader)
        arg_dict = args.__dict__
        for key, value in config_dict.items():
            if isinstance(value, list):
                for v in value:
                    arg_dict[key].append(v)
            else:
                arg_dict[key] = value
        args.config = args.config.name
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load whitelist
    whitelist_path = "data/moad_whitelist.pkl"
    import pickle
    with open(whitelist_path, 'rb') as f:
        whitelist_set = pickle.load(f)
    print(f"Loaded whitelist with {len(whitelist_set)} valid complexes")
    
    # Create datasets with whitelist
    # Set defaults for missing training args and override paths
    if not hasattr(args, 'matching'): args.matching = True
    if not hasattr(args, 'multiplicity'): args.multiplicity = 1
    if not hasattr(args, 'chain_cutoff'): args.chain_cutoff = None
    if not hasattr(args, 'all_atoms'): args.all_atoms = False
    if not hasattr(args, 'knn_only_graph'): args.knn_only_graph = False
    if not hasattr(args, 'receptor_radius'): args.receptor_radius = 30
    if not hasattr(args, 'c_alpha_max_neighbors'): args.c_alpha_max_neighbors = 10
    if not hasattr(args, 'atom_radius'): args.atom_radius = 5
    if not hasattr(args, 'atom_max_neighbors'): args.atom_max_neighbors = 8
    if not hasattr(args, 'moad_esm_embeddings_path'): args.moad_esm_embeddings_path = 'data/moad_esm2_embeddings.pt'
    if not hasattr(args, 'moad_esm_embeddings_sequences_path'): args.moad_esm_embeddings_sequences_path = 'data/moad_sequences.txt'
    
    # Override moad_dir to point to our actual data location
    args.moad_dir = 'data/BindingMOAD_2020_ab_processed_biounit'
    
    # Initialize datasets with whitelist
    # Use construct_loader like train.py does, but with our whitelist dataset
    # Temporarily modify the MOAD class to use whitelist
    original_moad = MOAD
    
    def create_whitelist_moad_class(whitelist_set):
        class WhitelistMOADForLoader(MOAD):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                # Apply whitelist filtering after initialization
                if hasattr(self, 'cluster_to_ligands') and len(whitelist_set) > 0:
                    filtered_cluster_to_ligands = {}
                    for cluster_name, ligands in self.cluster_to_ligands.items():
                        valid_ligands = [lig for lig in ligands if lig in whitelist_set]
                        if valid_ligands:
                            filtered_cluster_to_ligands[cluster_name] = valid_ligands
                    
                    print(f"🔄 Whitelist filtering: {len(self.cluster_to_ligands)} -> {len(filtered_cluster_to_ligands)} clusters")
                    self.cluster_to_ligands = filtered_cluster_to_ligands
        return WhitelistMOADForLoader
    
    WhitelistMOADForLoader = create_whitelist_moad_class(whitelist_set)
    
    # Create t_to_sigma function
    t_to_sigma = partial(t_to_sigma_compl, args=args)
    
    # Temporarily replace MOAD with our whitelist version
    import datasets.moad
    datasets.moad.MOAD = WhitelistMOADForLoader
    import datasets.loader
    datasets.loader.MOAD = WhitelistMOADForLoader
    
    try:
        train_loader, val_loader, val_dataset2 = construct_loader(args, t_to_sigma, device)
        print(f"Train loader batches: {len(train_loader)}")
        print(f"Val loader batches: {len(val_loader)}")
    finally:
        # Restore original MOAD class  
        datasets.moad.MOAD = original_moad
        datasets.loader.MOAD = original_moad
    
    if len(train_loader) == 0:
        print("ERROR: No training samples found! Check whitelist and dataset paths.")
        return
    
    # Create model
    model = get_model(args, device, t_to_sigma=t_to_sigma)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Optimizer and scheduler
    optimizer, scheduler = get_optimizer_and_scheduler(args, model)
    
    print(f"Starting training for {args.n_epochs} epochs...")
    print(f"Run name: {args.run_name}")
    
    best_val_loss = float('inf')
    
    for epoch in range(args.n_epochs):
        # Training
        train_losses = train_epoch(model, train_loader, optimizer, device, t_to_sigma, None, None)
        
        # Validation  
        val_losses = test_epoch(model, val_loader, device, t_to_sigma, None, None) if len(val_loader) > 0 else {}
        
        # Logging
        train_loss_str = f"  tr {train_losses.get('tr_loss', 0):.4f}"
        train_loss_str += f"   rot {train_losses.get('rot_loss', 0):.4f}" 
        train_loss_str += f"   tor {train_losses.get('tor_loss', 0):.4f}"
        train_loss_str += f"   sc {train_losses.get('sc_loss', 0):.4f}"
        
        val_loss_str = f"  tr {val_losses.get('tr_loss', 0):.4f}"
        val_loss_str += f"   rot {val_losses.get('rot_loss', 0):.4f}"
        val_loss_str += f"   tor {val_losses.get('tor_loss', 0):.4f}" 
        val_loss_str += f"   sc {val_losses.get('sc_loss', 0):.4f}"
        
        total_train_loss = sum(train_losses.values())
        total_val_loss = sum(val_losses.values()) if val_losses else 0
        
        lr = optimizer.param_groups[0]['lr'] if optimizer else getattr(args, 'lr', 0.001)
        
        print(f"Epoch {epoch}: Training loss {total_train_loss:.4f}{train_loss_str}  lr {lr:.4f}")
        if val_losses:
            print(f"Epoch {epoch}: Validation loss {total_val_loss:.4f}{val_loss_str}")
            
            if total_val_loss < best_val_loss:
                best_val_loss = total_val_loss
                print(f"Best Validation Loss {best_val_loss:.4f} on Epoch {epoch}")
        
        if scheduler:
            scheduler.step()

if __name__ == '__main__':
    main()
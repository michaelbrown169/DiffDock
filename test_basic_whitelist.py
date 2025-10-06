#!/usr/bin/env python3
"""
Simple test to verify whitelist model creation and basic functionality
"""

import os
import torch
from functools import partial
from datasets.moad import MOAD
from utils.parsing import parse_train_args
from utils.utils import get_model
from utils.diffusion_utils import t_to_sigma as t_to_sigma_compl

def test_whitelist_basic():
    # Parse arguments
    args = parse_train_args()
    args.moad_dir = "data/BindingMOAD_2020_ab_processed_biounit/"
    args.limit_complexes = 50
    args.n_epochs = 1
    args.batch_size = 4
    args.run_name = "whitelist_basic_test"
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load whitelist
    whitelist_path = "data/moad_whitelist.pkl"
    import pickle
    with open(whitelist_path, 'rb') as f:
        whitelist = pickle.load(f)
    print(f"Loaded whitelist with {len(whitelist)} valid complexes")
    
    # Create model
    t_to_sigma = partial(t_to_sigma_compl, args=args)
    model = get_model(args, device, t_to_sigma=t_to_sigma)
    print(f"✅ Model created successfully with {sum(p.numel() for p in model.parameters())} parameters")
    
    print("✅ Basic test completed successfully!")

if __name__ == "__main__":
    test_whitelist_basic()
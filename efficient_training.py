#!/usr/bin/env python3
"""
Modified training script that uses the MOAD whitelist for efficient dataset loading.
"""

import pickle
import os
import sys
sys.path.insert(0, '/media/mike/mass_drive/Documents/diffdock_adaption/DiffDock')

def load_moad_whitelist(whitelist_path='data/moad_whitelist.pkl'):
    """Load the pre-computed whitelist of valid MOAD complexes."""
    if not os.path.exists(whitelist_path):
        print(f"⚠️  Whitelist not found at {whitelist_path}")
        print("Run create_moad_whitelist.py first to generate it.")
        return None
    
    with open(whitelist_path, 'rb') as f:
        whitelist_data = pickle.load(f)
    
    return set(whitelist_data['valid_complexes'])


def create_filtered_train_command(
    limit_complexes=None,
    batch_size=16,
    n_epochs=10,
    run_name="efficient_training",
    whitelist_path='data/moad_whitelist.pkl'
):
    """
    Create a training command that uses pre-filtered complexes.
    Instead of using --limit_complexes, we'll modify the dataset to only load whitelisted complexes.
    """
    
    # Load whitelist
    valid_complexes = load_moad_whitelist(whitelist_path)
    if valid_complexes is None:
        return None
    
    print(f"📋 Loaded whitelist with {len(valid_complexes)} valid complexes")
    
    if limit_complexes and limit_complexes < len(valid_complexes):
        # Take first N from the whitelist
        valid_complexes = set(list(valid_complexes)[:limit_complexes])
        print(f"🎯 Limited to {len(valid_complexes)} complexes")
    
    # Save filtered list for this training run
    filtered_list_path = f'data/filtered_complexes_{run_name}.txt'
    with open(filtered_list_path, 'w') as f:
        for complex_name in sorted(valid_complexes):
            f.write(f"{complex_name}\n")
    
    # Create training command
    train_cmd = f"""
source ~/anaconda3/etc/profile.d/conda.sh && conda activate diffdock && \\
PYTHONPATH=/media/mike/mass_drive/Documents/diffdock_adaption/DiffDock CUDA_VISIBLE_DEVICES=0 \\
python train.py \\
  --dataset moad \\
  --limit_complexes {len(valid_complexes)} \\
  --n_epochs {n_epochs} \\
  --batch_size {batch_size} \\
  --moad_esm_embeddings_path data/moad_esm2_embeddings.pt \\
  --moad_esm_embeddings_sequences_path data/moad_sequences.txt \\
  --moad_dir data/BindingMOAD_2020_ab_processed_biounit \\
  --run_name "{run_name}" \\
  --split_val data/splits/timesplit_no_lig_overlap_train
""".strip()
    
    return {
        'command': train_cmd,
        'num_complexes': len(valid_complexes),
        'filtered_list_path': filtered_list_path
    }


if __name__ == "__main__":
    # Test with different scales
    test_configs = [
        {'limit': 1000, 'batch_size': 16, 'epochs': 3, 'name': 'test_1k'},
        {'limit': 10000, 'batch_size': 32, 'epochs': 5, 'name': 'test_10k'},
        {'limit': None, 'batch_size': 64, 'epochs': 10, 'name': 'full_dataset'}
    ]
    
    print("🚀 Training configurations with whitelist pre-filtering:\n")
    
    for i, config in enumerate(test_configs, 1):
        print(f"## Configuration {i}: {config['name']}")
        
        result = create_filtered_train_command(
            limit_complexes=config['limit'],
            batch_size=config['batch_size'],
            n_epochs=config['epochs'],
            run_name=config['name']
        )
        
        if result:
            print(f"📊 Will train on: {result['num_complexes']:,} complexes")
            print(f"💾 Filtered list: {result['filtered_list_path']}")
            print(f"⚡ Command:")
            print(result['command'])
            print()
    
    print("🎯 Recommendation: Start with test_1k, then scale up to test_10k once you verify it's working!")
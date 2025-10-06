#!/usr/bin/env python3
"""
Simple pre-filter for MOAD dataset to check file availability.
"""

import os
from pathlib import Path
import pickle


def simple_prefilter_moad(moad_dir, esm_embeddings_path, esm_sequences_path, max_complexes=1000):
    """Simple check of file existence without heavy dependencies."""
    
    print("🔍 Pre-filtering MOAD complexes (simple version)...")
    
    # Get all ligand files
    ligand_dir = os.path.join(moad_dir, 'pdb_superligand')
    protein_dir = os.path.join(moad_dir, 'pdb_protein')
    
    ligand_files = list(Path(ligand_dir).glob('*.pdb'))
    
    if max_complexes:
        ligand_files = ligand_files[:max_complexes]
    
    print(f"Found {len(ligand_files)} ligand files to check")
    
    valid_complexes = []
    invalid_reasons = {}
    
    # Load ESM sequences for checking
    esm_sequences = set()
    try:
        if os.path.exists(esm_sequences_path):
            with open(esm_sequences_path, 'r') as f:
                content = f.read()
                # Extract protein IDs from sequences file
                for line in content.split('\n'):
                    if line.startswith('>'):
                        protein_id = line[1:7]  # Extract first 6 chars after >
                        esm_sequences.add(protein_id)
        print(f"Found {len(esm_sequences)} ESM sequences")
    except Exception as e:
        print(f"Could not load ESM sequences: {e}")
    
    # Check ESM embeddings availability
    esm_embeddings_available = os.path.exists(esm_embeddings_path)
    print(f"ESM embeddings file exists: {esm_embeddings_available}")
    
    for i, ligand_file in enumerate(ligand_files):
        if i % 100 == 0:
            print(f"Checked {i}/{len(ligand_files)} complexes...")
            
        complex_name = ligand_file.stem
        protein_id = complex_name[:6]  # First 6 characters are PDB ID
        
        protein_path = os.path.join(protein_dir, f"{protein_id}.pdb")
        
        # Check all requirements
        checks = {
            'ligand_file_exists': ligand_file.exists(),
            'protein_file_exists': os.path.exists(protein_path),
            'esm_sequence_available': protein_id in esm_sequences,
            'esm_embeddings_file_exists': esm_embeddings_available
        }
        
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
    
    output_path = 'data/moad_simple_prefilter.pkl'
    with open(output_path, 'wb') as f:
        pickle.dump(results, f)
    
    print(f"\n💾 Results saved to {output_path}")
    print(f"🎯 Success rate: {results['stats']['success_rate']:.1%}")
    
    return results


if __name__ == "__main__":
    # Example usage
    results = simple_prefilter_moad(
        moad_dir='data/BindingMOAD_2020_ab_processed_biounit',
        esm_embeddings_path='data/moad_esm2_embeddings.pt',
        esm_sequences_path='data/moad_sequences.txt',
        max_complexes=5000  # Check first 5000 complexes
    )
    
    print(f"\nFound {len(results['valid_complexes'])} valid complexes")
    
    # Save list of valid complex names for easy use
    valid_names = [c['complex_name'] for c in results['valid_complexes']]
    with open('data/moad_valid_complex_names.txt', 'w') as f:
        for name in valid_names:
            f.write(f"{name}\n")
    
    print(f"Saved {len(valid_names)} valid complex names to data/moad_valid_complex_names.txt")
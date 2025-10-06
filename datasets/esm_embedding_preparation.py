import os
import pickle
from argparse import ArgumentParser
from Bio.PDB import PDBParser
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from tqdm import tqdm
from Bio import SeqIO

from datasets.constants import three_to_one

parser = ArgumentParser()
parser.add_argument('--out_file', type=str, default="data/prepared_for_esm.fasta")
parser.add_argument('--dataset', type=str, default="pdbbind")
parser.add_argument('--data_dir', type=str, default='/media/mike/mass_drive/Documents/diffdock_adaption/DiffDock/data/BindingMOAD_2020_ab_processed_biounit/pdb_protein/', help='')
parser.add_argument('--fix_embeddings', type=str, default=None, help='Path to embeddings file to reindex for numeric keys')
parser.add_argument('--fix_embeddings_out', type=str, default=None, help='Output path for reindexed embeddings')
args = parser.parse_args()

biopython_parser = PDBParser()


def get_structure_from_file(file_path):
    """Extract protein sequences from PDB file with robust error handling."""
    try:
        # Use QUIET parser to suppress warnings for malformed PDB files
        quiet_parser = PDBParser(QUIET=True)
        structure = quiet_parser.get_structure('protein', file_path)
        structure = structure[0]
        
        l = []
        for i, chain in enumerate(structure):
            seq = ''
            for res_idx, residue in enumerate(chain):
                if residue.get_resname() == 'HOH':
                    continue
                    
                # Check for required backbone atoms (CA, N, C)
                c_alpha, n, c = None, None, None
                for atom in residue:
                    if atom.name == 'CA':
                        c_alpha = list(atom.get_vector())
                    if atom.name == 'N':
                        n = list(atom.get_vector())
                    if atom.name == 'C':
                        c = list(atom.get_vector())
                
                # Only include residues with complete backbone
                if c_alpha != None and n != None and c != None:
                    try:
                        seq += three_to_one[residue.get_resname()]
                    except Exception as e:
                        seq += '-'
                        # Suppress excessive unknown AA warnings for cleaner output
                        if res_idx < 5:  # Only show first few warnings per protein
                            print(f"Unknown AA: {residue.get_resname()} in {os.path.basename(file_path)} -> replacing with '-'")
            l.append(seq)
        return l
    except Exception as e:
        print(f"Error processing {os.path.basename(file_path)}: {e}")
        return []

data_dir = args.data_dir
names = os.listdir(data_dir)

if args.dataset == 'pdbbind':
    sequences = []
    ids = []

    for name in tqdm(names):
        if name == '.DS_Store': continue
        if os.path.exists(os.path.join(data_dir, name, f'{name}_protein_processed.pdb')):
            rec_path = os.path.join(data_dir, name, f'{name}_protein_processed.pdb')
        else:
            rec_path = os.path.join(data_dir, name, f'{name}_protein.pdb')
        l = get_structure_from_file(rec_path)
        for i, seq in enumerate(l):
            sequences.append(seq)
            ids.append(f'{name}_chain_{i}')
    records = []
    for (index, seq) in zip(ids, sequences):
        record = SeqRecord(Seq(seq), str(index))
        record.description = ''
        records.append(record)
    SeqIO.write(records, args.out_file, "fasta")

elif args.dataset == 'moad':
    print("🧬 Processing BindingMOAD dataset for ESM embeddings...")
    
    # Filter out macOS resource files and get clean protein names
    names = [n[:6] for n in names if n != '.DS_Store' and not n.startswith('._')]
    
    # Use substantial subset for proper training (balance speed vs data size)
    names = names[:5000]   # <-- 5000 samples for good training data
    
    print(f"📊 Found {len(names)} protein files to process")
    
    records = []
    sequences_only = []  # For plain text output
    protein_ids = []     # Track protein names for embedding mapping
    skipped_count = 0
    
    for i, name in enumerate(tqdm(names, desc="Processing proteins")):
        rec_path = os.path.join(data_dir, f'{name}_protein.pdb')
        if not os.path.exists(rec_path):
            skipped_count += 1
            continue

        chains = get_structure_from_file(rec_path)
        if len(chains) == 0 or len(chains[0]) == 0:
            skipped_count += 1
            continue

        # Take first chain only (consistent with original DiffDock logic)
        seq = chains[0]
        
        # CRITICAL: Use numeric ID that matches array index
        # This is what BindingMOAD dataset loader expects!
        numeric_id = str(len(records))  # This will be 0, 1, 2, 3, ...
        record = SeqRecord(Seq(seq), id=numeric_id, description=f"moad_{name}")
        
        records.append(record)
        sequences_only.append(seq)
        protein_ids.append(name)

    print(f"✅ Successfully processed {len(records)} proteins")
    print(f"⚠️  Skipped {skipped_count} proteins (missing/invalid files)")

    # Write outputs in both required formats
    print(f"💾 Writing output files...")
    
    # 1. FASTA file (for ESM embedding generation)
    SeqIO.write(records, args.out_file, "fasta")
    print(f"   📄 FASTA: {args.out_file} (for ESM processing)")
    
    # 2. Plain text file (for BindingMOAD dataset loading)
    txt_out_file = args.out_file.replace('.fasta', '.txt')
    with open(txt_out_file, 'w') as f:
        for seq in sequences_only:
            f.write(seq + '\n')
    print(f"   📄 TXT: {txt_out_file} (for dataset loading)")
    
    print(f"\n🎉 Ready for ESM embedding generation!")
    print(f"Next step: Generate embeddings from {args.out_file}")
    print(f"The embeddings will automatically have the correct numeric keys (0, 1, 2, ...)")
    
    # Store protein mapping for debugging if needed
    mapping_file = args.out_file.replace('.fasta', '_mapping.txt')
    with open(mapping_file, 'w') as f:
        f.write("# Mapping: numeric_id -> protein_name\n")
        for i, protein_id in enumerate(protein_ids):
            f.write(f"{i}\t{protein_id}\n")
    print(f"   📄 Mapping: {mapping_file} (for debugging)")
    
    # Set protein_ids for later embedding reindexing
    current_protein_ids = protein_ids

# Handle embedding reindexing if requested (for existing embeddings with protein name keys)
if args.fix_embeddings is not None:
    import torch
    print(f"\n🔧 Reindexing existing embeddings: {args.fix_embeddings}")
    
    # Load original embeddings (assumed to have protein name keys)
    original_emb = torch.load(args.fix_embeddings)
    print(f"   📥 Loaded {len(original_emb)} original embeddings")
    
    # Use the protein mapping we just created
    if args.dataset == 'moad' and 'current_protein_ids' in locals():
        protein_names = current_protein_ids
    else:
        # Fallback: extract from FASTA descriptions
        protein_names = []
        with open(args.out_file) as fasta_file:
            for record in SeqIO.parse(fasta_file, "fasta"):
                # Extract protein name from description "moad_1c39_1" -> "1c39_1"
                protein_name = record.description.replace('moad_', '') if record.description.startswith('moad_') else record.id
                protein_names.append(protein_name)
    
    # Create new embeddings with numeric indices
    new_emb = {}
    missing_keys = []
    
    for i, protein_name in enumerate(protein_names):
        if protein_name in original_emb:
            new_emb[str(i)] = original_emb[protein_name]
        else:
            missing_keys.append(protein_name)
    
    print(f"   ✅ Successfully mapped {len(new_emb)} embeddings")
    if missing_keys:
        print(f"   ⚠️  Missing {len(missing_keys)} embeddings: {missing_keys[:5]}...")
    
    # Save reindexed embeddings
    out_path = args.fix_embeddings_out or args.fix_embeddings.replace('.pt', '_indexed.pt')
    torch.save(new_emb, out_path)
    print(f"   💾 Saved indexed embeddings: {out_path}")
    
    print(f"\n🎉 Complete! No post-processing needed.")
    print(f"Use: --moad_esm_embeddings_path {out_path}")
    print(f"Use: --moad_esm_embeddings_sequences_path {txt_out_file if args.dataset == 'moad' else 'sequences.txt'}")

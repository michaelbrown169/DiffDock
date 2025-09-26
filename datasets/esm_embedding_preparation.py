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
args = parser.parse_args()

biopython_parser = PDBParser()


def get_structure_from_file(file_path):
    structure = biopython_parser.get_structure('random_id', file_path)
    structure = structure[0]
    l = []
    for i, chain in enumerate(structure):
        seq = ''
        for res_idx, residue in enumerate(chain):
            if residue.get_resname() == 'HOH':
                continue
            residue_coords = []
            c_alpha, n, c = None, None, None
            for atom in residue:
                if atom.name == 'CA':
                    c_alpha = list(atom.get_vector())
                if atom.name == 'N':
                    n = list(atom.get_vector())
                if atom.name == 'C':
                    c = list(atom.get_vector())
            if c_alpha != None and n != None and c != None:  # only append residue if it is an amino acid
                try:
                    seq += three_to_one[residue.get_resname()]
                except Exception as e:
                    seq += '-'
                    print("encountered unknown AA: ", residue.get_resname(), ' in the complex ', file_path, '. Replacing it with a dash - .')
        l.append(seq)
    return l

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
    names = [n[:6] for n in names if n != '.DS_Store']

    # Limit to first 3000 entries for testing
    names = names[:3000]   # <-- REMOVE OR COMMENT OUT THIS LINE TO USE FULL DATASET

    records = []
    for name in tqdm(names):
        rec_path = os.path.join(data_dir, f'{name}_protein.pdb')
        if not os.path.exists(rec_path):
            print(f"Skipping {name}, no {name}_protein.pdb")
            continue

        chains = get_structure_from_file(rec_path)
        if len(chains) == 0:
            continue

        # Only take the first chain
        seq = chains[0]
        record = SeqRecord(Seq(seq), id=name, description="")
        records.append(record)

    SeqIO.write(records, args.out_file, "fasta")

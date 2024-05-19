##############################################################################
# 2022-9-30
# Add functions
##############################################################################

""" Cheminformatics utilities for VAE to enable comparison 
    of seed input and generated SMILES and general purposes """

import networkx as nx
import numpy as np
import random
import selfies as sf

from qed import qed

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors, BRICS, MCS, rdMolDescriptors, Crippen
from rdkit.Chem import QED as rd_qed
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Avalon import pyAvalonTools

from sascorer import calculateScore

from typing import List, Dict, Tuple, Set, Union, NewType

#import mol2vec
#from gensim.models import word2vec
#from mol2vec.features import mol2alt_sentence

import scipy
from scipy.spatial import distance

SMILES = NewType('SMILES', str)
SELFIES = NewType('SELFIES', str)

def check_SMILES(s: SMILES) -> bool:
    
    if Chem.MolFromSmiles(s) is not None:
    # just returning Chem.MolFromSmiles(s) is not None as bool is problematic
        return True
    return False

def percent_valid(smiles_list: List[SMILES],
                  from_set: bool = True) -> float:
    
    if from_set:
        smiles_set = set(smiles_list)
        return sum([int(check_SMILES(s)) for s in smiles_set])/len(smiles_set)
    return sum([int(check_SMILES(s)) for s in smiles_list])/len(smiles_list)
    

def balanced_parantheses(smiles: SMILES,
                         brac: Dict[str, int] = {'(': -1,
                                                ')': 1,
                                                '[': -1,
                                                ']': 1}) -> bool:
    
    return not bool(sum([brac[c] for c in smiles if c in brac.keys()]))

def num_branches(smiles: SMILES) -> int:
    
    if not balanced_parantheses(smiles):
        return None
    return smiles.count('(')

def unclosed_rings(smiles: SMILES) -> bool:
    
    return not all([smiles.count(d) % 2 == 0 for d in filter(lambda s: s.isdigit(), smiles)])

def get_mol(smiles: SMILES) -> Chem.Mol:
    
    mol = Chem.MolFromSmiles(smiles)
    return mol
    
def get_smiles(mol: Chem.Mol) -> SMILES:
    
    smiles = Chem.MolToSmiles(mol)
    return smiles
    
def get_rdk_fingerprint(smiles: SMILES) -> DataStructs.ExplicitBitVect:
    
    mol = Chem.MolFromSmiles(smiles)
    fp = Chem.RDKFingerprint(mol)
    return fp
    
def get_fingerprint_similarity_pair(fp1: DataStructs.ExplicitBitVect,
                                    fp2: DataStructs.ExplicitBitVect) -> float:
    
    sim = DataStructs.FingerprintSimilarity(fp1,fp2)
    return sim
    
def get_morgan_fingerprint(smiles: SMILES) -> DataStructs.ExplicitBitVect:
    
    mol = Chem.MolFromSmiles(smiles)
    fp = AllChem.GetMorganFingerprint(mol,2)
    return fp

def get_dice_similarity(fp1: DataStructs.ExplicitBitVect,
                        fp2: DataStructs.ExplicitBitVect) -> float:
    
    sim = DataStructs.DiceSimilarity(fp1, fp2)
    return sim
    
def get_num_heavy_atom_count(smiles: SMILES) -> int:
    
    mol = Chem.MolFromSmiles(smiles)
    return Chem.Lipinski.HeavyAtomCount(mol)
    
def get_molar_mass(smiles: SMILES) -> float:
    
    mol = Chem.MolFromSmiles(smiles)
    molar_mass = Descriptors.MolWt(mol)
    return molar_mass

def remove_stereo_carbs(smiles: SMILES, 
                        stereo_carbs: List[str] = ["[C@H]",
                                                   "[C@@H]",
                                                   "[C@]",
                                                   "[C@@]"]) -> SMILES:
    
    for c in stereo_carbs:
        smiles = smiles.replace(c, 'C')
    return smiles

def remove_single_direction(smiles: SMILES,
                            dirs: List[str] = ["/",
                                              "\\"]) -> SMILES:
    
    for d in dirs:
        smiles = smiles.replace(d, '')
    return smiles

def smiles_from_rec(m: Dict[str, SMILES]) -> SMILES:
    
    return m['molecule_structures']['canonical_smiles']

def randomized_same(randomized_smiles_list: List[SMILES]) -> bool:
    
    return len(set([Chem.MolToSmiles(Chem.MolFromSmiles(smiles)) for smiles in randomized_smiles_list])) == 1

def canonical_from_randomized(randomized_smiles_list: List[SMILES]) -> List[SMILES]:
    
    return list(set([Chem.MolToSmiles(Chem.MolFromSmiles(smiles)) for smiles in randomized_smiles_list]))

def canonical_randomized_dict(randomized_smiles_list: List[SMILES]) -> Dict[SMILES, List[SMILES]]:
    d = {canonical: [] for canonical in canonical_from_randomized(randomized_smiles_list)}
    for canonical in d:
        for randomized_smiles in randomized_smiles_list:
            if canonical == Chem.MolToSmiles(Chem.MolFromSmiles(randomized_smiles)):
                d[canonical].append(randomized_smiles)
    return d

def canonical_randomized_freq(randomized_smiles_list: List[SMILES]) -> Dict[SMILES, int]:
    d = canonical_randomized_dict(randomized_smiles_list)
    freq_dict = {k: len(d[k]) for k in d.keys()}
    return freq_dict

def verify_canonical_rand_freq(canonical_smiles_list: List[SMILES],
                               freq_each: int,
                               randomized_smiles_list: List[SMILES]) -> bool:
    
    freq_dict = canonical_randomized_freq(randomized_smiles_list)
    
    return set(freq_dict.keys()) == set(canonical_smiles_list) and all([freq_each == freq_dict[k] for k in freq_dict.keys()])

def rd_canonicalize(smiles: SMILES) -> SMILES:
    
    mol = Chem.MolFromSmiles(smiles)
    smi = Chem.MolToSmiles(mol)
    
    return smi

def rd_canonicalize_smiles_batch(smiles_list: List[SMILES]) -> List[SMILES]:
    
    return [rd_canonicalize(s) for s in smiles_list]

def is_smiles_batch_canonical(smiles_list: List[SMILES]) -> bool:
    
    return all([s == rd_canonicalize(s) for s in smiles_list])

def tanimoto_smiles_sim(smiles1: SMILES, smiles2: SMILES) -> float:
    
    mol1 = Chem.MolFromSmiles(smiles1)
    mol2 = Chem.MolFromSmiles(smiles2)

    fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, 2, nBits=2048)
    fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, 2, nBits=2048)
    
    return DataStructs.TanimotoSimilarity(fp1, fp2)

def ecfp_smiles_as_array(smiles):
    
    mol = Chem.MolFromSmiles(smiles)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
    arr = np.zeros((0,), dtype = np.int8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr

def BRICS_frags(smiles: SMILES) -> Set[str]:
    
    return BRICS.BRICSDecompose(Chem.MolFromSmiles(smiles))

def BRICS_set_smiles_batch(smiles_list: List[SMILES]) -> Set[str]:
    
    frag_set = set()
    for s in smiles_list:
        frag_set.update(BRICS_frags(s))
        
    return frag_set

def all_frags_smiles_batch(smiles_list: List[SMILES]):
    all_frags = []
    for s in smiles_list:
        frags_list = list(BRICS_frags(s))
        all_frags.extend(frags_list)
        
    return all_frags

def new_BRICS_frags_in_generated(trainset: List[SMILES],
                                 generated: List[SMILES]) -> Set[str]:
    
    trainset_frags = BRICS_set_smiles_batch(trainset)
    generated_frags = BRICS_set_smiles_batch(generated)
    
    return generated_frags.difference(trainset_frags)

def common_frags_batch_pair(smiles_list1: List[SMILES],
                            smiles_list2: List[SMILES]) -> Set[str]:
    
    frag_set1 = BRICS_set_smiles_batch(smiles_list1)
    frag_set2 = BRICS_set_smiles_batch(smiles_list2)
    
    return frag_set1.intersection(frag_set2)

def common_BRICS_frag_frac(ref_smiles: SMILES,
                           compare_smiles: SMILES) -> float:
    
    ref_frags = BRICS_frags(ref_smiles)
    compare_frags = BRICS_frags(compare_smiles)
    
    return len(ref_frags.intersection(compare_frags))/len(ref_frags)

def unique_frags_frac(smiles_list: List[SMILES]):
    # can be used as a metric for internal diversity of a SMILES batch
    frags_set = BRICS_set_smiles_batch(smiles_list)
    frags_list = all_frags_smiles_batch(smiles_list)
    return len(frags_set)/len(frags_list)


def get_scaffold(smiles: SMILES) -> SMILES:
    
    return MurckoScaffold.MurckoScaffoldSmilesFromSmiles(smiles)

def scaffold_set_smiles_batch(smiles_list: List[SMILES]) -> Set[str]:
    
    scaffold_set = set()
    for s in smiles_list:
        scaffold_set.add(get_scaffold(s))
        
    return scaffold_set

def new_scaffolds_in_generated(trainset: List[SMILES],
                               generated: List[SMILES]) -> Set[SMILES]:
    
    trainset_scaffolds = scaffold_set_smiles_batch(trainset)
    generated_scaffolds = scaffold_set_smiles_batch(generated)
    
    return generated_scaffolds.difference(trainset_scaffolds)

def smiles_pair_share_scaffold(smiles1: SMILES,
                               smiles2: SMILES) -> bool:
    
    return get_scaffold(smiles1) == get_scaffold(smiles2)

def agg_tanimoto_sim(ref_smiles_list: List[SMILES],
                     smiles: SMILES,
                     agg_type: str) -> float:
    
    ref_mols = [Chem.MolFromSmiles(s) for s in ref_smiles_list]
    ref_fp_list = [AllChem.GetMorganFingerprintAsBitVect(ref_mol, 2, nBits=2048) for ref_mol in ref_mols]
    
    smiles_fp = AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smiles), 2, nBits=2048)
    
    agg_sim = 0
    for fp in ref_fp_list:
        sim = DataStructs.TanimotoSimilarity(smiles_fp, fp)
        if agg_type == "max":
            if sim > agg_sim:
                agg_sim = sim
        
        if agg_type == "min":
            if sim < agg_sim:
                agg_sim = sim

        if agg_type == "mean":
            agg_sim += sim

    if agg_type == "mean":
        agg_sim = agg_sim/len(ref_fp_list)

    return agg_sim

#model_mol2vec = word2vec.Word2Vec.load('model_300dim.pkl')
def smiles_mol2vec(smiles: SMILES) -> float:
    
    sentence = mol2alt_sentence(Chem.MolFromSmiles(smiles), radius = 1)
    vec_node = 0
    for i in range(len(sentence)):
        vec = model_mol2vec.wv[sentence[i]]
        vec_node += vec

    return vec_node/len(sentence)

def vec_pair_sim(vec1: np.ndarray,
                 vec2: np.ndarray) -> float:
    
    sim = 1/(1 + distance.euclidean(vec1, vec2))
    
    return sim

def mol2vec_sim(smiles1: SMILES,
                smiles2: SMILES) -> float:
    
    mol2vec1 = smiles_mol2vec(smiles1)
    mol2vec2 = smiles_mol2vec(smiles2)
    
    return vec_pair_sim(mol2vec1, mol2vec2)

def mean_mol2vec_batch(smiles_list: List[SMILES]) -> float:
    
    mol2vecs = [smiles_mol2vec(s) for s in smiles_list]
    
    return np.mean(mol2vecs, axis = 0)

def list_by_sim(smiles_list: List[SMILES], 
                ref_smiles: SMILES, 
                sim_type: str, 
                descending: bool = True, 
                sim_funcs: dict = {'tanimoto': tanimoto_smiles_sim,
                                   'mol2vec': mol2vec_sim}) -> Dict[SMILES, float]:
    
    sim_func = sim_funcs[sim_type]
    scores_dict = {s: sim_func(ref_smiles, s) for s in smiles_list}
    scores_dict = dict(sorted(scores_dict.items(), key=lambda scores_dict: scores_dict[1], reverse = descending))
    return scores_dict

def list_by_property(smiles_list: List[SMILES],
                     property_func,
                     descending: bool = True,
                     return_dict = True):
    
    property_dict = {s: property_func(s) for s in smiles_list}
    property_dict = dict(sorted(property_dict.items(), key=lambda property_dict: property_dict[1], reverse = descending))
    
    if return_dict:
        return property_dict
    return [k for k in property_dict.keys()]

def simtype_ranks_same(smiles_list: List[SMILES],
                       ref_smiles: SMILES,
                       sim_funcs: dict = {'tanimoto': tanimoto_smiles_sim,
                                          'mol2vec': mol2vec_sim}) -> bool:
    ranks = []
    for k in sim_funcs.keys():
        ranks.append(list_by_sim(smiles_list, ref_smiles, k))
    return all([ranks[0] == rank for rank in ranks])

def internal_diversity(smiles_list: List[SMILES],
                       ref_smiles: SMILES,
                       sim_type: str,
                       sim_funcs: dict = {'tanimoto': tanimoto_smiles_sim,
                                          'mol2vec': mol2vec_sim}) -> float:
    
    sim_func = sim_funcs[sim_type]
    scores = [sim_func(ref_smiles, s) for s in smiles_list]
    
    return np.var(scores)

def most_similar(smiles_list: List[SMILES],
                 smiles_compare: SMILES,
                 sim_type: str,
                 sim_funcs: dict = {'tanimoto': tanimoto_smiles_sim,
                                    'mol2vec': mol2vec_sim}) -> SMILES:
    
    sim_func = sim_funcs[sim_type]
    
    return smiles_list[np.argmax([sim_func(smiles_compare, s) for s in smiles_list])]

def max_common_substruct_sim(ref_smiles: SMILES,
                             compare_smiles: SMILES) -> Tuple[float, float]:
    
    ref_mol = Chem.MolFromSmiles(ref_smiles)
    compare_mol = Chem.MolFromSmiles(compare_smiles)
    mcs = MCS.FindMCS([ref_mol, compare_mol])
    atom_mcs_frac = mcs.numAtoms/ len(ref_mol.GetAtoms())
    bond_mcs_frac = mcs.numBonds/ len(ref_mol.GetBonds())
    
    return atom_mcs_frac, bond_mcs_frac

def mcs_atom_sim(ref_smiles: SMILES,
                 compare_smiles: SMILES) -> float:
    
    return max_common_substruct_sim(ref_smiles, compare_smiles)[0]

def mcs_bond_sim(ref_smiles: SMILES,
                 compare_smiles: SMILES) -> float:
    
    return max_common_substruct_sim(ref_smiles, compare_smiles)[0]

def lipinski_ro5(smiles: SMILES) -> Dict[str, int]:
    
    mol = Chem.MolFromSmiles(smiles)
    
    mw = rdMolDescriptors.CalcExactMolWt(mol)
    logp = rdMolDescriptors.CalcCrippenDescriptors(mol)[0]
    hba = rdMolDescriptors.CalcNumHBA(mol)
    hbd = rdMolDescriptors.CalcNumHBD(mol)
    
    violations = {"hbd":0,
                  "hba":0,
                  "mw":0,
                  "logp":0}

    if hbd > 5:
        violations["hbd"] += 1
        
    if hba > 10:
        violations["hba"] += 1

    if mw >= 500:
        violations["mw"] += 1

    if logp > 5:
        violations["logp"] += 1

    return violations

def substrings(smiles: SMILES) -> Set[str]:
    
    return set([smiles[i: j] for i in range(len(smiles)) 
            for j in range(i + 1, len(smiles) + 1)])
    
def substrings_list(smiles: SMILES) -> List[str]:
    
    return [smiles[i: j] for i in range(len(smiles)) 
            for j in range(i + 1, len(smiles) + 1)]

def common_substrings(smiles1: SMILES,
                      smiles2: SMILES) -> Set[str]:
    
    return substrings(smiles1).intersection(substrings(smiles2))

def lcs(smiles1: SMILES,
        smiles2: SMILES) -> str:
    
    return max(common_substrings(smiles1, smiles2), key = len)
    
def nlcs_sim(smiles1: SMILES,
             smiles2: SMILES) -> float:
    
    # Normalized longest common subsequence
    
    # from A comparative study of SMILES-based compound similarity functions
    # for drug-target interaction prediction
    
    return (len(lcs(smiles1, smiles2))**2)/(len(smiles1) * len(smiles2))

def smiles_kernel_sim(smiles1: SMILES,
                      smiles2: SMILES) -> float:
    
    # SMILES representation-based string kernel
    
    # from A comparative study of SMILES-based compound similarity functions
    # for drug-target interaction prediction
    
    common_substrs = common_substrings(smiles1, smiles2)
    common_substrs = set(filter(lambda c: len(c) > 1, common_substrs))
    smiles1_substrs_list = substrings_list(smiles1)
    smiles2_substrs_list = substrings_list(smiles2)
    common_substrs_freq1 = {substr: smiles1_substrs_list.count(substr) for substr in common_substrs}
    common_substrs_freq2 = {substr: smiles2_substrs_list.count(substr) for substr in common_substrs}
    
    return np.dot([common_substrs_freq1[k] for k in common_substrs],
                  [common_substrs_freq2[k] for k in common_substrs])


def edit_distance(s1: str, s2: str) -> int:
    # Levenshtein distance
    
    if len(s1) > len(s2):
        s1, s2 = s2, s1

    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2+1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
        
    return distances[-1]

def edit_sim(smiles1: SMILES,
             smiles2: SMILES) -> float:
    # edit distance
    
    # from A comparative study of SMILES-based compound similarity functions
    # for drug-target interaction prediction
    
    return 1 - (edit_distance(smiles1, smiles2)/max(len(smiles1), len(smiles2)))

def fcfp(smiles: SMILES) -> DataStructs.ExplicitBitVect:
    # functional class fingerprint
    
    mol = Chem.MolFromSmiles(smiles)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol,
                                               radius = 2,
                                               useFeatures=True,
                                               nBits=2048)
    return fp

def tanimoto_fcfp_sim(smiles1: SMILES,
                      smiles2: SMILES) -> float:
    
    fp1 = fcfp(smiles1)
    fp2 = fcfp(smiles2)
    
    return DataStructs.TanimotoSimilarity(fp1, fp2)

def avalon_fp(smiles: SMILES) -> DataStructs.ExplicitBitVect:
    # avalon fingerprint
    
    mol = Chem.MolFromSmiles(smiles)
    fp = pyAvalonTools.GetAvalonFP(mol, nBits = 512)
    return fp

def tanimoto_avalonfp_sim(smiles1: SMILES,
                          smiles2: SMILES) -> float:
    
    fp1 = avalon_fp(smiles1)
    fp2 = avalon_fp(smiles2)
    
    return DataStructs.TanimotoSimilarity(fp1, fp2)





def inchi_from_smiles(smiles: SMILES) -> str:
    
    return Chem.MolToInchi(Chem.MolFromSmiles(smiles))

def convert_selfies_dataset_to_inchi(selfies_dataset: List[SELFIES]) -> List[str]:
    
    return [inchi_from_smiles(sf.decoder(s)) for s in selfies_dataset]

def substructure_matches(smiles: SMILES,
                         smiles_subs: SMILES) -> Tuple[tuple]:
    
    mol = Chem.MolFromSmiles(smiles)
    mol_subs = Chem.MolFromSmarts(smiles_subs)
    matches = mol.GetSubstructMatches(mol_subs)
    
    return matches

def parse_tokens_from_string_list(string_list,
                                  tokenizer):
    
    token_set = set()
    for s in string_list:
        for t in tokenizer.tokenize(s):
            token_set.add(t)
            
    return token_set

def compute_property_smiles_dict(smiles_dict: dict,
                                 property_func):
    property_dict = {r: [] for r in smiles_dict.keys()}
    for k in smiles_dict.keys():
        smiles_list = smiles_dict[k]
        property_vals = [property_func(s) for s in smiles_list]
        property_dict[k].extend(property_vals)
        print(f"Radius: {k} done")
        print('-'*50)
    return property_dict


def remove_wildcard_from_smarts(smarts: SMILES,
                                tokenizer) -> SMILES:
    
    smarts = ''.join(t for t in tokenizer.tokenize(smarts) if '*' not in t)
    
    if smarts.find('(') != -1:
        idx = np.array([i for i in range(len(smarts) - 1) if smarts[i] == '(' and smarts[i + 1] == ')'])
        smarts = ''.join(smarts[i] for i in range(len(smarts)) if i not in np.append(idx, idx + 1))
        
    return smarts

def merge_frag_to_string(base_str: Union[SMILES, SELFIES],
                          to_merge: Union[SMILES, SELFIES],
                          pos: str):
    
    if pos not in ["start", "end"]:
        raise ValueError("pos must be one of start or end")
    
    return base_str + to_merge if pos == 'end' else to_merge + base_st
                

def num_unique_structures(string_list: List[Union[SMILES, SELFIES]]) -> int:
    
    if is_selfies(string_list[0]):
        string_list = [sf.decoder(s) for s in string_list]
    
    inchi_list = [inchi_from_smiles(s) for s in string_list]
    
    return len(set(inchi_list))

def canonicalize_selfies(selfies: SELFIES):
    
    smiles = sf.decoder(selfies)
    canonical_smiles = rd_canonicalize(smiles)
    
    return sf.encoder(canonical_smiles)

def canonicalize_selfies_batch(selfies_list: List[SELFIES]):
    
    return [canonicalize_selfies(s) for s in selfies_list]

def ring_count(smiles):
    counts = 0
    for char in smiles:
        try:
            casted = int(char)
            counts += 1
        except Exception:
            continue
    return counts/2

def atom_count(smiles:str, atom:str):
    return smiles.count(atom.upper()) + smiles.count(atom.lower())



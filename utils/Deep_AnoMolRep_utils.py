import matplotlib.pyplot as plt
import numpy as np
import random
import selfies as sf
import rdkit
import tensorflow as tf
from numpy.random import rand
from numpy.linalg import norm
from numpy import dot,sin,cos,outer,pi
from scipy.spatial import distance
from typing import List, Dict, Tuple, Union, NewType, Callable
from rdkit import Chem

SMILES = NewType('SMILES', str)
SELFIES = NewType('SELFIES', str)

def get_char_idx_mappers(SMILES_list: List[SMILES]) -> Tuple[np.ndarray, Dict[str, int], set]:
                            
    """ Functions mapping SMILES character to token/index identifier
    and back
    
    Args:
        SMILES_list (list): list of SMILES strings in dataset
    
    Returns
        (tuple): index to character, character to index, and vocab
    """
    
    # Get vocabulary of unique characters set
    vocab = set()
    for s in SMILES_list:
        for char in s:
            vocab.add(char)
    # Map index identifier to SMILES character
    index_to_char = np.array(sorted(vocab))
    # Map SMILES character to index identifier
    char_to_index = {y:x for (x, y) in enumerate(index_to_char)}
    
    return index_to_char, char_to_index, vocab

def split_dataset(dataset: List[SMILES], size: int) -> Tuple[List[SMILES], List[SMILES]]:
    
    size_to_split = size
    train_set = dataset[:size]
    val_set = dataset[size:]
    return train_set, val_set

def convert_to_SMILES(tokens: List[int],
                      index_to_char: np.ndarray) -> SMILES:
    
	return ''.join([index_to_char[c] for c in tokens if index_to_char[c] != ' '])

def unpad_smiles(smiles: SMILES) -> SMILES:
    
    """ Unpad SMILES to validate for RDKit functions """
    
    return smiles.replace(' ', '')

def get_smiles_from_logits(logits, code_to_char):
    smiles = ''.join([code_to_char[np.argmax(logits.numpy()[0][i])] for i in range(len(logits[0]))])
    return unpad_smiles(smiles)

def generate_circle(num_dims: int,
                    n_points: int,
                    origin: np.ndarray,
                    radius: float) -> np.ndarray:
    
    """ Args:
            num_dims (int): number of dimensions in vector space (latent dimension
            of VAE)
            n_points (int): number of points in circle (number of vectors to generate
            at radius from origin)
            origin (numpy.ndarray): origin vector (encoded mean)
            radius (float): radius of circle (distance between generated vectors
            and origin)
        
        Returns:
            circle (numpy.ndarray): generated vectors
    """
    
    if len(origin) != num_dims:
        raise ValueError("Origin dimensions must equal num_dimensions")
    V1, V2 = rand(2, num_dims)
    u1 = V1/norm(V1)
    u2 = V2/norm(V2)
    V3 = u1 - dot(u1,u2)*u2
    u3 = V3/norm(V3)
    theta = np.arange(0,2*pi, 2*pi/n_points)
    circle = origin + radius * (outer(cos(theta), u2) + outer(sin(theta), u3))
    return circle

def random_sample_at_radius(origin: np.ndarray, 
                            radius: float,
                            n_points: int = 1000) -> np.ndarray:
    
    """
    Sample point at a radius from the origin by generating points along
    circumference and randomly sampling one
    """
    
    if tf.is_tensor(origin):
        num_dims = origin.shape[1]
        origin = origin.numpy()[0]
    else:
        num_dims = len(origin)
        
    circle = generate_circle(num_dims = num_dims,
                             n_points = n_points,
                             origin = origin,
                             radius = radius)
    
    circle_list = list(circle)
    
    return random.choice(circle_list)

def validity_at_radius(radius: float,
                       num_to_gen: int,
                       latent_centroid: np.ndarray,
                       token_dict: list, 
                       model):
    """
    Checks validity percentage at circumference radius for fixed sized generated SMILES
    """
    count = 0
    valid_mols = []
    for i in range(num_to_gen):
        random_sample = random_sample_at_radius(latent_centroid,radius)
        logits = model.decode(np.array([latent_centroid,random_sample]))
        molecule = get_smiles_from_logits(tf.convert_to_tensor([logits[1].numpy()]),token_dict)
        if Chem.MolFromSmiles(sf.decoder(molecule)) is not None:
            count +=1
            valid_mols.append(molecule)
    validity_percentage = (count/num_to_gen)*100
    return validity_percentage, valid_mols

def validity_null_model(smiles: list):   
    """
    Checks validity for a list of SMILES generated by a null model
    """ 
    converted =[]
    invalid = []
    count = 0
    for i in smiles:
        if Chem.MolFromSmiles(sf.decoder(i)) is not None:
            converted.append(i)
            count +=1
        elif Chem.MolFromSmiles(sf.decoder(i)) is None:
            invalid.append(i)
    val = (count/len(smiles))*100
    return val, converted, invalid

def rd_canonicalize(smiles: SMILES) -> SMILES:
    
    mol = Chem.MolFromSmiles(smiles)
    smi = Chem.MolToSmiles(mol)
    
    return smi

def recon_categorical_accuracy(encoded_selfies,
                               decoded_selfies,
                               tokenizer):
    """
    Calculate categorical accuracy as the percentage of SELFIES tokens
    in the decoded (output) SELFIES sequence that were accurately
    reconstructed on a positional basis wrt the encoded selfies sequence
   
    Formula inferred from SI table showing categorical accuracy
    by Gomez-Bombarelli et al. in Automatic Chemical Design Using
    a Data-Driven Continuous Representation of Molecules
   
    Args:
        encoded_selfies (str): SELFIES sequence encoded into the VAE
                               latent space, which computes its
                               corresponding mean vector and log-var
                               vector pair describing posterior distribution
        decoded_selfies (str): SELFIES sequence generated by decoding the
                               mean vector of the encoded posterior
                               distribution in the latent space
   
    Returns:
        cat_acc (float): Categorical accuracy as the percentage of
                         SELFIES tokens in the decoded SELFIES sequence
                         that were accurately reconstructed on a positional
                         basis wrt the encoded SELFIES sequence
    """
   
    encoded_token_lst = tokenizer.tokenize(encoded_selfies)
    decoded_token_lst = tokenizer.tokenize(decoded_selfies)
   
    encode_len = len(encoded_token_lst)
    decode_len = len(decoded_token_lst)
   
    matches = 0
    for i in range(min(encode_len, decode_len)):
        if encoded_token_lst[i] == decoded_token_lst[i]:
            matches += 1
   
    cat_acc = matches/decode_len
   
    return cat_acc
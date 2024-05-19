##############################################################################
# 2022-9-30
# Add functions
##############################################################################
""" Functions and classes for VAE data analysis and processing """
import matplotlib.pyplot as plt
import numpy as np
import random
import selfies as sf
import tensorflow as tf
import VAE_chem_utils as chem

from umap import UMAP
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from numpy.random import rand
from numpy.linalg import norm
from numpy import dot,sin,cos,outer,pi
from scipy.spatial import distance
from typing import List, Dict, Tuple, Union, NewType

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

def get_val_train_sets(dataset: List[SMILES],
                       split: float) -> Tuple[List[SMILES], List[SMILES]]:
    
    """ Split dataset of SMILES strings into training and validation sets
    
    Args:
        dataset (list): list of SMILES strings in dataset
        split (float): fraction to split into validation set
    
    Returns:
        (tuple): trainset and validation set
    """
    
    dataset_len = len(dataset)
    val_idxs = random.sample(range(dataset_len), round(split*dataset_len))
    val_set = [dataset[i] for i in val_idxs]
    train_set = list(filter(lambda s: s not in val_set, dataset))
    return train_set, val_set


def get_optimizer(optim: str, 
                  lr: float):
    
    """ Get optimizer with set learning rate for optimizer.apply_gradients """
    
    if optim == "Adam":
        return tf.keras.optimizers.Adam(lr)
    elif optim == "RMSProp":
        return tf.keras.optimizers.RMSprop(lr)
    elif optim == "SGD":
        return tf.keras.optimizers.SGD(lr)
    else:
        raise ValueError("Invalid optimizer type")
        

def lr_batch_up(lr: float, 
                batch: int, 
                updtd_batch: int) -> float:
    
    """ Update lr """
    
    return lr * np.sqrt(updtd_batch/batch)

def is_padded(smiles: SMILES) -> bool:
    
    """ Check is SMILES is padded """
    
    return smiles[-1] == ' '

def unpad_smiles(smiles: SMILES) -> SMILES:
    
    """ Unpad SMILES to validate for RDKit functions """
    
    return smiles.replace(' ', '')
    

def unpad_all_smiles(dataset: List[SMILES]) -> List[SMILES]:
    
    """ Unpad all SMILES in dataset """
    
    return [unpad_smiles(s) for s in dataset]

def padded_smiles(dataset: List[SMILES]) -> List[SMILES]:
    
    """ Get list of padded SMILES in dataset """
    
    padded = []
    for s in dataset:
        if s[-1] == ' ':
            padded.append(s)
    return padded
    
    
def all_padded(smiles_list: List[SMILES]) -> bool:
    
    """ Check if all SMILES have been padded with spaces """
    
    return all([s[-1] == ' ' for s in smiles_list])

def pad_list(l,
             length):
    
    """ pad SMILES string with spaces up to length """
    
    return [c + ' '*(length - len(c)) for c in l]


def pad_all(smiles_list: List[SMILES], 
            line_length: int) -> List[SMILES]:
    
    """ pad all SMILES strings in list with spaces up to length """
    
    smiles = list(filter(lambda s: len(s) <= line_length, smiles_list))
    return pad_list(smiles, line_length)

def convert_to_SMILES(tokens: List[int],
                      index_to_char: np.ndarray) -> SMILES:
    
	return ''.join([index_to_char[c] for c in tokens if index_to_char[c] != ' '])

def sort_smiles_list(smiles_list: List[SMILES],
                     order: str = "ascending") -> List[SMILES]:
    
    """ Sort SMILES list """
    
    ascending = list(sorted(smiles_list, key = len))
    if order == "descending":
        return reversed(ascending)
    return ascending

def get_smiles_batches(smiles_list: List[SMILES],
                       batch_size: int,
                       num_batches: int) -> Dict[int, List[SMILES]]:
    
    """ Get dictionary of SMILES batches """
    
    batch_dict = dict()
    for i in range(len(smiles_list)//num_batches - num_batches):
        batch_dict[i] = random.sample(smiles_list[i:i+num_batches], batch_size)
    return batch_dict

def partition_list(lst: list,
                   num_in_part: int,
                   func: str = 'len') -> Tuple[list, list]:
    
    parts = [lst[i:i + num_in_part] for i in range(0, len(lst), num_in_part)]
    try:
        parts_props = [np.mean([eval(func)(p) for p in part]) for part in parts]
    except TypeError:
        parts_props = None
    return parts, parts_props

def get_smiles_at_zero_latent(model,
                              latent_dim: int):
    
    latent_zero_vector = np.array([0.00 for _ in range(latent_dim)])
    latent_zero_vector = tf.convert_to_tensor(np.expand_dims(latent_zero_vector, axis = 0))
    logits_at_zero = model.decode(latent_zero_vector)
    return get_smiles_from_logits(logits_at_zero)

def get_smiles_from_latent(model, latent):
    logits = model.decode(latent)
    return get_smiles_from_logits(logits)

def smiles_from_latent_list(latent_list,
                            char_to_code,
                            model):
    smiles_list = []
    for l in latent_list:
        l = process_smiles_input(l, char_to_code)
        smiles = get_smiles_from_latent(model, l)
        smiles_list.append(smiles)
    return smiles_list

def smiles_from_normal_sample(model, latent_dim):
    sample = tf.random.normal([1, latent_dim])
    return get_smiles_from_latent(model, sample)
# generated smiles are invalid, expect higher validity when sampling from normal distribution if KL loss weight
# (beta) is increased

# make distribution of SMILES length

def latent_space_similarity(latent1: Union[np.ndarray, tf.Tensor],
                            latent2: Union[np.ndarray, tf.Tensor],
                            latent_dim: int) -> float:
    
    if tf.is_tensor(latent1):
        latent1 = latent1.numpy()
        latent2 = latent2.numpy()
    count = 0
    while len(latent1) != latent_dim:
        latent1 = latent1[0]
        latent2 = latent2[0]
        count += 1
        if count > 3:
            break
    if len(latent1) != latent_dim:
        raise ValueError("Length of latents must match latent_dim")
    return 1/(1 + distance.euclidean(latent1, latent2))


def get_logits_from_smiles_code(smiles_code, model):
    """ get logits from SMILES codes (decoder output from mean and latent sample z) """
    mean, logvar, logits_from_z = model(smiles_code) 
    # logits_from_z = model.decode(z) and z = mean + (stddev * ~N(0,1))
    logits_from_mean = model.decode(mean)
    return logits_from_mean, logits_from_z


def get_smiles_from_logits(logits: tf.Tensor,
                           code_to_char: np.ndarray) -> SMILES:
    
    # get SMILES from logits directly instead of sampling
    smiles = ''.join([code_to_char[np.argmax(logits.numpy()[0][i])] for i in range(len(logits[0]))])
    return unpad_smiles(smiles)

def process_smiles_input(smiles_input: Union[SMILES, np.ndarray, tf.Tensor],
                         char_to_code: Dict[str, int]) -> tf.Tensor:
    
    if isinstance(smiles_input, str):
        smiles_codes = np.array([char_to_code[c] for c in smiles_input])
    elif isinstance(smiles_input, np.ndarray):
        smiles_codes = smiles_input
    elif tf.is_tensor(smiles_input):
        smiles_codes = smiles_input.numpy()
    else:
        raise TypeError("smiles_input must be one of string, numpy array, or tensor types")
    
    return tf.convert_to_tensor(np.expand_dims(smiles_codes, axis = 0))

""" generate vectors at a radius from origin """
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

def rand_cos_sim(v: np.ndarray, 
                 costheta: float,
                 same_magnitude: bool = True) -> np.ndarray:
    
    # Create random vector with cosine_similarity costheta with v, with same or different
    # magnitude
    u = v / np.linalg.norm(v)
    # Pick a random vector:
    r = np.random.multivariate_normal(np.zeros_like(v), np.eye(len(v)))
    # Form a vector perpendicular to v:
    uperp = r - r.dot(u)*u
    # Make it a unit vector:
    uperp = uperp / np.linalg.norm(uperp)
    w = costheta*u + np.sqrt(1 - costheta**2)*uperp
    
    if same_magnitude:
        w *= (np.linalg.norm(v)/np.linalg.norm(w))
        
    return w

def cosine_sim(v1: np.ndarray,
               v2: np.ndarray) -> float:
    
    return 1 - distance.cosine(v1, v2)

def random_sample_in_circle(origin, radius):
    
    random_radius = np.random.uniform(0.0, radius, 1)[0]
    
    if tf.is_tensor(origin):
        num_dims = origin.shape[1]
        origin = origin.numpy()[0]
    else:
        num_dims = len(origin)
        
    circle = generate_circle(num_dims = num_dims,
                             n_points = 1000,
                             origin = origin,
                             radius = random_radius)
    
    circle_list = list(circle)
    
    return random.choice(circle_list)

def random_sample_at_radius(origin, radius):
    
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
                             n_points = 1000,
                             origin = origin,
                             radius = radius)
    
    circle_list = list(circle)
    
    return random.choice(circle_list)

def random_sample_in_shell(origin: np.ndarray, 
                           lower_radius: float, 
                           upper_radius: float, 
                           n_points: int = 1000):
    
    random_radius = np.random.uniform(lower_radius, upper_radius, 1)[0]
    
    if tf.is_tensor(origin):
        num_dims = origin.shape[1]
        origin = origin.numpy()[0]
    else:
        num_dims = len(origin)
        
    circle = util.generate_circle(num_dims = num_dims,
                                  n_points = n_points,
                                  origin = origin,
                                  radius = random_radius)
    
    circle_list = list(circle)
    
    return random.choice(circle_list)

def circle_rads_stats(origin: np.ndarray,
                      circle: np.ndarray) -> Dict[str, float]:
    
    rads = [distance.euclidean(origin, circle[i]) for i in range(len(circle))]
    return {'mean': np.mean(rads), 'std': np.std(rads)}


def generate_2d_circles(radius_list: List[float],
                        n_points: int):
    num_dims = 2
    origin = np.array([0,0])
    circles = [generate_circle(num_dims, n_points, origin, radius_list[i]) for i in range(len(radius_list))]
    for circle in circles:
        plt.scatter([circle[i][0] for i in range(len(circle))], [circle[i][1] for i in range(len(circle))])

def generate_2d_circles_with_diff_n_points(radius_list: List[float],
                                           n_points_rad: List[float]):
    num_dims = 2
    origin = np.array([0,0])
    circles = [generate_circle(num_dims, n_points_rad[i], origin, radius_list[i]) for i in range(len(radius_list))]
    for circle in circles:
         plt.scatter([circle[i][0] for i in range(len(circle))], [circle[i][1] for i in range(len(circle))])

        
def plot_loss(epochs_list: List[float],
              total_loss_vals: List[float], 
              val_loss_vals: List[float]):
    
    plt.rcParams['figure.figsize'] = [9, 9]    
    training_plot, = plt.plot(epochs_list, total_loss_vals)
    validation_plot, = plt.plot(epochs_list, val_loss_vals)
    plt.legend([training_plot, validation_plot], ['Training', 'Validation'], fontsize = 16)
    plt.xlabel('Epochs', fontsize = 16)
    plt.ylabel('Total Loss (reconstruction loss + 0.5 *kl_loss)', fontsize = 16)
    plt.title('Total Loss vs. Epoch for VAE Training', fontsize = 16)
    plt.show()

def distance_list(vectors: Union[np.ndarray, list],
                  origin = None) -> List[float]:
    
    if origin is None:
        origin = np.array([0.0 for _ in range(len(vectors[0]))])
    return [distance.euclidean(origin, v) for v in vectors]

def centroid_vectors(vectors: np.ndarray) -> np.ndarray:
    
    return np.mean(vectors, axis = 0)

def distance_consecutive_pairs(vec_list: Union[np.ndarray, List]) -> List[float]:
    
    return [distance.euclidean(vec_list[i], vec_list[i + 1]) for i in range(len(vec_list) - 1)]

def stats_per_dim(latent_vectors: np.ndarray) -> List[Tuple[float, float, float]]:
    
    num_dims = len(latent_vectors[0])
    stats = []
    for i in range(num_dims):
        vals = [l[i] for l in latent_vectors]
        stats.append((np.mean(vals), np.var(vals), np.std(vals)))
    return stats

def idx_at_plateau(int_list: List[int]) -> int:
    for i in range(len(int_list)):
        if len(set(int_list[i:])) == 1:
            return i

def standardize(vals: np.ndarray) -> np.ndarray:
    return (vals - np.mean(vals))/np.std(vals)

def get_2d_cords_from_reduced(transformed: np.ndarray,
                              standardize_vals: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    
    x = np.array([arr[0] for arr in transformed])
    y = np.array([arr[1] for arr in transformed])
    
    if standardize_vals:
        return standardize(x), standardize(y)
    return x, y

def generate_points_for_sampling(radius_vals: list,
                                 num_per_radius: list,
                                 origin: np.ndarray) -> List[np.ndarray]:
    
    points = []
    for i in range(len(radius_vals)):
        circle = generate_circle(num_dims = len(origin),
                                 n_points = num_per_radius[i],
                                 origin = origin,
                                 radius = radius_vals[i])
        points.extend(list(circle))
    return points


find_2nd = lambda string, substring: string.find(substring, string.find(substring) + 1)
num_samples_from_string = lambda string: int(string[string.find(':') + 1: string.find(',')])
unique_count_from_string = lambda string: int(string[find_2nd(string, ':') + 1: string.find('\n')])

def get_num_samples_unique_count_from_txt(data: List[str]) -> Tuple[np.ndarray]:
    
    num_samples_to_plot = [num_samples_from_string(d) for d in data]
    unique_count_to_plot = [unique_count_from_string(d) for d in data]
    return np.array(num_samples_to_plot), np.array(unique_count_to_plot)

def mean_tanimoto_score_along_circumference(latent_origin: np.ndarray,
                                            radius_vals: list,
                                            num_points_each_radius: list,
                                            model):
    num_dims = len(latent_origin)
    radius_vals_len = len(radius_vals)
    if radius_vals_len != len(num_points_each_radius):
        raise ValueError("length of radius_vals must equal num_points_each_radius")
    smiles_at_origin = get_smiles_from_logits(model.decode(tf.convert_to_tensor([latent_origin])))
    mean_tanimoto_score_each_radius = []
    for i in range(radius_vals_len):
        generated_smiles = []
        circle = generate_circle(num_dims = num_dims,
                                 n_points = num_points_each_radius[i],
                                 origin = latent_origin,
                                 radius = radius_vals[i])
        for point in circle:
            selfies = get_smiles_from_logits(model.decode(tf.convert_to_tensor([point])))
            try:
                smiles = sf.decoder(selfies)
                generated_smiles.append(smiles)
            except Exception:
                continue
        mean_tanimoto_radius_val = np.mean([chem.tanimoto_smiles_sim(smiles_at_origin, smile) for smile in generated_smiles])
        mean_tanimoto_score_each_radius.append(mean_tanimoto_radius_val)
        print(f"Radius: {radius_vals[i]} done")
    return radius_vals, mean_tanimoto_score_each_radius

def generate_num_SMILES_in_radius(radius: float,
                                  num_to_generate: int,
                                  latent_origin: np.ndarray,
                                  model,
                                  print_at: int,
                                  code_to_char: np.ndarray):
    
    num_samples = 0
    unique_smiles = []
    unique_inchi = set()
    
    while len(unique_smiles) < num_to_generate:
    
        num_samples += 1
        point = random_sample_in_circle(origin = latent_origin, radius = radius)
        logits = model.decode(tf.convert_to_tensor([point]))
        selfies = get_smiles_from_logits(logits, code_to_char)
        smiles = sf.decoder(selfies)

        can_smiles = chem.rd_canonicalize(smiles)
        inchi = chem.inchi_from_smiles(can_smiles)

        inchi_set_len = len(unique_inchi)
        unique_inchi.add(inchi)

        if len(unique_inchi) > inchi_set_len:
            unique_smiles.append(can_smiles)
        
        unique_smiles_len = len(unique_smiles)
        
        if unique_smiles_len % print_at == 0:
            print(f"Number of Unique Molecules: {unique_smiles_len}")
            print(f"Number of Random Samples: {num_samples}")
    
    return unique_smiles


def locate_latent_space_global_boundary(latent_dims: int,
                                        start_radius: float,
                                        increment: float,
                                        sampling_radius: float,
                                        model,
                                        code_to_char: np.ndarray,
                                        batch_size: int = 100,
                                        latent_origin: Union[np.ndarray, str] = 'zero'):
    
    # Get radius from origin where uniqueness reduces to 1
    # consider the global latent space boundary where repetitive sampling
    # always decodes to one molecule
    
    if latent_origin == 'zero':
        latent_origin = np.array([0.0 for _ in range(latent_dims)])
    
    # Set initial default value of num_uniques_iter at num_samples
    num_uniques_iter = batch_size
    current_radius = start_radius
    iters = 0
    
    while num_uniques_iter != 1:
        
        # Start calculating uniqueness at multiple origins at increasing radius
        # from start_radius, generate origins increasing by increment until
        # num_uniques_iter = 1
        
        iters += 1
        current_radius += increment
        generated_smiles_iter = []
        
        # Calculate uniqueness at origin at current radius
        
        # sample random point at current radius to be sampling origin
        random_origin_current_radius = random_sample_at_radius(origin = latent_origin,
                                                               radius = current_radius)

        batch = np.array([random_sample_in_circle(origin = random_origin_current_radius,
                                                  radius = sampling_radius)
                                                  for _ in range(batch_size)])
        
        logits_batch = model.decode(batch)
        
        for logit in logits_batch:
            selfies = get_smiles_from_logits(logits = tf.convert_to_tensor([logit.numpy()]),
                                             code_to_char = code_to_char)
            smiles = sf.decoder(selfies)
            generated_smiles_iter.append(smiles)
        
        num_uniques_iter = chem.num_unique_structures(generated_smiles_iter)
        
        print(f"Number of Iterations: {iters}")
        print(f"Current Radius: {current_radius}")
        print(f"Number of Uniques: {num_uniques_iter}")
        print(f"Uniqueness: {num_uniques_iter/batch_size}")
        
    return current_radius

def twoD_scatterplot_from_highD_vectors(highD_vec_arrays_list: list,
                                       dim_reduce_func_or_type, 
                                       xlabel_fontsize: tuple,
                                       ylabel_fontsize: tuple,
                                       titlelabel_fontsize: tuple,
                                       labels: list,
                                       plot_using: str,
                                       n_components = 2,
                                       color_marker_list: list = None,
                                       rc_params = (12, 10)):
    
    """
    Args:
        highD_vec_arrays_list
        dim_reduce_func_or_type (function or str): 
        xlabel_fontsize (Tuple[str, str]):
        ylabel_fontsize (Tuple[str, str]):
        titlelabel_fontsize (Tuple[str, str]):
        labels (List[str]):
        plot_using (str):
        n_components (int):
        color_marker_list (List[tuple[str, str]]):
        rc_params (Tuple[int, int]):
    
    
    
    """
    
    """
    if color_marker_list is not None:
        if len(highD_vec_arrays_list) != len(color_marker_list)
            raise ValueError('check lengths for highD_vec_arrays_list, color_marker_list and labels must be the same')
    if len(highD_vec_arrays_list) != len(labels):
        raise ValueError('check lengths for highD_vec_arrays_list, color_marker_list and labels must be the same')
    """
    if isinstance(dim_reduce_func_or_type, str):
        if dim_reduce_func_or_type == 'pca':
            dim_reducer = PCA(n_components = n_components).fit_transform
        elif dim_reduce_func_or_type == 'tsne':
            dim_reducer = TSNE(n_components = n_components).fit_transform
        elif dim_reduce_func_or_type == 'umap':
            dim_reducer = UMAP(n_components = n_components).fit_transform
    elif hasattr(dim_reduce_func_or_type, '__call__'):
        dim_reducer = dim_reduce_func_or_type
    else:
        raise TypeError('dim_reduce_func_or_type must be either string or function')
        
    concatenated = []
    for arr in highD_vec_arrays_list:
        concatenated.extend(arr)
    
    reduced_dims = dim_reducer(np.array(concatenated))
    x_vals, y_vals = [d[0] for d in reduced_dims], [d[1] for d in reduced_dims]
    
    idx_tuples = []
    for i in range(len(highD_vec_arrays_list)):
        if i == 0:
            idx_tuples.append((0, len(highD_vec_arrays_list[i]) + 1))
        else:
            idx_tuples.append((idx_tuples[-1][1], idx_tuples[-1][1] + len(highD_vec_arrays_list[i])))
    
    if plot_using == 'plt':
        if color_marker_list is None:
            plt.rcParams["figure.figsize"] = rc_params
            for i in range(len(idx_tuples)):
                tup = idx_tuples[i]
                plt.scatter(x_vals[tup[0]:tup[1]], y_vals[tup[0]:tup[1]], label = labels[i])
            plt.legend()
            plt.xlabel(xlabel_fontsize[0], fontsize = xlabel_fontsize[1])
            plt.ylabel(ylabel_fontsize[0], fontsize = ylabel_fontsize[1])
            plt.title(titlelabel_fontsize[0], fontsize = titlelabel_fontsize[1])
        else:
            plt.rcParams["figure.figsize"] = rc_params
            for i in range(len(idx_tuples)):
                tup = idx_tuples[i]
                plt.scatter(x_vals[tup[0]:tup[1]], y_vals[tup[0]:tup[1]], label = labels[i], 
                            c = color_marker_list[i][0], marker = color_marker_list[i][1])
            plt.legend()
            plt.xlabel(xlabel_fontsize[0], fontsize = xlabel_fontsize[1])
            plt.ylabel(ylabel_fontsize[0], fontsize = ylabel_fontsize[1])
            plt.title(titlelabel_fontsize[0], fontsize = titlelabel_fontsize[1])
            
    elif plot_using == 'sns':
        if color_marker_list is None:
            pass
            
        else:
            pass

def smiles_from_txt(filename):
    with open(filename, 'r') as f:
        smiles = [s[:-1] for s in f]
    return smiles


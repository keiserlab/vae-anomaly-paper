import numpy as np
import selfies as sf
import VAE_chem_utils as chem
from deepchem.feat.smiles_tokenizer import BasicSmilesTokenizer

class Process_SMILES_Reps(BasicSmilesTokenizer):
    
    def __init__(self, 
                 dataset,
                 rep_type,
                 index_to_token = None,
                 token_to_index = None):
        
        if index_to_token is not None:
            if not isinstance(index_to_token, np.ndarray) or not isinstance(token_to_index, dict):
                raise TypeError("index_to_token must be an array and token_to_index must be a dictionary")
            
            if rep_type == "selfies":
                for token in index_to_token:
                    if not token.startswith('['):
                        raise ValueError("invalid selfies token type")
            

                    
        BasicSmilesTokenizer.__init__(self)
        
        self.encode_to_selfies = sf.encoder
        self.decode_from_selfies = sf.decoder
        
        self.dataset = dataset
        self.index_to_token = index_to_token
        self.token_to_index = token_to_index
        self.vocab = None
        self.codes_len = None
        self.count_dict = None
        self.randomized_dataset = None
        self.num_randomized_per_smiles = None
        self.vocab_update_count = 0
        
        if rep_type == "selfies":
            self.encoder_func = self.encode_to_selfies
            self.decoder_func = self.decode_from_selfies
        elif rep_type == "smiles":
            self.encoder_func = self.decoder_func = lambda x: x
        else:
            raise ValueError("rep_type must be one of selfies or smiles")
        

    def canonicalize_dataset(self):
        self.dataset = chem.rd_canonicalize_smiles_batch(self.dataset)
    
    def apply_rep_type(self):
        self.rep_type_applied = True
        self.dataset = [self.encoder_func(s) for s in self.dataset]
    
    def set_vocab_codeslen(self):
        # set dataset to randomized
        self.vocab = set()
        self.codes_len = 0
        for s in self.dataset:
            tokens = self.tokenize(s)
            tokens_len = len(tokens)
            if tokens_len > self.codes_len:
                self.codes_len = tokens_len
            for t in tokens:
                self.vocab.add(t)
        self.vocab.add(' ')
        self.vocab_update_count += 1
        
    
    def remove_raretoken_smiles(self,
                                min_instances,
                                reset_vocab_codeslen = True):
        
        # Second remove_raretoken_smiles function to test speed
        # set dataset to randomized
        self.count_dict = {token : 0 for token in self.vocab}
        smiles_tokens = [self.tokenize(s) for s in self.dataset]
        for token in self.vocab:
            for tokens_list in smiles_tokens:
                if token in tokens_list:
                    self.count_dict[token] += 1
        
        for k in self.count_dict.keys():
            if self.count_dict[k] <= min_instances:
                self.dataset = list(filter(lambda s: k not in s, self.dataset))
        if reset_vocab_codeslen:
            self.set_vocab_codeslen()
            
            
    def assign_max_codeslen(self, 
                            max_codeslen, 
                            reset_vocab_codeslen = True):
        
        # set dataset to randomized
        self.dataset = list(filter(lambda s: len(self.tokenize(s)) <= max_codeslen, self.dataset))
        if reset_vocab_codeslen:
            self.set_vocab_codeslen()

        
    def set_token_index_maps(self):
        self.index_to_token = np.array(sorted(self.vocab))
        self.token_to_index = {y:x for (x, y) in enumerate(self.index_to_token)}
        
        
    def process_all(self, obj, meths = [".de_stereo_dataset()",
                                        ".set_vocab_codeslen()",
                                        ".set_token_index_maps()"]):
        for m in meths:
            eval(str(obj) + m)
    

    def smilestokens_to_indexes(self, 
                                s, 
                                padding = True):
        
        tokens_indexes = [self.token_to_index[t] for t in self.tokenize(s)]
        if padding:
            return tokens_indexes + [self.token_to_index[' '] for _ in range(self.codes_len - len(tokens_indexes))]
        return tokens_indexes
    
    def convert_to_SMILES(self, 
                          indexes, 
                          with_padding = False, 
                          ind_to_token = None):
        
        if ind_to_token is None:
            ind_to_token = self.index_to_token
        if with_padding:
            return ''.join([ind_to_token[i] for i in indexes])
        return ''.join([ind_to_token[i] for i in indexes if ind_to_token[i] != ' '])
        
    def get_smiles_codes(self):
        return [self.smilestokens_to_indexes(s) for s in self.dataset]
    
    def reset_dataset_to_SMILES(self,
                                canonicalize = False):
        if canonicalize:
            self.dataset = [chem.rd_canonicalize(self.decoder_func(s)) for s in self.dataset]
        else:
            self.dataset = [self.decoder_func(s) for s in self.dataset]
        print("vocab codes len not reset")
    
    def summarize_data(self):
        print(f"Vocab Length: {len(self.vocab)}")
        print(f"Max Codes Length: {self.codes_len}")
        print(f"Token Occurance Frequency: {self.count_dict}")
        


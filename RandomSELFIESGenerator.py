import os
import duckdb
import random
import selfies as sf

from typing import List, Callable
from collections import Counter

class RandomMolecularStringGenerator:
    def __init__(self,
                 base_dataset: List[str],
                 base_canonical_smiles_dataset: List[str],
                 tokenizer: Callable,
                 csv_filename: str):
        
        self.base_dataset = base_dataset
        self.base_canonical_smiles_dataset = base_canonical_smiles_dataset 
        self.tokenizer = tokenizer
        self.dataset_len = len(base_dataset)
        self.tokenprob_dict_per_idx = []
        self.FILE_SELFIES = csv_filename
        
    def set_params(self):
        
        self.token_vocab = set()
        self.tokenMatrix = []
        self.tokenLens = []
        self.FILE_DUCKDB = 'selfies.duckdb'
        self.dbexists = os.path.isfile(self.FILE_DUCKDB)
        self.conn = duckdb.connect(self.FILE_DUCKDB)
        self.conn.execute(f'DROP TABLE IF EXISTS dataset')
        self.conn.execute(f'CREATE TABLE dataset (selfies VARCHAR); COPY dataset FROM {self.FILE_SELFIES};')

        
        
        for chem_string in self.base_dataset:
            chem_string_tokens = self.tokenizer.tokenize(chem_string)
            self.tokenMatrix.append(chem_string_tokens)
            tokens_len = len(chem_string_tokens)
            self.tokenLens.append(tokens_len)
            self.token_vocab.update(chem_string_tokens)
        
        self.token_vocab_lst = sorted(self.token_vocab)
        self.maxTokenLen = max(self.tokenLens)
        self.minTokenLen = min(self.tokenLens)
        
        for string_tokens in self.tokenMatrix:
            tokensLen = len(string_tokens)
            numToPad = self.maxTokenLen - tokensLen
            string_tokens.extend([' ']*numToPad)
        
        for i in range(self.maxTokenLen):
            token_list = [tok_lst[i] for tok_lst in self.tokenMatrix]
            self.tokenprob_dict_per_idx.append(self.normalize_dict(dict(Counter(token_list))))
    
    def normalize_dict(self,
                       my_dict):
        
        self.norm_sum = sum(my_dict.values())
        for i in my_dict:
            my_dict[i] = float(my_dict[i]/self.norm_sum)
        
        return my_dict
                
    def generate_naive_random(self, 
                              num_to_generate: int) -> List[str]:
        
        generated = []
        for _ in range(num_to_generate):
            
            # randomly pick length of SELFIES string to generate (number of tokens)
            # in [minTokenLen, maxTokenLen]
            selfiesTokenLen = random.randint(self.minTokenLen, self.maxTokenLen)
            
            # compile random_selfies by uniform random sampling for each index in generated SELFIES string 
            random_selfies = ''.join([self.token_vocab_lst[random.randint(0, len(self.token_vocab) - 1)] for _ in range(selfiesTokenLen)])
            generated.append(random_selfies)
            
        return generated 
    
    def generate_shuffle_random(self,
                                num_to_generate: int) -> List[str]:
        generated = []
        for row in self.conn.execute(f"SELECT * FROM dataset USING SAMPLE {num_to_generate}").fetchall():
            row_tokenized = self.tokenizer.tokenize(row[0])
            shuffled_row = random.sample(row_tokenized, len(row_tokenized))
            generated.append(''.join(shuffled_row))
        return generated
     
    
    def generate_itdr_random(self, 
                             num_to_generate: int) -> List[str]:
        generated = []
        for _ in range(num_to_generate):
            random_selfies = ''
            # select token based on token probability distributions per index (column) of the
            # token matrix
            for idx in range(0, self.maxTokenLen):
                token = random.choices(list(self.tokenprob_dict_per_idx[idx].keys()),
                                       weights = list(self.tokenprob_dict_per_idx[idx].values()),
                                       k = 1)[0]
                if token != ' ':
                    random_selfies += token
            generated.append(random_selfies)
            
        return generated


import duckdb
import random
import selfies as sf

from typing import List, Callable
from collections import Counter

class RandomMolecularStringGenerator:
    def __init__(self,
                 base_dataset: List[str],
                 base_canonical_smiles_dataset: List[str],
                 tokenizer: Callable):
        
        self.base_dataset = base_dataset
        self.base_canonical_smiles_dataset = base_canonical_smiles_dataset 
        self.tokenizer = tokenizer
        self.dataset_len = len(base_dataset)
        self.tokenprob_dict_per_idx = []
        
    def set_params(self):
        
        self.token_vocab = set()
        self.tokenMatrix = []
        self.tokenLens = []
        self.string_gen = (s for s in self.base_dataset)
        self.conn = duckdb.connect()
        self.conn.execute("DROP TABLE IF EXISTS my_table")
        self.conn.execute("CREATE TABLE my_table (column_selfies VARCHAR)")
        
        for row in self.string_gen:
            self.conn.execute("INSERT INTO my_table VALUES (?)", [row])
            
        self.db_size = self.conn.execute("SELECT COUNT(*) FROM my_table").fetchone()[0]
        
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
        # generate num_to_generate random row indices
        random_indices = random.sample(range(self.db_size), num_to_generate)
        # Read each row one-by-one, shuffle it, and write it to the output list
        for index in random_indices:
            row_string = self.conn.execute(f"SELECT * FROM my_table LIMIT 1 OFFSET {index}").fetchone()
            row_string_tokenized = self.tokenizer.tokenize(row_string[0])
            shuffled_string_tokens = random.sample(row_string_tokenized, len(row_string_tokenized))
            generated.append(''.join(shuffled_string_tokens))
            
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


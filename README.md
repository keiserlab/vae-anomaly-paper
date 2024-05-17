# Fuzz testing molecular representation using deep variational anomaly generation 

### About/Synopsis


We radially survey the outlier regions of the latent space of a Variational Autoencoder (VAE) trained on SELFIES strings with the goal of generating strings (representational anomalies) that impose a fuzz test over the SELFIES string representation and resultantly, test its robustness. We find that a specific radial decoding domain of the VAE latent space outperforms three null fuzz testers (SELFIES generators) at the task of minimizing the percentage of valid strings generated, with validity defined in terms of the SELFIES string's conversion success to a valid SMILES representation. We propose that the VAE and associated anomaly generation approach offer an effective tool for assessing the robustness of molecular representations. 

![image1 (1)](https://github.com/keiserlab/vae-anomaly-paper/assets/85256012/587e5b98-0004-4d95-8ca5-e7c1b4a0512e)

### Highlights


- We generated SELFIES sets of size 10,000 per 994 evenly-spaced latent radii from 6.0 to 1000.0. Then, we calculated the validity percentage within generated sets as the percentage of valid SELFIES strings. We found the VAE radially organized validity percentage of SELFIES sets as a function of their generative radius in the latent space. We found two key radial decoding domains: 1) R < 13.0, which always generated valid strings, and 2) R > 28.0, which consistently outperformed the best null generator (naive random) at the task of minimizing validity percentage in generated SELFIES sets of fixed size (10,000).
- We also generated SELFIES sets using three null methods to establish a performance baseline for validity minimization. These null generators vary based on their informativeness of the training distribution of strings: 1) naive random: we first randomly sample a sequence size, and then populate the sequence by sampling tokens with replacement uniformly at random from the token vocabulary; 2) shuffle random: we first sample a string from the training dataset, and then shuffle it internally by re-arranging tokens in the sequence; 3) index-token distribution random: we conceptualize the dataset as a matrix of tokens, with rows representing SELFIES strings and columns representing lists of constituent tokens. This defines a discrete distribution of tokens per each index (columns) of the padded sequences in the dataset. Then we sample a token uniformly at random per each index and compile the final sequence by concatenating the sampled tokens and ignoring the special padding character.
- Amongst the null models, naive random generates SELFIES sets with 73.88% validity (constant rate), shuffle random with 99.71% validity, and index-token distribution random with 99.91% validity. The VAE latent space, when surveyed at a generative radius of 61.0, decodes SELFIES sets with only 11.24% validity, which outperforms the naive random method at validity minimization by a margin of 62.64%. Furthermore, the generative radial domain of 28.0 < R < 1000.0 consistently outperformed the naive random method at minimizing validity percentage in generated SELFIES sets -- establishing an applicability domain for the VAE's fuzz testing abilities over SELFIES robustness.
- On tracing sources of error in invalid SELFIES, we identified two “troublesome” atom tokens ([Na+1] and [K+1]) that were consistently bonded beyond their valid valences and consequently yielded invalid SMILES. However, the errors were fixable using manual valence correction consistent with the module’s otherwise automatic bond-correcting approach.

### Dependencies

```python 
rdkit
tensorflow
```

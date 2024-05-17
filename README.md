# Fuzz testing molecular representation using deep variational anomaly generation 

### About/Synopsis
We radially survey the outlier regions of the latent space of a Variational Autoencoder (VAE) trained on SELFIES strings with the goal of generating strings (representational anomalies) that impose a fuzz test over the SELFIES string representation and resultantly, test its robustness. We find that a specific radial decoding domain of the VAE latent space outperforms three null fuzz testers (SELFIES generators) at the task of minimizing the percentage of valid strings generated, with validity defined in terms of the SELFIES string's conversion success to a valid SMILES representation. We propose that the VAE and associated anomaly generation approach offer an effective tool for assessing the robustness of molecular representations. 

![image1 (1)](https://github.com/keiserlab/vae-anomaly-paper/assets/85256012/587e5b98-0004-4d95-8ca5-e7c1b4a0512e)

### Highlights
- We generated SELFIES sets of size 10,000 per 994 evenly-spaced radii from 6.0 to 1000.0 and calculated the validity percentage within generated sets as the percentage of valid SELFIES strings. We found the VAE radially organized validity percentage of SELFIES sets as a function of their generative radius. We found two key radial decoding domains: 1) R < 13.0, which always generated valid strings, and 2) R > 28.0, which consistently outperformed the best null generator (naive random) at the task of minimizing validity percentage in generated SELFIES sets of fixed size (10,000).

  

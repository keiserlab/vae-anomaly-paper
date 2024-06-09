##############################################################################
##############################################################################

import tensorflow as tf

from tensorflow.python.keras import layers, Model, Sequential

class MolVariationalAutoencoder(Model):
    
    def __init__(self, input_length,
                       vocab_size,
                       latent_dim,
                       rnn_units):
    
        super(MolVariationalAutoencoder, self).__init__()
    
        self.input_length = input_length
        self.vocab_size = vocab_size
        self.latent_dim = latent_dim
        self.rnn_units = rnn_units
        
        self.encoder = Sequential(
                [
                layers.Embedding(vocab_size, vocab_size, input_length = input_length, mask_zero=False),
                layers.Conv1D(filters = 9, kernel_size = 9, padding = 'SAME', activation = 'tanh'),
                layers.Conv1D(filters = 9, kernel_size = 9, padding = 'SAME', activation = 'tanh'),
                layers.Conv1D(filters = 11, kernel_size = 10, padding = 'SAME', activation = 'tanh'),
                layers.Flatten(),
                layers.Dense(latent_dim + latent_dim)
                ]
                )
        

        self.decoder = Sequential(
                [
                layers.Dense(input_length, input_shape = (latent_dim,)),
                layers.Reshape((input_length, 1), input_shape=(input_length,)),
                layers.GRU(units=rnn_units, return_sequences=True),
                layers.GRU(units=rnn_units, return_sequences=True),
                layers.GRU(units=rnn_units, return_sequences=True),
                layers.Dense(vocab_size)
                ]
                )
        
    def call(self, x):
        mean, logvar = self.encode(x)
        latent = self.reparametrize(mean, logvar)
        logits = self.decoder(latent)
        return mean, logvar, logits
        
    def encode(self, x):
        return tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
        
    def decode(self, x):
        return self.decoder(x)
        
    def reparametrize(self, mean, logvar):
        stddev = tf.exp(0.5 * logvar)
        sample = tf.random.normal(mean.shape)
        return stddev * sample + mean
        

        
def kl_div(mean, logvar):
    kl_div = 0.5 * tf.reduce_sum(tf.square(mean) + tf.exp(logvar) - logvar - 1, axis=1)
    return tf.reduce_mean(kl_div)
    
def recon_cross_entropy_loss(labels, logits):
    cross_entropy_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels = labels, logits = logits)
    cross_entropy_loss = tf.reduce_sum(cross_entropy_loss, axis=1)
    return tf.reduce_mean(cross_entropy_loss)

def total_loss(mean, logvar, labels, logits, 
                   kl_loss_lambda, with_kl_weight = True):
        
    if with_kl_weight:
        loss = kl_loss_lambda * kl_div(mean, logvar) + recon_cross_entropy_loss(labels, logits)
    else:
        loss = kl_div(mean, logvar) + recon_cross_entropy_loss(labels, logits)
    return loss


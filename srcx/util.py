import jax 
import jax.numpy as jnp
import haiku as hk 
from typing import Dict

class FeedforwardModel(hk.Module):
    def __init__(self, d_in, d_out, config, name: str | None = None):
        super(FeedforwardModel,self).__init__(name)
        self.output_sizes = [w for w in config['net_width']] + [d_out]
        self.activation = jax.nn.relu if config['activation']=="relu" else jax.nn.tanh
    
    def __call__(self, inputs):
        return hk.nets.MLP(output_sizes=self.output_sizes,activation=self.activation)(inputs)

def print_elapsedtime(delta):
    hours, rem = divmod(delta, 3600)
    minutes, seconds = divmod(rem, 60)
    print("Elapsed time: {:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))

def gini(array): #https://github.com/oliviaguest/gini
    """Calculate the Gini of a jax.numpy array."""
    # based on bottom eq: http://www.statsdirect.com/help/content/image/stat0206_wmf.gif
    # from: http://www.statsdirect.com/help/default.htm#nonparametric_methods/gini.htm
    array = array.flatten() # all values are treated equally, arrays must be 1d
    if jnp.amin(array) < 0:
        array -= jnp.amin(array) # values cannot be negative
    array += 0.0000001 # values cannot be 0
    array = jnp.sort(array) # values must be sorted
    index = jnp.arange(1, array.shape[0]+1) # index per array element
    n = array.shape[0] # number of array elements
    return (jnp.sum((2 * index - n  - 1) * array)) / (n * jnp.sum(array)) # Gini coefficient
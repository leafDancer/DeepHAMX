import haiku as hk 
import jax
import jax.numpy as jnp 

class ActorCriticKS(hk.Module):
    '''
        actor-critic network of KS
        policy_output.shape = (...,1) value_output.shape = (...,)
    '''
    def __init__(self, hidden_dim=16, name="actor_acritic"):
        super().__init__(name)
        self.hidden_dim = hidden_dim

    def __call__(self, inputs:jnp.ndarray):
        x = inputs.astype(jnp.float32)
        x = jax.nn.tanh(hk.Linear(self.hidden_dim)(inputs))
        # actor
        actor_mean = hk.nets.MLP((self.hidden_dim,self.hidden_dim,1),activation=jax.nn.tanh)(x)
        # critic
        critic = hk.nets.MLP((self.hidden_dim,self.hidden_dim,1),activation=jax.nn.tanh)(x).squeeze(axis=-1)
        return actor_mean, critic
    

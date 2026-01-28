import jax
import jax.numpy as jnp
from flax import linen as nn

class DuelingDQN(nn.Module):
    """
    Dueling DQN Architecture.
    Splits the network into two streams: Value (V) and Advantage (A).
    Q(s, a) = V(s) + (A(s, a) - mean(A(s, a)))
    """
    
    @nn.compact
    def __call__(self, x):
        # 1. INPUT EMBEDDING
        x = jnp.log2(jnp.maximum(x, 1.0)).astype(jnp.int32)
        x = nn.Embed(num_embeddings=32, features=64)(x)        
        
        # 2. CONVOLUTIONAL TRUNK (The "Eyes")
        conv_row = nn.Conv(features=64, kernel_size=(4, 1), padding='VALID')(x)
        conv_col = nn.Conv(features=64, kernel_size=(1, 4), padding='VALID')(x)
        
        # Flatten and combine
        x_row = conv_row.reshape((x.shape[0], -1))
        x_col = conv_col.reshape((x.shape[0], -1))
        x = jnp.concatenate([x_row, x_col], axis=-1)
        
        x = nn.Dense(256)(x)
        x = nn.relu(x)

        # 3. DUELING STREAMS (The "Brain")
        
        # Stream A: Value (How good is the board state?)
        v = nn.Dense(128)(x)
        v = nn.relu(v)
        v = nn.Dense(1)(v)
        
        # Stream B: Advantage (How good is each specific action?)
        a = nn.Dense(128)(x)
        a = nn.relu(a)
        a = nn.Dense(4)(a)
        
        # 4. RECOMBINATION
        q_values = v + (a - jnp.mean(a, axis=-1, keepdims=True))
        
        return q_values
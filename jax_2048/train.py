import os
import jax
import jax.numpy as jnp
import optax
from flax import serialization
from flax.training import train_state
from jax_2048.game import step, add_random_tile
from jax_2048.model import DuelingDQN

# Hyperparameters
BATCH_SIZE = 4096
LEARNING_RATE = 1e-4
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_END = 0.01
DECAY_STEPS = 100_000
SYNC_INTERVAL = 100
CHECKPOINT_FILE = "brain.msgpack"

# Track both current and target networks
class TrainState(train_state.TrainState):
    target_params: any

def create_train_state(rng, learning_rate):
    """Initialize the model and optimizer."""
    model = DuelingDQN()
    dummy_input = jnp.zeros((1, 4, 4), dtype=jnp.int32)
    variables = model.init(rng, dummy_input)
    params = variables['params']
    
    # Clip gradients to prevent explosions
    tx = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(learning_rate)
    )
    
    return TrainState.create(
        apply_fn=model.apply,
        params=params,
        target_params=params,
        tx=tx
    )

# Checkpointing
def save_checkpoint(state, filename=CHECKPOINT_FILE):
    """Saves the model weights to a file."""
    with open(filename, "wb") as f:
        f.write(serialization.to_bytes(state.params))
    print(f"Saved checkpoint to {filename}")

def load_checkpoint(state, filename=CHECKPOINT_FILE):
    """Loads weights from file if it exists."""
    if not os.path.exists(filename):
        print("No checkpoint found. Starting fresh.")
        return state
    
    print(f"Loading checkpoint from {filename}...")
    with open(filename, "rb") as f:
        bytes_data = f.read()
        loaded_params = serialization.from_bytes(state.params, bytes_data)
        
    return state.replace(params=loaded_params, target_params=loaded_params)

@jax.jit
def train_step(state, grids, key, epsilon):
    """One training step: select action, step game, compute loss, update weights."""
    key_action, key_step = jax.random.split(key)

    # Select action with epsilon-greedy strategy
    q_values = state.apply_fn({'params': state.params}, grids)
    best_actions = jnp.argmax(q_values, axis=-1)
    
    random_actions = jax.random.randint(key_action, best_actions.shape, 0, 4)
    should_explore = jax.random.bernoulli(key_action, epsilon, best_actions.shape)
    actions = jax.lax.select(should_explore, random_actions, best_actions)

    # Step the environment
    next_grids, rewards, dones = jax.vmap(step, in_axes=(0, 0, 0))(
        grids, actions, jax.random.split(key_step, grids.shape[0])
    )

    # Reset finished games with fresh boards
    fresh_grids = jax.vmap(add_random_tile)(jnp.zeros_like(next_grids), jax.random.split(key_step, grids.shape[0]))
    fresh_grids = jax.vmap(add_random_tile)(fresh_grids, jax.random.split(key_action, grids.shape[0]))
    next_grids = jnp.where(dones[:, None, None], fresh_grids, next_grids)
    
    # Compute loss using Double DQN
    def loss_fn(params):
        current_q_all = state.apply_fn({'params': params}, grids)
        current_q = jnp.take_along_axis(current_q_all, actions[:, None], axis=1).squeeze()

        target_q_all = state.apply_fn({'params': state.target_params}, next_grids)
        max_next_q = jnp.max(target_q_all, axis=-1)
        scaled_rewards = jnp.log2(rewards + 1.0)
        target = scaled_rewards + GAMMA * max_next_q * (1 - dones)
        
        return jnp.mean((target - current_q) ** 2)

    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    state = state.apply_gradients(grads=grads)

    return state, next_grids, loss, rewards

def train_loop():
    rng = jax.random.PRNGKey(42)
    rng, key_init = jax.random.split(rng)
    
    print("Initializing Model...")
    state = create_train_state(key_init, LEARNING_RATE)
    state = load_checkpoint(state)
    
    # Initialize batch of games
    grids = jnp.zeros((BATCH_SIZE, 4, 4), dtype=jnp.int32)
    grids = jax.vmap(add_random_tile)(grids, jax.random.split(rng, BATCH_SIZE))
    grids = jax.vmap(add_random_tile)(grids, jax.random.split(rng, BATCH_SIZE))

    print(f"Starting training on {BATCH_SIZE} parallel environments...")
    
    avg_reward = 0
    avg_loss = 0
    
    try:
        for i in range(1, 1000001):
            rng, step_key = jax.random.split(rng)
            
            epsilon = max(EPSILON_END, EPSILON_START - (i / DECAY_STEPS))
            
            state, grids, loss, rewards = train_step(state, grids, step_key, epsilon)
            
            if i % SYNC_INTERVAL == 0:
                state = state.replace(target_params=state.params)
                
            avg_loss += loss
            avg_reward += jnp.mean(rewards)
            
            if i % 500 == 0:
                print(f"Step {i} | Loss: {avg_loss/500:.4f} | Avg Reward: {avg_reward/500:.4f} | Epsilon: {epsilon:.2f}")
                avg_loss = 0
                avg_reward = 0
            
            if i % 2000 == 0:
                save_checkpoint(state)

    except KeyboardInterrupt:
        print("\nStopping training...")
    
    save_checkpoint(state)
    print("Goodbye!")

if __name__ == "__main__":
    train_loop()
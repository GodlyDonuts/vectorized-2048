import jax
import jax.numpy as jnp
import time
import os
from jax_2048.game import step, add_random_tile
from jax_2048.model import DuelingDQN
from jax_2048.train import create_train_state, load_checkpoint

def watch_game():
    # 1. Load the Brain
    rng = jax.random.PRNGKey(42)
    state = create_train_state(rng, learning_rate=0.0) # LR doesn't matter for inference
    
    if not os.path.exists("brain.msgpack"):
        print("No trained brain found! Run train.py first.")
        return

    state = load_checkpoint(state, "brain.msgpack")
    print("Brain loaded! Getting ready to play...")

    # 2. Setup one game
    grid = jnp.zeros((1, 4, 4), dtype=jnp.int32) # Batch size 1
    grid = add_random_tile(grid, rng)
    grid = add_random_tile(grid, rng)
    
    # Compile the brain's decision function
    @jax.jit
    def get_action(params, g):
        q_values = state.apply_fn({'params': params}, g)
        return jnp.argmax(q_values, axis=-1)

    # 3. Play Loop
    key = jax.random.PRNGKey(123)
    
    print("\n--- STARTING GAME ---\n")
    
    while True:
        # VISUALIZE
        # Convert JAX array to standard Python list for printing
        board =  grid[0].tolist()
        os.system('cls' if os.name == 'nt' else 'clear') # Clear terminal
        print("-" * 20)
        for row in board:
            # Format nicely: replace 0 with dot, align numbers
            print(" ".join(f"{x if x!=0 else '.':^4}" for x in row))
        print("-" * 20)
        
        # DECIDE
        action = get_action(state.params, grid)[0]
        moves = ["Left", "Up", "Right", "Down"]
        print(f"AI Chose: {moves[action]}")
        
        # ACT
        key, subkey = jax.random.split(key)
        # We need to expand dims for action because 'step' expects a batch
        actions = jnp.array([action])
        grid, reward, done = step(grid, actions, subkey)
        
        if done[0]:
            print("\nGAME OVER")
            break
            
        time.sleep(0.3) # Wait 0.3s so you can see what happened

if __name__ == "__main__":
    watch_game()
import jax
import jax.numpy as jnp
import time
from jax_2048.game import step, add_random_tile

# 1. CONFIGURATION
BATCH_SIZE = 4096 * 2
NUM_STEPS = 100

def benchmark():
    print(f"Preparing to simulate {BATCH_SIZE} games for {NUM_STEPS} steps...")
    
    # 2. INITIALIZATION
    key = jax.random.PRNGKey(0)
    keys = jax.random.split(key, BATCH_SIZE)
    
    grids = jnp.zeros((BATCH_SIZE, 4, 4), dtype=jnp.int32)
    
    init_func = jax.vmap(add_random_tile)
    grids = init_func(grids, keys)
    keys = jax.random.split(keys[0], BATCH_SIZE)
    grids = init_func(grids, keys)

    # 3. VECTORIZE THE STEP FUNCTION
    step_batch = jax.vmap(step, in_axes=(0, 0, 0))
    
    # JIT Compile
    step_fast = jax.jit(step_batch)

    # Generate random actions for the benchmark (0, 1, 2, or 3)
    key_actions = jax.random.PRNGKey(999)
    all_actions = jax.random.randint(key_actions, (NUM_STEPS, BATCH_SIZE), 0, 4)
    
    # Generate keys for every step
    key_steps = jax.random.split(jax.random.PRNGKey(123), NUM_STEPS * BATCH_SIZE)
    key_steps = key_steps.reshape(NUM_STEPS, BATCH_SIZE, 2)

    print("Compiling and Warming up... (This takes a second)")
    _ = step_fast(grids, all_actions[0], key_steps[0])
    
    # 4. THE RACE
    print("Running benchmark...")
    
    grids.block_until_ready() 
    start_time = time.time()

    current_grids = grids
    for i in range(NUM_STEPS):
        current_grids, rewards, dones = step_fast(current_grids, all_actions[i], key_steps[i])
    
    current_grids.block_until_ready() 
    end_time = time.time()
    
    # 5. RESULTS
    duration = end_time - start_time
    total_steps = BATCH_SIZE * NUM_STEPS
    sps = total_steps / duration
    
    print(f"\n--- RESULTS ---")
    print(f"Total Steps: {total_steps:,}")
    print(f"Time:        {duration:.4f} seconds")
    print(f"Speed:       {sps:,.0f} steps/second")
    print(f"----------------")
    
    if sps > 1_000_000:
        print("🚀 Status: HYPERSPEED (Ready for RL)")
    elif sps > 100_000:
        print("✅ Status: FAST (Good for CPU)")
    else:
        print("⚠️ Status: SLOW (Something might be wrong)")

if __name__ == "__main__":
    benchmark()
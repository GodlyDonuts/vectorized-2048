import jax
import jax.numpy as jnp
from jax import jit, vmap

# ==========================================
# 1. CORE LOGIC (The Math)
# ==========================================

def shift_row_left(row):
    is_nonzero = row != 0
    indices = jnp.argsort(~is_nonzero, stable=True)
    return row[indices]

def merge_row_left(row):
    score = 0
    
    # --- Pair 1 (Indices 0 and 1) ---
    cond = (row[0] == row[1]) & (row[0] != 0)
    # If merge: row[0] doubles, row[1] becomes 0
    score += (row[0] * 2) * cond
    row = row.at[0].set(row[0] * (1 + cond))
    row = row.at[1].set(row[1] * (1 - cond))
    
    # --- Pair 2 (Indices 1 and 2) ---
    cond = (row[1] == row[2]) & (row[1] != 0)
    score += (row[1] * 2) * cond
    row = row.at[1].set(row[1] * (1 + cond))
    row = row.at[2].set(row[2] * (1 - cond))

    # --- Pair 3 (Indices 2 and 3) ---
    cond = (row[2] == row[3]) & (row[2] != 0)
    score += (row[2] * 2) * cond
    row = row.at[2].set(row[2] * (1 + cond))
    row = row.at[3].set(row[3] * (1 - cond))
    
    return row, score

def move_row_left(row):
    """Executes one full row move: Shift -> Merge -> Shift."""
    row = shift_row_left(row)
    row, score = merge_row_left(row)
    row = shift_row_left(row)
    return row, score

# Vectorize to work on the whole grid at once (treats grid as a list of rows)
move_grid_left = vmap(move_row_left)

# ==========================================
# 2. ROTATION LOGIC (The Linear Algebra Trick)
# ==========================================

def execute_move(grid, action):
    """
    actions: 0=Left, 1=Up, 2=Right, 3=Down
    We rotate the board so the desired direction becomes 'Left',
    execute the move, and then rotate back.
    """
    # 1. Transform grid to "Left-facing" perspective
    grid_oriented = jax.lax.switch(action, [
        lambda g: g,              # 0: Left (No change)
        lambda g: g.T,            # 1: Up (Transpose)
        lambda g: jnp.fliplr(g),  # 2: Right (Flip Mirror)
        lambda g: jnp.fliplr(g.T) # 3: Down (Transpose + Flip)
    ], grid)

    # 2. Apply move
    new_grid_oriented, scores = move_grid_left(grid_oriented)
    total_score = jnp.sum(scores)

    # 3. Transform back to original perspective
    final_grid = jax.lax.switch(action, [
        lambda g: g,
        lambda g: g.T,            # Transpose is its own inverse
        lambda g: jnp.fliplr(g),  # Flip is its own inverse
        lambda g: jnp.fliplr(g).T # Reverse: Flip then Transpose
    ], new_grid_oriented)

    return final_grid, total_score

# ==========================================
# 3. GAME STATE MANAGEMENT
# ==========================================

def add_random_tile(grid, key):
    """Adds a 2 (90%) or 4 (10%) to a random empty slot."""
    key_loc, key_val = jax.random.split(key)
    
    # 1. Find empty spots
    is_empty = (grid == 0)
    
    # 2. Check if we CAN add (This is a boolean tracer, not a Python bool)
    can_add = jnp.any(is_empty)
    
    # 3. Lottery for location
    random_scores = jax.random.uniform(key_loc, grid.shape)
    random_scores = random_scores + (is_empty * 1000.0) - 1000.0
    winner_idx = jnp.argmax(random_scores.ravel())
    
    # 4. Determine value (10% chance of 4)
    should_be_4 = jax.random.bernoulli(key_val, p=0.1)
    val = 2 + 2 * should_be_4
    
    # 5. Create a CANDIDATE grid (with the tile added)
    grid_flat = grid.ravel()
    candidate_flat = grid_flat.at[winner_idx].set(val)
    candidate_grid = candidate_flat.reshape(grid.shape)
    
    return jax.lax.select(can_add, candidate_grid, grid)

@jit
def step(grid, action, key):
    """
    The main game loop function.
    Returns: next_grid, reward, done
    """
    key_step, key_spawn = jax.random.split(key)
    
    # 1. Try to move
    next_grid, reward = execute_move(grid, action)
    
    # 2. Did the board change?
    changed = jnp.any(next_grid != grid)
    
    # 3. If changed, spawn a tile. If not, keep 'next_grid' as is.
    final_grid = jax.lax.cond(
        changed,
        lambda g: add_random_tile(g, key_spawn),
        lambda g: g,
        next_grid
    )
    
    # 4. Check if Done
    done = jnp.all(final_grid != 0)
    
    return final_grid, reward, done

# ==========================================
# 4. MANUAL PLAY (For Testing)
# ==========================================

if __name__ == "__main__":
    # Initialize
    key = jax.random.PRNGKey(42)
    grid = jnp.zeros((4, 4), dtype=jnp.int32)
    grid = add_random_tile(grid, key)
    key, subkey = jax.random.split(key)
    grid = add_random_tile(grid, subkey)

    print("Controls: 0=Left, 1=Up, 2=Right, 3=Down, q=Quit")
    
    while True:
        print("\n" + "-"*20)
        print(grid)
        user_input = input("Move: ")
        
        if user_input == 'q':
            break
            
        try:
            action = int(user_input)
            if action not in [0, 1, 2, 3]: raise ValueError
        except:
            print("Invalid input.")
            continue
            
        key, subkey = jax.random.split(key)
        grid, reward, done = step(grid, action, subkey)
        
        print(f"Reward: {reward}")
        if done:
            print("Game Over!")
            break
# vectorized-2048

[![JAX](https://img.shields.io/badge/JAX-Accelerated-blue?logo=google&style=for-the-badge)](https://github.com/google/jax)
[![Flax](https://img.shields.io/badge/Flax-Neural_Network-orange?style=for-the-badge)](https://github.com/google/flax)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

> **A massively parallel Deep Reinforcement Learning environment for 2048, written entirely in JAX.**
>
> Simulates **4,096 games simultaneously** on a single GPU/TPU, achieving training speeds of over **200,000 steps per second** (on M1 Air) and inference speeds of **2M+ steps per second**.

---

## ⚡️ The "Why"
Standard Python implementations of 2048 (using lists or NumPy) are CPU-bound and sequential. Training a Deep Q-Network (DQN) on a CPU can take days to reach "Grandmaster" level performance.

By rewriting the entire game logic as **stateless, branchless matrix transformations** in JAX, we can:
1.  **Vectorize everything:** Run thousands of environments in parallel as a single batch operation.
2.  **Eliminate bottlenecks:** The simulation *and* the neural network training loop run entirely on the GPU (end-to-end), with zero CPU-GPU data transfer overhead.
3.  **Differentiable Physics:** The game engine itself is differentiable, opening doors for model-based RL approaches.

## 🚀 Benchmarks (M1 MacBook Air)

| Implementation | Type | Steps / Second | Speedup |
| :--- | :--- | :--- | :--- |
| Standard Python | CPU (Sequential) | ~500 | 1x |
| **Vectorized JAX** | **GPU (Parallel)** | **~2,000,000** | **4,000x** |

*(Benchmarks run using `benchmark.py` with batch size 8192)*

## 🧠 Model Architecture

This project implements a **Dueling Deep Q-Network (Dueling DQN)** with several optimizations for the grid-based nature of 2048:

* **Learned Embeddings:** Instead of One-Hot encoding, the model learns a 64-dimensional embedding for each tile value, allowing it to understand semantic relationships (e.g., `1024` is "similar" to `2048`).
* **Custom Kernels:** Uses parallel `(4,1)` and `(1,4)` Convolutional layers to capture row-sliding and column-sliding patterns specifically.
* **Dueling Streams:** Separates state-value estimation ($V(s)$) from action-advantage estimation ($A(s,a)$) for better sample efficiency.
* **Double DQN:** Decouples action selection from target calculation to reduce maximization bias.

## 🛠️ Installation

```bash
# Clone the repository
git clone [https://github.com/yourusername/vectorized-2048.git](https://github.com/yourusername/vectorized-2048.git)
cd vectorized-2048

# Install dependencies (JAX, Flax, Optax)
pip install jax jaxlib flax optax

```

## 🎮 Usage

### 1. Train the Agent

Start the massively parallel training loop. This will simulate 4,096 environments at once.

```bash
# Run as a module to handle imports correctly
python -m jax_2048.train

```

* *Checkpoints are saved automatically to `brain.msgpack`.*
* *Training stabilizes around 60k-100k steps.*

### 2. Watch it Play

Load the trained brain and watch a game in real-time (human speed).

```bash
python -m jax_2048.watch

```

### 3. Benchmark Your Hardware

Test your raw GPU throughput.

```bash
python -m jax_2048.benchmark

```

## 📂 Project Structure

```text
vectorized-2048/
├── jax_2048/
│   ├── game.py       # The Physics Engine: Branchless JAX logic
│   ├── model.py      # The Brain: Dueling DQN Architecture
│   ├── train.py      # The Teacher: Parallel Training Loop
│   └── watch.py      # Visualization script
└── README.md

```

## 🔬 Technical Highlights (For Recruiters)

* **Branchless Programming:** The game logic contains **zero `if` statements**. All logic (merging, sliding, spawning) is implemented using boolean masking and mathematical operations (`jax.lax.select`, `jnp.where`) to ensure JIT compilation compatibility.
* **Logarithmic Reward Scaling:** Rewards are scaled using  to prevent gradient explosions when high-value tiles (2048, 4096) are merged.
* **Gradient Clipping:** Applied via Optax to ensure training stability during late-game states.

## 📜 License

MIT

```
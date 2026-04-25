# 12D Production Distributed Training (DDP)

This folder contains the **Production-Grade** infrastructure for training the 12D Cosmic Synapse Transformer across multiple GPUs or nodes.

## 🚀 How to Run

We use `torchrun` for robust process management.

### Single Node, Multiple GPUs
To train on a single machine with 4 GPUs:
```bash
torchrun --nproc_per_node=4 dist_train_12d.py --batch_size 32
```

### Multi-Node (Cluster)
To train on 2 nodes (Master + Worker):

**On Master (Node 0):**
```bash
torchrun --nproc_per_node=4 --nnodes=2 --node_rank=0 --master_addr="192.168.1.1" --master_port=1234 dist_train_12d.py
```

**On Worker (Node 1):**
```bash
torchrun --nproc_per_node=4 --nnodes=2 --node_rank=1 --master_addr="192.168.1.1" --master_port=1234 dist_train_12d.py
```

## 🛠️ Configuration

*   **`dist_train_12d.py`**: The main training script (Optimized for 12D).
*   **`--batch_size`**: Per-GPU batch size.
*   **`--d_model`**: Model dimension (default 256).

## 💡 Production Features
*   **12D Optimization**: Tailored for the specific compute patterns of the Cosmic Synapse.
*   **Stability**: Uses proven DDP patterns for long-running jobs.

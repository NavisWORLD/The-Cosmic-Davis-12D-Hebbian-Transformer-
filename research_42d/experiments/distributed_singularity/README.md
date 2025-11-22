# Distributed Singularity Training (DDP)

This folder contains the production-ready infrastructure for training the Cosmic Synapse (12D) and Hyper-Cosmic (42D) models across multiple GPUs or multiple nodes.

## 🚀 How to Run

We use `torchrun` (standard PyTorch launcher) to handle process creation and synchronization.

### Single Node, Multiple GPUs
To train on a single machine with 4 GPUs:
```bash
torchrun --nproc_per_node=4 dist_train.py --model 42D --batch_size 32
```

### Multi-Node (Cluster)
To train on 2 nodes (Master + Worker):

**On Master (Node 0):**
```bash
torchrun --nproc_per_node=4 --nnodes=2 --node_rank=0 --master_addr="192.168.1.1" --master_port=1234 dist_train.py
```

**On Worker (Node 1):**
```bash
torchrun --nproc_per_node=4 --nnodes=2 --node_rank=1 --master_addr="192.168.1.1" --master_port=1234 dist_train.py
```

## 🛠️ Configuration

*   **`dist_train.py`**: The main training script.
*   **`--model`**: Choose `12D` or `42D`.
*   **`--batch_size`**: Per-GPU batch size.
*   **`--d_model`**: Model dimension (default 256).

## 💡 Key Features
*   **DistributedDataParallel (DDP)**: Synchronizes gradients across all devices.
*   **DistributedSampler**: Ensures each GPU sees a unique slice of the dataset.
*   **Rank-0 Logging**: Only the master process prints logs to avoid console spam.

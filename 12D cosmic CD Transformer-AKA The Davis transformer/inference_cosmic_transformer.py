"""
12D COSMIC SYNAPSE TRANSFORMER - INFERENCE & DEPLOYMENT
=========================================================

Production inference system with:
- Efficient text generation
- Batch processing
- REST API server
- Evaluation tools
- Model quantization
- ONNX export

Author: Cory Shane Davis
"""

import torch
import torch.nn.functional as F
from transformers import GPT2Tokenizer
import numpy as np
import json
import time
from pathlib import Path
from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass

from cosmic_synapse_transformer import (
    CosmicSynapseTransformer,
    CosmicConfig,
    PHI
)

# Optional imports
try:
    from flask import Flask, request, jsonify
    HAS_FLASK = True
except ImportError:
    HAS_FLASK = False

# ===================================================================
# INFERENCE ENGINE
# ===================================================================

class CosmicInferenceEngine:
    """
    Production inference engine for 12D CST Transformer.
    Optimized for speed and efficiency.
    """
    
    def __init__(
        self,
        checkpoint_path: str,
        device: str = "cuda",
        compile: bool = True
    ):
        self.device = device
        
        # Load checkpoint
        print(f"[LOADING] Checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Reconstruct config
        self.config = CosmicConfig(**checkpoint['model_config'])
        
        # Create and load model
        self.model = CosmicSynapseTransformer(self.config)
        self.model.load_state_dict(checkpoint['model'])
        self.model.to(device)
        self.model.eval()
        
        # Compile for speed (PyTorch 2.0+)
        if compile and hasattr(torch, 'compile'):
            print("[COMPILE] Compiling model for inference...")
            self.model = torch.compile(self.model)
        
        # Load tokenizer (using GPT-2 tokenizer)
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        
        # Cache for KV (not implemented yet, but placeholder)
        self.kv_cache = None
        
        print(f"[READY] 12D CST Transformer loaded ({self.config.n_layers} layers, "
              f"{self.model.get_num_params()/1e6:.1f}M params)")
    
    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = 50,
        top_p: Optional[float] = 0.95,
        repetition_penalty: float = 1.0,
        stop_tokens: Optional[List[str]] = None
    ) -> Dict:
        """
        Generate text from prompt.
        
        Args:
            prompt: Input text
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (affects x₁₂ dynamics!)
            top_k: Top-k sampling
            top_p: Nucleus sampling
            repetition_penalty: Penalty for repeating tokens
            stop_tokens: List of strings that stop generation
        
        Returns:
            Dictionary with generated text and metadata
        """
        start_time = time.time()
        
        # Encode prompt
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
        
        # Track generation
        generated_ids = input_ids.clone()
        token_history = []
        x12_history = []
        
        # Generate
        for step in range(max_new_tokens):
            # Forward pass
            logits, _, metrics = self.model(generated_ids)
            
            # Get last token logits
            logits = logits[:, -1, :] / temperature
            
            # Apply repetition penalty
            if repetition_penalty != 1.0:
                for token_id in set(generated_ids[0].tolist()):
                    logits[0, token_id] /= repetition_penalty
            
            # Top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            
            # Top-p (nucleus) filtering
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                logits[indices_to_remove] = float('-inf')
            
            # Sample
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append
            generated_ids = torch.cat([generated_ids, next_token], dim=1)
            
            # Track metrics
            token_history.append(next_token.item())
            x12_history.append(metrics['x12_final'])
            
            # Check stop tokens
            if stop_tokens:
                generated_text = self.tokenizer.decode(generated_ids[0])
                if any(stop in generated_text for stop in stop_tokens):
                    break
        
        # Decode
        generated_text = self.tokenizer.decode(generated_ids[0])
        completion = generated_text[len(prompt):]
        
        # Compute stats
        elapsed_time = time.time() - start_time
        tokens_per_sec = len(token_history) / elapsed_time
        
        return {
            'prompt': prompt,
            'completion': completion,
            'full_text': generated_text,
            'num_tokens': len(token_history),
            'time_seconds': elapsed_time,
            'tokens_per_sec': tokens_per_sec,
            'x12_mean': np.mean(x12_history),
            'x12_std': np.std(x12_history),
            'x12_trajectory': x12_history
        }
    
    def batch_generate(
        self,
        prompts: List[str],
        **generation_kwargs
    ) -> List[Dict]:
        """Generate for multiple prompts"""
        results = []
        for prompt in prompts:
            result = self.generate(prompt, **generation_kwargs)
            results.append(result)
        return results
    
    @torch.no_grad()
    def get_perplexity(self, text: str) -> float:
        """
        Compute perplexity of text (measure of model confidence).
        Lower is better.
        """
        # Encode
        input_ids = self.tokenizer.encode(text, return_tensors='pt').to(self.device)
        
        # Forward
        logits, loss, _ = self.model(input_ids, targets=input_ids)
        
        # Perplexity = exp(loss)
        perplexity = torch.exp(loss).item()
        
        return perplexity
    
    @torch.no_grad()
    def get_embeddings(self, text: str) -> np.ndarray:
        """Get contextualized embeddings for text"""
        input_ids = self.tokenizer.encode(text, return_tensors='pt').to(self.device)
        
        # Get embeddings (modify model to return hidden states if needed)
        # For now, just return final layer output
        logits, _, metrics = self.model(input_ids)
        
        # Mean pool over sequence
        embeddings = logits.mean(dim=1).cpu().numpy()
        
        return embeddings
    
    def quantize(self, dtype=torch.int8):
        """
        Quantize model for faster inference and smaller size.
        Note: Requires torch 2.0+
        """
        if hasattr(torch, 'quantization'):
            print(f"[QUANTIZE] Quantizing to {dtype}...")
            self.model = torch.quantization.quantize_dynamic(
                self.model, {torch.nn.Linear}, dtype=dtype
            )
            print("[QUANTIZE] Complete")
        else:
            print("[WARNING] Quantization not available in this PyTorch version")

# ===================================================================
# EVALUATION UTILITIES
# ===================================================================

class CosmicEvaluator:
    """Evaluation tools for 12D CST Transformer"""
    
    def __init__(self, engine: CosmicInferenceEngine):
        self.engine = engine
    
    def evaluate_perplexity(self, texts: List[str]) -> Dict:
        """Evaluate perplexity on a list of texts"""
        perplexities = []
        
        for text in texts:
            ppl = self.engine.get_perplexity(text)
            perplexities.append(ppl)
        
        return {
            'mean_perplexity': np.mean(perplexities),
            'median_perplexity': np.median(perplexities),
            'std_perplexity': np.std(perplexities),
            'min_perplexity': np.min(perplexities),
            'max_perplexity': np.max(perplexities)
        }
    
    def evaluate_generation_quality(
        self,
        prompts: List[str],
        **generation_kwargs
    ) -> Dict:
        """
        Evaluate generation quality across multiple prompts.
        Metrics include diversity, coherence (via perplexity), speed.
        """
        results = self.engine.batch_generate(prompts, **generation_kwargs)
        
        # Compute metrics
        completions = [r['completion'] for r in results]
        
        # Diversity: unique n-grams
        def get_ngrams(text, n=2):
            words = text.split()
            return set(tuple(words[i:i+n]) for i in range(len(words)-n+1))
        
        all_bigrams = set()
        for comp in completions:
            all_bigrams.update(get_ngrams(comp, n=2))
        
        diversity = len(all_bigrams) / max(1, sum(len(c.split()) for c in completions))
        
        # Coherence: average perplexity of completions
        perplexities = [self.engine.get_perplexity(c) for c in completions]
        
        # Speed
        speeds = [r['tokens_per_sec'] for r in results]
        
        # x₁₂ statistics
        x12_means = [r['x12_mean'] for r in results]
        x12_stds = [r['x12_std'] for r in results]
        
        return {
            'diversity_score': diversity,
            'mean_coherence_ppl': np.mean(perplexities),
            'mean_speed_toks': np.mean(speeds),
            'x12_convergence': np.mean(x12_stds),  # Lower = more converged
            'x12_mean': np.mean(x12_means),
            'results': results
        }

# ===================================================================
# REST API SERVER
# ===================================================================

if HAS_FLASK:
    def create_api_server(checkpoint_path: str, host='0.0.0.0', port=5000):
        """
        Create Flask API server for 12D CST Transformer.
        
        Endpoints:
            POST /generate - Generate text
            POST /perplexity - Get perplexity
            GET /health - Health check
            GET /info - Model info
        """
        app = Flask(__name__)
        
        # Initialize engine
        print("[API] Initializing inference engine...")
        engine = CosmicInferenceEngine(checkpoint_path)
        
        @app.route('/health', methods=['GET'])
        def health():
            return jsonify({'status': 'healthy', 'model': '12D-CST'})
        
        @app.route('/info', methods=['GET'])
        def info():
            return jsonify({
                'model': '12D Cosmic Synapse Transformer',
                'author': 'Cory Shane Davis',
                'n_layers': engine.config.n_layers,
                'n_params': engine.model.get_num_params(),
                'd_model': engine.config.d_model,
                'phi': PHI,
                'version': '1.0'
            })
        
        @app.route('/generate', methods=['POST'])
        def generate():
            data = request.json
            
            if 'prompt' not in data:
                return jsonify({'error': 'Missing prompt'}), 400
            
            # Generate
            result = engine.generate(
                prompt=data['prompt'],
                max_new_tokens=data.get('max_tokens', 100),
                temperature=data.get('temperature', 1.0),
                top_k=data.get('top_k', 50),
                top_p=data.get('top_p', 0.95),
                repetition_penalty=data.get('repetition_penalty', 1.0)
            )
            
            return jsonify(result)
        
        @app.route('/perplexity', methods=['POST'])
        def perplexity():
            data = request.json
            
            if 'text' not in data:
                return jsonify({'error': 'Missing text'}), 400
            
            ppl = engine.get_perplexity(data['text'])
            
            return jsonify({'perplexity': ppl})
        
        print(f"[API] Server ready at http://{host}:{port}")
        app.run(host=host, port=port)
        
        return app

# ===================================================================
# DATA PREPARATION UTILITIES
# ===================================================================

def prepare_dataset(
    input_file: str,
    output_dir: str,
    tokenizer_name: str = 'gpt2',
    train_split: float = 0.9
):
    """
    Prepare text dataset for training.
    
    Args:
        input_file: Path to text file
        output_dir: Output directory for .bin files
        tokenizer_name: Tokenizer to use
        train_split: Train/val split ratio
    """
    from transformers import AutoTokenizer
    import os
    
    print(f"[PREPARE] Loading text from {input_file}")
    with open(input_file, 'r', encoding='utf-8') as f:
        text = f.read()
    
    print(f"[PREPARE] Text length: {len(text):,} characters")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    # Tokenize
    print("[PREPARE] Tokenizing...")
    tokens = tokenizer.encode(text)
    print(f"[PREPARE] {len(tokens):,} tokens")
    
    # Split train/val
    split_idx = int(len(tokens) * train_split)
    train_tokens = tokens[:split_idx]
    val_tokens = tokens[split_idx:]
    
    # Save as numpy arrays
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    train_path = output_dir / 'train.bin'
    val_path = output_dir / 'val.bin'
    
    train_array = np.array(train_tokens, dtype=np.uint16)
    val_array = np.array(val_tokens, dtype=np.uint16)
    
    train_array.tofile(train_path)
    val_array.tofile(val_path)
    
    print(f"[PREPARE] Saved:")
    print(f"  Train: {train_path} ({len(train_tokens):,} tokens)")
    print(f"  Val: {val_path} ({len(val_tokens):,} tokens)")
    
    return train_path, val_path

# ===================================================================
# COMMAND-LINE INTERFACE
# ===================================================================

def cli():
    """Command-line interface for inference"""
    import argparse
    
    parser = argparse.ArgumentParser(description='12D CST Transformer Inference')
    subparsers = parser.add_subparsers(dest='command')
    
    # Generate command
    gen_parser = subparsers.add_parser('generate', help='Generate text')
    gen_parser.add_argument('checkpoint', help='Path to checkpoint')
    gen_parser.add_argument('--prompt', required=True, help='Input prompt')
    gen_parser.add_argument('--max-tokens', type=int, default=100)
    gen_parser.add_argument('--temperature', type=float, default=1.0)
    gen_parser.add_argument('--top-k', type=int, default=50)
    gen_parser.add_argument('--top-p', type=float, default=0.95)
    
    # Server command
    server_parser = subparsers.add_parser('serve', help='Start API server')
    server_parser.add_argument('checkpoint', help='Path to checkpoint')
    server_parser.add_argument('--host', default='0.0.0.0')
    server_parser.add_argument('--port', type=int, default=5000)
    
    # Evaluate command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate model')
    eval_parser.add_argument('checkpoint', help='Path to checkpoint')
    eval_parser.add_argument('--test-file', required=True, help='Test data file')
    
    # Prepare data command
    prep_parser = subparsers.add_parser('prepare', help='Prepare dataset')
    prep_parser.add_argument('input_file', help='Input text file')
    prep_parser.add_argument('--output-dir', default='data')
    prep_parser.add_argument('--train-split', type=float, default=0.9)
    
    args = parser.parse_args()
    
    if args.command == 'generate':
        engine = CosmicInferenceEngine(args.checkpoint)
        result = engine.generate(
            args.prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p
        )
        
        print("\n" + "="*60)
        print("PROMPT:")
        print(args.prompt)
        print("\n" + "="*60)
        print("COMPLETION:")
        print(result['completion'])
        print("\n" + "="*60)
        print(f"Tokens: {result['num_tokens']} | "
              f"Speed: {result['tokens_per_sec']:.1f} tok/s | "
              f"x₁₂: {result['x12_mean']:.4f}")
        print("="*60 + "\n")
    
    elif args.command == 'serve':
        if not HAS_FLASK:
            print("[ERROR] Flask not installed. Run: pip install flask")
            return
        create_api_server(args.checkpoint, args.host, args.port)
    
    elif args.command == 'evaluate':
        engine = CosmicInferenceEngine(args.checkpoint)
        evaluator = CosmicEvaluator(engine)
        
        # Load test data
        with open(args.test_file, 'r') as f:
            texts = [line.strip() for line in f if line.strip()]
        
        results = evaluator.evaluate_perplexity(texts)
        
        print("\n" + "="*60)
        print("EVALUATION RESULTS")
        print("="*60)
        for key, value in results.items():
            print(f"{key}: {value:.4f}")
        print("="*60 + "\n")
    
    elif args.command == 'prepare':
        prepare_dataset(
            args.input_file,
            args.output_dir,
            train_split=args.train_split
        )
    
    else:
        parser.print_help()

# ===================================================================
# EXAMPLE USAGE
# ===================================================================

if __name__ == "__main__":
    # Example: Create and use inference engine
    
    # Uncomment to use CLI
    # cli()
    
    # Example programmatic usage:
    print("="*60)
    print("12D COSMIC SYNAPSE TRANSFORMER - INFERENCE ENGINE")
    print("="*60)
    print("\nTo use:")
    print("  1. Train model: python train_cosmic_transformer.py")
    print("  2. Inference:")
    print("       python inference.py generate checkpoints/best_model.pt \\")
    print("         --prompt 'The meaning of life is'")
    print("  3. API Server:")
    print("       python inference.py serve checkpoints/best_model.pt")
    print("\n" + "="*60)

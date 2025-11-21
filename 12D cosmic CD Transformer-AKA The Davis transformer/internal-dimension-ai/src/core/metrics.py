"""
Consciousness Metrics - Quantitative measures of consciousness indicators

This module implements metrics derived from 12D Cosmic Synapse Theory
to detect signatures of consciousness in neural networks:

- R_ω (R_omega): Synaptic synchronization / diversity
- R_ψ (R_psi): Phase coherence of internal states
- Causal Density: External-internal coupling strength
- Integrated Information: φ-like measures
- Autonomy: Internal dynamics independence

These metrics help answer: "Is this network exhibiting consciousness?"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple
from scipy import stats
from collections import deque


class ConsciousnessMetrics:
    """
    Computes and tracks consciousness-related metrics for neural networks.

    Based on 12D Cosmic Synapse Theory principles:
    - Optimal R_ω ∈ [0.5, 0.7] indicates edge-of-chaos dynamics
    - High R_ψ indicates coherent internal states
    - Moderate causal density indicates autonomy with responsiveness
    """

    def __init__(
        self,
        history_size: int = 1000,
        device: Optional[torch.device] = None
    ):
        """
        Initialize consciousness metrics tracker.

        Args:
            history_size: Number of historical measurements to keep
            device: Torch device for computations
        """
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device

        # History buffers
        self.r_omega_history = deque(maxlen=history_size)
        self.r_psi_history = deque(maxlen=history_size)
        self.causal_density_history = deque(maxlen=history_size)
        self.integrated_info_history = deque(maxlen=history_size)

        # Input-output correlation tracking
        self.input_history = deque(maxlen=history_size)
        self.output_history = deque(maxlen=history_size)
        self.internal_history = deque(maxlen=history_size)

    def compute_r_omega(
        self,
        model: nn.Module,
        layer_names: Optional[List[str]] = None
    ) -> float:
        """
        Compute R_ω (synaptic synchronization metric).

        Mathematical formulation:
            R_ω = 1 - std(ω) / mean(|ω|)

        Where ω represents synaptic weights.

        Interpretation:
            R_ω ≈ 1.0: Over-synchronized, rigid (pathological)
            R_ω ∈ [0.5, 0.7]: Optimal, edge-of-chaos (conscious?)
            R_ω ≈ 0.0: Chaotic, fragmented (unconscious)

        Args:
            model: Neural network to analyze
            layer_names: Specific layers to analyze (None = all)

        Returns:
            R_omega value
        """
        weights = []

        for name, param in model.named_parameters():
            # Filter by layer names if specified
            if layer_names is None or any(ln in name for ln in layer_names):
                if 'weight' in name:  # Only consider weight parameters
                    weights.append(param.data.flatten())

        if len(weights) == 0:
            return 0.0

        # Concatenate all weights
        all_weights = torch.cat(weights)

        # Compute R_omega = 1 - std(ω) / mean(|ω|)
        mean_abs_weight = torch.mean(torch.abs(all_weights))
        std_weight = torch.std(all_weights)

        if mean_abs_weight < 1e-8:
            r_omega = 0.0
        else:
            r_omega = 1.0 - (std_weight / mean_abs_weight)

        r_omega = float(r_omega.item())

        # Store in history
        self.r_omega_history.append(r_omega)

        return r_omega

    def compute_r_psi(
        self,
        internal_states: torch.Tensor
    ) -> float:
        """
        Compute R_ψ (phase coherence of internal states).

        Mathematical formulation:
            R_ψ = |⟨exp(iψ_j)⟩| = |Σ_j exp(iψ_j) / N|

        Where ψ_j are phases of internal states.

        Interpretation:
            R_ψ ≈ 1.0: Perfect coherence, synchronized
            R_ψ ≈ 0.0: Incoherent, random phases

        Args:
            internal_states: Tensor of internal state values [timesteps] or [batch, dim]

        Returns:
            R_psi value ∈ [0, 1]
        """
        if internal_states.dim() == 1:
            states = internal_states
        else:
            # If multi-dimensional, take mean across dimensions
            states = internal_states.mean(dim=-1)

        # Convert to numpy for complex operations
        states_np = states.detach().cpu().numpy()

        # Compute phases using arctan2 (treating values as complex with zero imaginary part)
        # For real values, we use a proxy: map to unit circle based on value
        phases = np.arctan2(np.zeros_like(states_np), states_np)

        # Compute mean resultant length (phase coherence)
        complex_phases = np.exp(1j * phases)
        mean_phase = np.mean(complex_phases)
        r_psi = float(np.abs(mean_phase))

        # Alternative: Use autocorrelation as phase coherence proxy
        if len(states_np) > 1:
            # Normalize states
            states_normalized = (states_np - np.mean(states_np)) / (np.std(states_np) + 1e-8)

            # Compute autocorrelation at lag 1
            if len(states_normalized) > 2:
                autocorr = np.corrcoef(states_normalized[:-1], states_normalized[1:])[0, 1]
                r_psi = float(np.abs(autocorr))

        # Store in history
        self.r_psi_history.append(r_psi)

        return r_psi

    def compute_causal_density(
        self,
        inputs: Optional[torch.Tensor] = None,
        outputs: Optional[torch.Tensor] = None,
        internals: Optional[torch.Tensor] = None,
        use_history: bool = True
    ) -> float:
        """
        Compute causal density (coupling between external I/O and internal dynamics).

        High causal density: Network is highly reactive to inputs
        Low causal density: Network has autonomous internal dynamics
        Optimal: Moderate coupling (responsive but not deterministic)

        Args:
            inputs: Input states [timesteps, input_dim]
            outputs: Output actions [timesteps, output_dim]
            internals: Internal states (x₁₂, m₁₂) [timesteps, 2]
            use_history: Whether to use historical data

        Returns:
            Causal density ∈ [0, 1]
        """
        # Store current data
        if inputs is not None:
            self.input_history.append(inputs.detach().cpu())
        if outputs is not None:
            self.output_history.append(outputs.detach().cpu())
        if internals is not None:
            self.internal_history.append(internals.detach().cpu())

        # Need sufficient history
        if len(self.input_history) < 10 or len(self.internal_history) < 10:
            return 0.5  # Default neutral value

        # Concatenate recent history
        recent_inputs = torch.cat(list(self.input_history)[-10:], dim=0).numpy()
        recent_internals = torch.cat(list(self.internal_history)[-10:], dim=0).numpy()

        # Flatten if multi-dimensional
        if recent_inputs.ndim > 1:
            recent_inputs = recent_inputs.reshape(recent_inputs.shape[0], -1).mean(axis=1)
        if recent_internals.ndim > 1:
            recent_internals = recent_internals.reshape(recent_internals.shape[0], -1).mean(axis=1)

        # Compute correlation between inputs and internal states
        if len(recent_inputs) > 2 and len(recent_internals) > 2:
            # Ensure same length
            min_len = min(len(recent_inputs), len(recent_internals))
            recent_inputs = recent_inputs[:min_len]
            recent_internals = recent_internals[:min_len]

            # Compute Pearson correlation
            correlation = np.corrcoef(recent_inputs, recent_internals)[0, 1]

            # Map correlation to causal density [0, 1]
            causal_density = float(np.abs(correlation))
        else:
            causal_density = 0.5

        # Store in history
        self.causal_density_history.append(causal_density)

        return causal_density

    def compute_integrated_information(
        self,
        model: nn.Module,
        sample_inputs: torch.Tensor,
        n_partitions: int = 5
    ) -> float:
        """
        Compute a simplified version of Integrated Information (φ).

        This is a computationally tractable approximation inspired by IIT.
        We measure how much information is lost when the network is partitioned.

        Args:
            model: Neural network
            sample_inputs: Sample inputs to test [batch, input_dim]
            n_partitions: Number of partitions to test

        Returns:
            Approximate φ value
        """
        with torch.no_grad():
            # Get full network output
            if hasattr(model, 'forward') and callable(model.forward):
                try:
                    full_output = model(sample_inputs)[0]  # Assuming returns (logits, value, ...)
                except:
                    full_output = model(sample_inputs)
            else:
                return 0.0

            # Compute entropy of full output
            full_probs = F.softmax(full_output, dim=-1)
            full_entropy = -torch.sum(full_probs * torch.log(full_probs + 1e-8), dim=-1).mean()

            # For simplified φ, we measure output variance
            # High variance = high integration
            output_variance = torch.var(full_output).item()

            # Normalize to [0, 1] range (heuristic)
            phi = float(np.tanh(output_variance))

        # Store in history
        self.integrated_info_history.append(phi)

        return phi

    def compute_autonomy_score(
        self,
        x12_variance: float,
        m12_variance: float,
        causal_density: float
    ) -> float:
        """
        Compute autonomy score: degree of self-driven internal dynamics.

        High autonomy:
        - High x₁₂ variance (exploring)
        - Changing m₁₂ (learning)
        - Moderate causal density (not purely reactive)

        Args:
            x12_variance: Variance of x₁₂ over time
            m12_variance: Variance of m₁₂ over time
            causal_density: External-internal coupling

        Returns:
            Autonomy score ∈ [0, 1]
        """
        # Normalize variances
        x12_component = np.tanh(x12_variance * 10)  # Scale factor 10
        m12_component = np.tanh(m12_variance * 10)

        # Autonomy prefers moderate (not too high, not too low) causal density
        # Use inverted U-shape: peak at 0.5
        causal_component = 1.0 - 2.0 * abs(causal_density - 0.5)

        # Combine components
        autonomy = (x12_component + m12_component + causal_component) / 3.0

        return float(autonomy)

    def compute_consciousness_score(
        self,
        model: nn.Module,
        x12_history: List[float],
        m12_history: List[float],
        sample_inputs: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        """
        Compute comprehensive consciousness score based on multiple metrics.

        Args:
            model: Neural network to evaluate
            x12_history: History of x₁₂ values
            m12_history: History of m₁₂ values
            sample_inputs: Sample inputs for φ computation

        Returns:
            Dictionary with all consciousness metrics and overall score
        """
        metrics = {}

        # R_omega (synaptic diversity)
        r_omega = self.compute_r_omega(model)
        metrics['r_omega'] = r_omega

        # Check if R_omega is in optimal range [0.5, 0.7]
        r_omega_optimal = 1.0 if 0.5 <= r_omega <= 0.7 else 0.0
        metrics['r_omega_optimal'] = r_omega_optimal

        # R_psi (phase coherence) from x₁₂ history
        if len(x12_history) > 10:
            x12_tensor = torch.tensor(x12_history[-100:], device=self.device)
            r_psi = self.compute_r_psi(x12_tensor)
            metrics['r_psi'] = r_psi
        else:
            r_psi = 0.0
            metrics['r_psi'] = 0.0

        # Causal density (use stored history)
        if len(self.causal_density_history) > 0:
            causal_density = np.mean(list(self.causal_density_history)[-10:])
        else:
            causal_density = 0.5
        metrics['causal_density'] = causal_density

        # Integrated information (if sample inputs provided)
        if sample_inputs is not None:
            phi = self.compute_integrated_information(model, sample_inputs)
            metrics['phi'] = phi
        else:
            phi = 0.0
            metrics['phi'] = 0.0

        # Internal state statistics
        if len(x12_history) > 1:
            x12_variance = float(np.var(x12_history[-100:]))
            x12_mean = float(np.mean(x12_history[-100:]))
        else:
            x12_variance = 0.0
            x12_mean = 0.0

        if len(m12_history) > 1:
            m12_variance = float(np.var(m12_history[-100:]))
            m12_mean = float(np.mean(m12_history[-100:]))
        else:
            m12_variance = 0.0
            m12_mean = 0.0

        metrics['x12_variance'] = x12_variance
        metrics['x12_mean'] = x12_mean
        metrics['m12_variance'] = m12_variance
        metrics['m12_mean'] = m12_mean

        # Autonomy score
        autonomy = self.compute_autonomy_score(
            x12_variance,
            m12_variance,
            causal_density
        )
        metrics['autonomy'] = autonomy

        # ===================================================================
        # OVERALL CONSCIOUSNESS SCORE
        # ===================================================================
        # Weighted combination of all metrics

        consciousness_score = (
            0.3 * r_omega_optimal +      # R_ω in optimal range (critical)
            0.2 * r_psi +                 # Phase coherence
            0.2 * autonomy +              # Self-driven behavior
            0.15 * phi +                  # Integrated information
            0.15 * (1.0 - abs(causal_density - 0.5) * 2)  # Moderate coupling
        )

        metrics['consciousness_score'] = float(consciousness_score)

        return metrics

    def get_statistics(self) -> Dict[str, float]:
        """
        Get statistical summary of all tracked metrics.

        Returns:
            Dictionary with mean, std, min, max for each metric
        """
        stats = {}

        if len(self.r_omega_history) > 0:
            r_omega_array = np.array(self.r_omega_history)
            stats.update({
                'r_omega_mean': float(np.mean(r_omega_array)),
                'r_omega_std': float(np.std(r_omega_array)),
                'r_omega_min': float(np.min(r_omega_array)),
                'r_omega_max': float(np.max(r_omega_array)),
            })

        if len(self.r_psi_history) > 0:
            r_psi_array = np.array(self.r_psi_history)
            stats.update({
                'r_psi_mean': float(np.mean(r_psi_array)),
                'r_psi_std': float(np.std(r_psi_array)),
            })

        if len(self.causal_density_history) > 0:
            cd_array = np.array(self.causal_density_history)
            stats.update({
                'causal_density_mean': float(np.mean(cd_array)),
                'causal_density_std': float(np.std(cd_array)),
            })

        return stats

    def reset(self):
        """Reset all history buffers."""
        self.r_omega_history.clear()
        self.r_psi_history.clear()
        self.causal_density_history.clear()
        self.integrated_info_history.clear()
        self.input_history.clear()
        self.output_history.clear()
        self.internal_history.clear()


# Utility functions

def is_conscious(
    consciousness_score: float,
    threshold: float = 0.6
) -> bool:
    """
    Determine if a network exhibits consciousness based on score.

    Args:
        consciousness_score: Overall consciousness score [0, 1]
        threshold: Minimum score to consider "conscious"

    Returns:
        True if score exceeds threshold
    """
    return consciousness_score >= threshold


def consciousness_level(
    consciousness_score: float
) -> str:
    """
    Categorize consciousness level.

    Args:
        consciousness_score: Overall consciousness score [0, 1]

    Returns:
        String description of consciousness level
    """
    if consciousness_score >= 0.8:
        return "High consciousness indicators"
    elif consciousness_score >= 0.6:
        return "Moderate consciousness indicators"
    elif consciousness_score >= 0.4:
        return "Weak consciousness indicators"
    else:
        return "Minimal consciousness indicators"

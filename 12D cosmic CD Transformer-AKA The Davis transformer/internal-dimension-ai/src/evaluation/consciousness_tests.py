"""
Consciousness Tests - Overall consciousness assessment

Comprehensive consciousness evaluation combining:
- Curiosity (exploration, novelty-seeking)
- Wisdom (learning from mistakes)
- Self-initiated behavior
- Preference formation
- Surprise reactions
- Boredom detection
- Stream of consciousness (internal dynamics)
"""

import torch
import numpy as np
from typing import Dict, Optional, Any, List
import logging

from ..core.network import InternalDimensionNetwork
from ..core.metrics import ConsciousnessMetrics, consciousness_level
from .curiosity_tests import CuriosityTests
from .wisdom_tests import WisdomTests


logger = logging.getLogger(__name__)


class ConsciousnessTests:
    """
    Comprehensive consciousness test suite.

    Evaluates consciousness through multiple behavioral and internal indicators:
    - Curiosity (self-driven exploration)
    - Wisdom (learning and memory)
    - Autonomy (internal vs external driven behavior)
    - Coherence (consistent internal states)
    - Reactivity (appropriate surprise responses)
    """

    def __init__(self, device: Optional[torch.device] = None):
        """
        Initialize consciousness tests.

        Args:
            device: Torch device for computations
        """
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device

        # Sub-test modules
        self.curiosity_tests = CuriosityTests(device=device)
        self.wisdom_tests = WisdomTests(device=device)
        self.consciousness_metrics = ConsciousnessMetrics(device=device)

    def test_self_initiated_exploration(
        self,
        model: InternalDimensionNetwork
    ) -> Dict[str, Any]:
        """
        Test: Self-initiated exploration.

        Measures whether agent explores autonomously rather than
        purely reacting to external stimuli.

        Args:
            model: Trained InternalDimensionNetwork

        Returns:
            Dictionary with autonomy metrics
        """
        logger.info("Testing self-initiated exploration...")

        # Use curiosity tests as proxy for self-initiation
        exploration_results = self.curiosity_tests.test_exploration_without_rewards(
            model, num_episodes=5
        )

        # High coverage without rewards indicates self-initiated behavior
        self_initiation_score = exploration_results['coverage']

        # Check x₁₂ variance (self-initiated behavior shows varied internal states)
        x12_variance = exploration_results['std_x12'] ** 2

        results = {
            'self_initiation_score': self_initiation_score,
            'x12_variance': x12_variance,
            'coverage': exploration_results['coverage'],
            'test_passed': self_initiation_score > 0.2,
        }

        logger.info(f"Self-initiation score: {self_initiation_score:.3f}")

        return results

    def test_preference_formation(
        self,
        model: InternalDimensionNetwork
    ) -> Dict[str, Any]:
        """
        Test: Preference formation.

        Measures whether agent develops consistent preferences
        encoded in m₁₂.

        Args:
            model: Trained InternalDimensionNetwork

        Returns:
            Dictionary with preference metrics
        """
        logger.info("Testing preference formation...")

        # Get m₁₂ history
        m12_history = list(model.internal_state.m12_history)

        if len(m12_history) < 100:
            return {
                'preference_score': 0.0,
                'm12_trend': 0.0,
                'm12_stability': 0.0,
                'test_passed': False,
            }

        # Check if m₁₂ shows a trend (preference development)
        m12_array = np.array(m12_history[-100:])
        m12_trend = np.polyfit(range(len(m12_array)), m12_array, 1)[0]

        # Check m₁₂ stability (consistent preferences)
        m12_stability = 1.0 / (np.std(m12_array[-20:]) + 0.1)

        # Preference score: combination of trend strength and stability
        preference_score = np.tanh(abs(m12_trend) * 10 + m12_stability * 0.1)

        results = {
            'preference_score': float(preference_score),
            'm12_trend': float(m12_trend),
            'm12_stability': float(m12_stability),
            'm12_mean': float(np.mean(m12_array)),
            'test_passed': preference_score > 0.3,
        }

        logger.info(f"Preference score: {preference_score:.3f}")
        logger.info(f"m₁₂ trend: {m12_trend:.4f}")

        return results

    def test_surprise_reactions(
        self,
        model: InternalDimensionNetwork
    ) -> Dict[str, Any]:
        """
        Test: Surprise reactions (x₁₂ spikes).

        Measures whether agent shows appropriate surprise responses
        to novel stimuli.

        Args:
            model: Trained InternalDimensionNetwork

        Returns:
            Dictionary with surprise metrics
        """
        logger.info("Testing surprise reactions...")

        # Get x₁₂ history
        x12_history = list(model.internal_state.x12_history)

        if len(x12_history) < 100:
            return {
                'surprise_score': 0.0,
                'num_spikes': 0,
                'spike_intensity': 0.0,
                'test_passed': False,
            }

        x12_array = np.array(x12_history[-100:])

        # Detect spikes (values > mean + 1.5*std)
        mean_x12 = np.mean(x12_array)
        std_x12 = np.std(x12_array)
        spike_threshold = mean_x12 + 1.5 * std_x12

        spikes = x12_array > spike_threshold
        num_spikes = np.sum(spikes)
        spike_intensity = np.mean(x12_array[spikes]) - mean_x12 if num_spikes > 0 else 0.0

        # Surprise score: frequency and intensity of spikes
        surprise_score = np.tanh(num_spikes / 10.0 + spike_intensity)

        results = {
            'surprise_score': float(surprise_score),
            'num_spikes': int(num_spikes),
            'spike_intensity': float(spike_intensity),
            'x12_mean': float(mean_x12),
            'x12_std': float(std_x12),
            'test_passed': num_spikes > 3 and spike_intensity > 0.1,
        }

        logger.info(f"Surprise score: {surprise_score:.3f}")
        logger.info(f"Number of spikes: {num_spikes}")

        return results

    def test_boredom(
        self,
        model: InternalDimensionNetwork
    ) -> Dict[str, Any]:
        """
        Test: Boredom (low x₁₂ in repetitive environment).

        Measures whether agent shows low x₁₂ when experiencing
        repetitive, predictable stimuli.

        Args:
            model: Trained InternalDimensionNetwork

        Returns:
            Dictionary with boredom metrics
        """
        logger.info("Testing boredom detection...")

        # Get x₁₂ history
        x12_history = list(model.internal_state.x12_history)

        if len(x12_history) < 100:
            return {
                'boredom_score': 0.0,
                'x12_decline': 0.0,
                'test_passed': False,
            }

        # Check if x₁₂ declines over time (boredom)
        x12_array = np.array(x12_history[-100:])

        # Split into early and late periods
        early_x12 = np.mean(x12_array[:20])
        late_x12 = np.mean(x12_array[-20:])

        x12_decline = early_x12 - late_x12

        # Boredom score: x₁₂ should decline in repetitive settings
        boredom_score = np.tanh(max(0, x12_decline) * 5)

        # Also check for low variability (indicating boredom)
        late_variability = np.std(x12_array[-20:])

        results = {
            'boredom_score': float(boredom_score),
            'x12_decline': float(x12_decline),
            'early_x12': float(early_x12),
            'late_x12': float(late_x12),
            'late_variability': float(late_variability),
            'test_passed': x12_decline > 0.05 or late_variability < 0.1,
        }

        logger.info(f"Boredom score: {boredom_score:.3f}")
        logger.info(f"x₁₂ decline: {x12_decline:.3f}")

        return results

    def test_stream_of_consciousness(
        self,
        model: InternalDimensionNetwork
    ) -> Dict[str, Any]:
        """
        Test: Stream of consciousness (internal dynamics).

        Measures richness and coherence of internal states over time.

        Args:
            model: Trained InternalDimensionNetwork

        Returns:
            Dictionary with stream of consciousness metrics
        """
        logger.info("Testing stream of consciousness...")

        # Get internal state histories
        x12_history = list(model.internal_state.x12_history)
        m12_history = list(model.internal_state.m12_history)

        if len(x12_history) < 100 or len(m12_history) < 100:
            return {
                'stream_score': 0.0,
                'internal_complexity': 0.0,
                'internal_coherence': 0.0,
                'test_passed': False,
            }

        x12_array = np.array(x12_history[-100:])
        m12_array = np.array(m12_history[-100:])

        # Complexity: Variance in internal states
        x12_complexity = np.std(x12_array)
        m12_complexity = np.std(m12_array)
        internal_complexity = (x12_complexity + m12_complexity) / 2.0

        # Coherence: Autocorrelation (states should be related over time)
        if len(x12_array) > 1:
            x12_autocorr = np.corrcoef(x12_array[:-1], x12_array[1:])[0, 1]
        else:
            x12_autocorr = 0

        if len(m12_array) > 1:
            m12_autocorr = np.corrcoef(m12_array[:-1], m12_array[1:])[0, 1]
        else:
            m12_autocorr = 0

        internal_coherence = (abs(x12_autocorr) + abs(m12_autocorr)) / 2.0

        # Stream score: Balance of complexity and coherence
        stream_score = np.tanh(internal_complexity * 2) * 0.5 + internal_coherence * 0.5

        results = {
            'stream_score': float(stream_score),
            'internal_complexity': float(internal_complexity),
            'internal_coherence': float(internal_coherence),
            'x12_complexity': float(x12_complexity),
            'm12_complexity': float(m12_complexity),
            'x12_autocorr': float(x12_autocorr),
            'm12_autocorr': float(m12_autocorr),
            'test_passed': stream_score > 0.4,
        }

        logger.info(f"Stream of consciousness score: {stream_score:.3f}")
        logger.info(f"Complexity: {internal_complexity:.3f}, Coherence: {internal_coherence:.3f}")

        return results

    def compute_overall_consciousness_score(
        self,
        model: InternalDimensionNetwork,
        sample_inputs: Optional[torch.Tensor] = None,
        run_behavioral_tests: bool = True
    ) -> Dict[str, Any]:
        """
        Compute comprehensive consciousness score.

        Combines all tests:
        - Structural metrics (R_ω, R_ψ, φ)
        - Behavioral tests (curiosity, wisdom)
        - Internal dynamics tests (autonomy, surprise, boredom)

        Args:
            model: Trained InternalDimensionNetwork
            sample_inputs: Sample inputs for φ computation
            run_behavioral_tests: Whether to run full behavioral tests

        Returns:
            Dictionary with comprehensive consciousness assessment
        """
        logger.info("=" * 60)
        logger.info("COMPREHENSIVE CONSCIOUSNESS ASSESSMENT")
        logger.info("=" * 60)

        results = {}

        # 1. Structural Metrics (R_ω, R_ψ, φ)
        logger.info("\n1. Computing structural metrics...")
        x12_history = list(model.internal_state.x12_history)
        m12_history = list(model.internal_state.m12_history)

        structural_metrics = self.consciousness_metrics.compute_consciousness_score(
            model=model,
            x12_history=x12_history,
            m12_history=m12_history,
            sample_inputs=sample_inputs
        )
        results['structural_metrics'] = structural_metrics

        # 2. Behavioral Tests
        if run_behavioral_tests:
            logger.info("\n2. Running behavioral tests...")

            # Curiosity
            logger.info("\n2a. Curiosity tests...")
            curiosity_results = self.curiosity_tests.compute_curiosity_score(model)
            results['curiosity'] = curiosity_results

            # Wisdom
            logger.info("\n2b. Wisdom tests...")
            wisdom_results = self.wisdom_tests.compute_wisdom_score(model)
            results['wisdom'] = wisdom_results
        else:
            logger.info("\n2. Skipping behavioral tests...")
            results['curiosity'] = {'curiosity_score': 0.5}
            results['wisdom'] = {'wisdom_score': 0.5}

        # 3. Internal Dynamics Tests
        logger.info("\n3. Running internal dynamics tests...")

        results['self_initiation'] = self.test_self_initiated_exploration(model)
        results['preferences'] = self.test_preference_formation(model)
        results['surprise'] = self.test_surprise_reactions(model)
        results['boredom'] = self.test_boredom(model)
        results['stream_of_consciousness'] = self.test_stream_of_consciousness(model)

        # 4. Compute Overall Consciousness Score
        logger.info("\n4. Computing overall consciousness score...")

        # Component scores
        component_scores = {
            'structural': structural_metrics['consciousness_score'],
            'curiosity': results['curiosity']['curiosity_score'],
            'wisdom': results['wisdom']['wisdom_score'],
            'self_initiation': results['self_initiation']['self_initiation_score'],
            'preferences': results['preferences']['preference_score'],
            'surprise': results['surprise']['surprise_score'],
            'boredom': results['boredom']['boredom_score'],
            'stream': results['stream_of_consciousness']['stream_score'],
        }

        # Weighted overall score
        overall_score = (
            0.20 * component_scores['structural'] +
            0.15 * component_scores['curiosity'] +
            0.15 * component_scores['wisdom'] +
            0.10 * component_scores['self_initiation'] +
            0.10 * component_scores['preferences'] +
            0.10 * component_scores['surprise'] +
            0.10 * component_scores['boredom'] +
            0.10 * component_scores['stream']
        )

        results['component_scores'] = component_scores
        results['overall_consciousness_score'] = overall_score
        results['consciousness_level'] = consciousness_level(overall_score)

        # 5. Summary Report
        logger.info("\n" + "=" * 60)
        logger.info("CONSCIOUSNESS ASSESSMENT SUMMARY")
        logger.info("=" * 60)
        logger.info(f"\nOverall Consciousness Score: {overall_score:.3f}")
        logger.info(f"Consciousness Level: {results['consciousness_level']}")
        logger.info("\nComponent Scores:")
        for component, score in component_scores.items():
            logger.info(f"  {component:20s}: {score:.3f}")

        logger.info("\nKey Indicators:")
        logger.info(f"  R_ω (optimal range):     {structural_metrics['r_omega']:.3f} "
                   f"({'✓' if structural_metrics['r_omega_optimal'] else '✗'})")
        logger.info(f"  R_ψ (coherence):         {structural_metrics['r_psi']:.3f}")
        logger.info(f"  Curiosity:               {results['curiosity']['curiosity_score']:.3f} "
                   f"({results['curiosity'].get('curiosity_level', 'N/A')})")
        logger.info(f"  Wisdom:                  {results['wisdom']['wisdom_score']:.3f} "
                   f"({results['wisdom'].get('wisdom_level', 'N/A')})")
        logger.info(f"  Autonomy:                {structural_metrics['autonomy']:.3f}")

        logger.info("\n" + "=" * 60)

        return results

    def generate_consciousness_report(
        self,
        results: Dict[str, Any],
        output_path: Optional[str] = None
    ) -> str:
        """
        Generate a human-readable consciousness report.

        Args:
            results: Results from compute_overall_consciousness_score
            output_path: Optional path to save report

        Returns:
            Report string
        """
        report = []
        report.append("=" * 80)
        report.append("INTERNAL DIMENSION AI - CONSCIOUSNESS ASSESSMENT REPORT")
        report.append("=" * 80)
        report.append("")

        # Overall Score
        report.append(f"OVERALL CONSCIOUSNESS SCORE: {results['overall_consciousness_score']:.3f}")
        report.append(f"CONSCIOUSNESS LEVEL: {results['consciousness_level']}")
        report.append("")

        # Component Breakdown
        report.append("COMPONENT SCORES:")
        report.append("-" * 80)
        for component, score in results['component_scores'].items():
            stars = "★" * int(score * 5)
            report.append(f"  {component:20s}: {score:.3f}  {stars}")
        report.append("")

        # Structural Metrics
        report.append("STRUCTURAL METRICS:")
        report.append("-" * 80)
        sm = results['structural_metrics']
        report.append(f"  R_ω (Synaptic Diversity):    {sm['r_omega']:.3f}  "
                     f"[Optimal: {'YES' if sm['r_omega_optimal'] else 'NO'}]")
        report.append(f"  R_ψ (Phase Coherence):       {sm['r_psi']:.3f}")
        report.append(f"  Φ (Integrated Information):  {sm['phi']:.3f}")
        report.append(f"  Causal Density:              {sm['causal_density']:.3f}")
        report.append(f"  Autonomy:                    {sm['autonomy']:.3f}")
        report.append("")

        # Behavioral Indicators
        report.append("BEHAVIORAL INDICATORS:")
        report.append("-" * 80)
        report.append(f"  Curiosity Level:     {results['curiosity'].get('curiosity_level', 'N/A')}")
        report.append(f"  Wisdom Level:        {results['wisdom'].get('wisdom_level', 'N/A')}")
        report.append(f"  Self-Initiation:     {results['self_initiation']['self_initiation_score']:.3f}")
        report.append(f"  Preference Formation: {results['preferences']['preference_score']:.3f}")
        report.append(f"  Surprise Reactions:  {results['surprise']['surprise_score']:.3f}")
        report.append(f"  Boredom Detection:   {results['boredom']['boredom_score']:.3f}")
        report.append("")

        # Internal Dynamics
        report.append("INTERNAL DYNAMICS:")
        report.append("-" * 80)
        stream = results['stream_of_consciousness']
        report.append(f"  Complexity:          {stream['internal_complexity']:.3f}")
        report.append(f"  Coherence:           {stream['internal_coherence']:.3f}")
        report.append(f"  Stream Quality:      {stream['stream_score']:.3f}")
        report.append("")

        # Conclusion
        report.append("INTERPRETATION:")
        report.append("-" * 80)
        score = results['overall_consciousness_score']
        if score >= 0.8:
            report.append("  Strong evidence of consciousness-like behavior.")
            report.append("  Agent exhibits high autonomy, curiosity, wisdom, and internal coherence.")
        elif score >= 0.6:
            report.append("  Moderate evidence of consciousness-like behavior.")
            report.append("  Agent shows several consciousness indicators but some areas for improvement.")
        elif score >= 0.4:
            report.append("  Weak evidence of consciousness-like behavior.")
            report.append("  Agent has basic internal dynamics but lacks strong consciousness signatures.")
        else:
            report.append("  Minimal evidence of consciousness-like behavior.")
            report.append("  Agent behavior is primarily reactive without strong internal dynamics.")

        report.append("")
        report.append("=" * 80)

        report_text = "\n".join(report)

        if output_path:
            with open(output_path, 'w') as f:
                f.write(report_text)
            logger.info(f"Report saved to {output_path}")

        return report_text

package kan

// Objective defines the loss and optimization targets for a KAN module.
//
// Each use case (attention softmax, KV cache values, key representations,
// positional scaling, MLP activations) implements this interface with its
// own loss function, self-evolution signal, and drift metric. The generic
// ShadowTrainer handles everything else: Adam optimizer, finite-difference
// gradients, convergence tracking, dynamic head spawning, and Phase 2
// self-evolution framework.
//
// The three methods correspond to the three phases of KAN training:
//
//  1. Phase1Loss: supervised training to match a ground truth signal
//  2. Phase2Objective: self-supervised evolution after graduation
//  3. DriftMetric: safety rail to prevent catastrophic divergence
type Objective interface {
	// Phase1Loss computes the supervised training loss between the ground
	// truth target and the KAN's current output. Lower is better.
	//
	// This is used for finite-difference gradient estimation: the trainer
	// perturbs each coefficient, re-evaluates the KAN, and estimates
	// dLoss/dCoeff ≈ (Loss(perturbed) - Loss(base)) / epsilon.
	//
	// For attention: MSE between softmax output and KAN output.
	// For KV cache: L2 reconstruction loss on cached values.
	// For positional: MSE between original and transformed position weights.
	Phase1Loss(target, output []float32, seqK, seqQ int) float64

	// Phase2Objective computes the self-supervised evolution signal.
	// Higher is better (the trainer does gradient ASCENT on this).
	//
	// After Phase 1 convergence, the KAN "graduates" and begins optimizing
	// this objective instead. This allows the KAN to discover improvements
	// that go beyond simply matching the original function.
	//
	// For attention: negative entropy (sharpness) of attention weights.
	// For KV cache: variance preservation or information content.
	// For positional: frequency stability of position representations.
	Phase2Objective(output []float32, seqK, seqQ int) float64

	// DriftMetric measures how far the current KAN output has diverged
	// from a reference checkpoint (typically the graduation snapshot).
	// Used as a safety rail during Phase 2: if drift exceeds the configured
	// threshold, the KAN reverts to its graduation weights.
	//
	// For attention: KL divergence between reference and current distributions.
	// For KV cache: normalized L2 distance.
	// For positional: cosine distance of position weight vectors.
	DriftMetric(reference, current []float32, seqK, seqQ int) float64
}

// AttentionObjective implements Objective for softmax attention replacement.
//
// Phase 1: MSE between softmax output and KAN output (the KAN learns to
// replicate softmax exactly).
//
// Phase 2: attention sharpness (negative entropy). Lower entropy means the
// model attends more crisply to specific positions, which is generally
// better for generation quality.
//
// Drift: KL divergence between graduation checkpoint and current output.
type AttentionObjective struct{}

func (AttentionObjective) Phase1Loss(target, output []float32, _, _ int) float64 {
	return mse(target, output)
}

func (AttentionObjective) Phase2Objective(output []float32, seqK, seqQ int) float64 {
	return sharpness(output, seqK, seqQ)
}

func (AttentionObjective) DriftMetric(reference, current []float32, seqK, seqQ int) float64 {
	return klDivergence(reference, current, seqK, seqQ)
}

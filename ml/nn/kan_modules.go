package nn

import (
	"log/slog"
	"sync"

	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/ml/nn/kan"
)

// KANModule represents a registered KAN training module. Each module type
// (attention, KV cache values, key representations, positional scaling, MLP)
// provides its own pending item type and flush/reset logic, but they all
// share the same lifecycle: collect tensors during forward pass, flush
// training after Compute(), reset counters after graph reservation.
type KANModule struct {
	Name    string
	Trainer *kan.ShadowTrainer
	Flush   func(trainer *kan.ShadowTrainer) // process pending items
	Reset   func()                           // discard pending items + reset counters
}

var (
	kanModules   []KANModule
	kanModulesMu sync.RWMutex
)

// RegisterKANModule adds a KAN training module to the global registry.
// Each module is flushed after Compute() and reset after graph reservation.
func RegisterKANModule(mod KANModule) {
	kanModulesMu.Lock()
	defer kanModulesMu.Unlock()
	kanModules = append(kanModules, mod)
	slog.Info("KAN module registered", "name", mod.Name)
}

// FlushAllKANModules processes deferred training work for all registered
// KAN modules. Must be called after Compute() so tensor data is materialized.
func FlushAllKANModules() {
	kanModulesMu.RLock()
	modules := kanModules
	kanModulesMu.RUnlock()

	for _, mod := range modules {
		mod.Flush(mod.Trainer)
	}
}

// ResetAllKANModules discards pending items and resets counters for all
// registered modules WITHOUT processing them. Use after graph reservation.
func ResetAllKANModules() {
	kanModulesMu.RLock()
	modules := kanModules
	kanModulesMu.RUnlock()

	for _, mod := range modules {
		mod.Reset()
	}
}

// PendingItem is a generic deferred training item. Module-specific code
// populates Input (the KAN's input data) and optionally Target (ground
// truth for Phase 1). After Compute(), the flush handler reads the
// tensors and calls TrainStep or Phase2Step.
type PendingItem struct {
	ModuleName string
	Key        string    // layer identifier (e.g., "layer_0")
	Input      ml.Tensor // input to the KAN transform
	Target     ml.Tensor // ground truth for Phase 1 (nil for converged layers)
	Shape      []int     // tensor shape for dimension extraction
	Converged  bool
}

// ExtractDimensions returns (seqK, effectiveSeqQ) from a tensor shape
// of the form [seqK, heads, seqQ]. Heads are flattened into the batch
// dimension since each head is treated as an independent pattern.
func ExtractDimensions(shape []int) (seqK, effectiveSeqQ int) {
	seqK = 1
	heads := 1
	seqQ := 1
	if len(shape) >= 1 {
		seqK = shape[0]
	}
	if len(shape) >= 2 {
		heads = shape[1]
	}
	if len(shape) >= 3 {
		seqQ = shape[2]
	}
	return seqK, seqQ * heads
}

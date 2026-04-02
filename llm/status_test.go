package llm

import (
	"os"
	"strings"
	"testing"
)

func TestStatusWriterErrorPrefixes(t *testing.T) {
	w := NewStatusWriter(os.Stderr)
	w.Write([]byte("CUDA error: out of memory\n"))
	if w.LastErrMsg == "" {
		t.Fatal("expected error message from CUDA error")
	}
	if !strings.Contains(w.LastErrMsg, "CUDA error") {
		t.Fatalf("unexpected error message: %s", w.LastErrMsg)
	}
}

func TestStatusWriterCrashDetection(t *testing.T) {
	tests := []struct {
		name    string
		input   string
		crashed bool
	}{
		{"panic", "panic: runtime error: invalid memory address\n", true},
		{"sigsegv", "signal: segmentation fault\n", true},
		{"sigabrt", "signal: aborted\n", true},
		{"fatal error", "fatal error: unexpected signal during runtime execution\n", true},
		{"normal output", "llm_load_print_meta: model type = 7B\n", false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			w := NewStatusWriter(os.Stderr)
			w.Write([]byte(tt.input))
			if tt.crashed && w.LastErrMsg == "" {
				t.Fatal("expected crash to be detected and LastErrMsg to be set")
			}
			if tt.crashed && !strings.Contains(w.LastErrMsg, "runner process crashed") {
				t.Fatalf("expected crash prefix in error message, got: %s", w.LastErrMsg)
			}
			if !tt.crashed && w.LastErrMsg != "" {
				t.Fatalf("unexpected error message: %s", w.LastErrMsg)
			}
		})
	}
}

func TestStatusWriterRecentLines(t *testing.T) {
	w := NewStatusWriter(os.Stderr)

	// Write more than maxRecentLines
	for i := range maxRecentLines + 10 {
		w.Write([]byte("line " + string(rune('A'+i%26)) + "\n"))
	}

	// Then trigger a crash
	w.Write([]byte("fatal error: unexpected signal\n"))

	if w.LastErrMsg == "" {
		t.Fatal("expected error message after crash")
	}
	// Should contain the crash line
	if !strings.Contains(w.LastErrMsg, "unexpected signal") {
		t.Fatalf("expected crash details in error, got: %s", w.LastErrMsg)
	}
}

func TestStatusWriterCrashOverridesExplicitError(t *testing.T) {
	w := NewStatusWriter(os.Stderr)
	// First set an explicit error via error prefix
	w.Write([]byte("CUDA error: out of memory\n"))

	// Then trigger a crash — crash message should take priority since it
	// includes recent output context (which contains the CUDA error too)
	w.Write([]byte("signal: segmentation fault\n"))
	if !strings.Contains(w.LastErrMsg, "runner process crashed") {
		t.Fatalf("expected crash message, got: %s", w.LastErrMsg)
	}
	// The crash message should include both the CUDA error and the signal
	if !strings.Contains(w.LastErrMsg, "CUDA error") {
		t.Fatalf("expected crash context to include original CUDA error, got: %s", w.LastErrMsg)
	}
	if !strings.Contains(w.LastErrMsg, "segmentation fault") {
		t.Fatalf("expected crash context to include signal, got: %s", w.LastErrMsg)
	}
}

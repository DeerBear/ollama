package llm

import (
	"bytes"
	"fmt"
	"os"
	"strings"
	"sync"
)

const maxRecentLines = 20

// StatusWriter is a writer that captures error messages from the llama runner process
type StatusWriter struct {
	LastErrMsg string
	out        *os.File

	mu          sync.Mutex
	recentLines []string
	crashed     bool
}

func NewStatusWriter(out *os.File) *StatusWriter {
	return &StatusWriter{
		out: out,
	}
}

// TODO - regex matching to detect errors like
// libcublasLt.so.11: cannot open shared object file: No such file or directory

var errorPrefixes = []string{
	"error:",
	"CUDA error",
	"ROCm error",
	"cudaMalloc failed",
	"\"ERR\"",
	"error loading model",
	"GGML_ASSERT",
	"Deepseek2 does not support K-shift",
}

// crashIndicators are strings that signal the runner process has crashed
// (e.g. Go runtime panic, signal-based crash, or fatal error in cgo code)
var crashIndicators = []string{
	"panic:",
	"fatal error:",
	"signal: segmentation fault",
	"signal: aborted",
	"SIGSEGV",
	"SIGABRT",
	"SIGBUS",
	"runtime error:",
	"unexpected fault address",
}

func (w *StatusWriter) Write(b []byte) (int, error) {
	w.mu.Lock()

	// Track recent output lines for crash diagnostics
	text := string(b)
	for _, line := range strings.Split(text, "\n") {
		trimmed := strings.TrimSpace(line)
		if trimmed == "" {
			continue
		}
		w.recentLines = append(w.recentLines, trimmed)
		if len(w.recentLines) > maxRecentLines {
			w.recentLines = w.recentLines[len(w.recentLines)-maxRecentLines:]
		}
	}

	// Detect crash indicators in the output — these take priority over generic error prefixes
	if !w.crashed {
		for _, indicator := range crashIndicators {
			if bytes.Contains(b, []byte(indicator)) {
				w.crashed = true
				w.LastErrMsg = fmt.Sprintf("runner process crashed: %s", strings.Join(w.recentLines, "; "))
				break
			}
		}
	} else {
		// Update crash message with additional context lines as they arrive
		w.LastErrMsg = fmt.Sprintf("runner process crashed: %s", strings.Join(w.recentLines, "; "))
	}

	// Only check for error prefixes if no crash was detected
	if !w.crashed {
		for _, prefix := range errorPrefixes {
			if _, after, ok := bytes.Cut(b, []byte(prefix)); ok {
				w.LastErrMsg = prefix + string(bytes.TrimSpace(after))
			}
		}
	}

	w.mu.Unlock()

	return w.out.Write(b)
}

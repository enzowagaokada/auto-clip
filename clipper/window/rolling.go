package window

import (
	"sync"
	"time"

	"auto-clip/clipper/preprocess"
)

// Rolling stores messages in chronological order and is safe for an EventSub
// writer and inference-loop reader to use concurrently.
type Rolling struct {
	mu       sync.RWMutex
	duration time.Duration
	messages []preprocess.Message
}

func New(duration time.Duration) *Rolling {
	return &Rolling{duration: duration}
}

func (r *Rolling) Add(message preprocess.Message) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.messages = append(r.messages, message)
}

// Snapshot returns messages in [now-duration, now]. Messages arriving out of
// order remain supported; the preprocessor performs a stable chronological sort.
func (r *Rolling) Snapshot(now time.Time) []preprocess.Message {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.pruneLocked(now)
	cutoff := now.Add(-r.duration)
	result := make([]preprocess.Message, 0, len(r.messages))
	for _, message := range r.messages {
		if !message.Time.Before(cutoff) && !message.Time.After(now) {
			result = append(result, message)
		}
	}
	return result
}

func (r *Rolling) Len(now time.Time) int {
	return len(r.Snapshot(now))
}

func (r *Rolling) pruneLocked(now time.Time) {
	cutoff := now.Add(-r.duration)
	kept := r.messages[:0]
	for _, message := range r.messages {
		if !message.Time.Before(cutoff) {
			kept = append(kept, message)
		}
	}
	r.messages = kept
}

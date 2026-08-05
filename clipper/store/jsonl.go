package store

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sync"
	"time"
)

type Candidate struct {
	SessionID          string    `json:"session_id"`
	CandidateID        string    `json:"candidate_id"`
	Streamer           string    `json:"streamer"`
	BroadcasterID      string    `json:"broadcaster_id,omitempty"`
	StreamID           string    `json:"stream_id,omitempty"`
	DetectedAt         time.Time `json:"detected_at"`
	TargetAt           time.Time `json:"target_at"`
	StreamOffsetSecond float64   `json:"stream_offset_seconds"`
	Score              float32   `json:"score"`
	Threshold          float32   `json:"threshold"`
	MessageCount       int       `json:"message_count"`
	UniqueUsers        int       `json:"unique_users"`
	ManifestSHA256     string    `json:"model_manifest_sha256"`
	RawFeatures        []float32 `json:"raw_features"`
	ScaledFeatures     []float32 `json:"scaled_features"`
	Messages           []Message `json:"messages"`
}

type Message struct {
	Time time.Time `json:"time"`
	User string    `json:"user,omitempty"`
	Text string    `json:"text"`
}

type SessionCounters struct {
	SessionID       string    `json:"session_id"`
	Streamer        string    `json:"streamer"`
	BroadcasterID   string    `json:"broadcaster_id,omitempty"`
	StreamID        string    `json:"stream_id,omitempty"`
	StreamStartedAt time.Time `json:"stream_started_at"`
	StartedAt       time.Time `json:"started_at"`
	EndedAt         time.Time `json:"ended_at"`
	MessagesSeen    uint64    `json:"messages_seen"`
	InferencesRun   uint64    `json:"inferences_run"`
	CandidatesFound uint64    `json:"candidates_found"`
	InferenceErrors uint64    `json:"inference_errors"`
}

type JSONL struct {
	mu         sync.Mutex
	candidates *os.File
	sessions   *os.File
}

func Open(candidatePath, sessionPath string) (*JSONL, error) {
	candidates, err := openAppend(candidatePath)
	if err != nil {
		return nil, fmt.Errorf("open candidates JSONL: %w", err)
	}
	sessions, err := openAppend(sessionPath)
	if err != nil {
		_ = candidates.Close()
		return nil, fmt.Errorf("open sessions JSONL: %w", err)
	}
	return &JSONL{candidates: candidates, sessions: sessions}, nil
}

func openAppend(path string) (*os.File, error) {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return nil, err
	}
	return os.OpenFile(path, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0o644)
}

func (s *JSONL) AppendCandidate(candidate Candidate) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	return appendAndSync(s.candidates, candidate)
}

// AppendSession writes a final immutable counter snapshot. A new process or
// reconnect uses a new session_id instead of mutating prior records.
func (s *JSONL) AppendSession(counters SessionCounters) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	return appendAndSync(s.sessions, counters)
}

func appendAndSync(file *os.File, value any) error {
	data, err := json.Marshal(value)
	if err != nil {
		return err
	}
	data = append(data, '\n')
	if _, err := file.Write(data); err != nil {
		return err
	}
	return file.Sync()
}

func (s *JSONL) Close() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	first := s.candidates.Close()
	second := s.sessions.Close()
	if first != nil {
		return first
	}
	return second
}

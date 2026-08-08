package store

import (
	"encoding/csv"
	"encoding/json"
	"fmt"
	"math"
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

// CandidateReview is the scrollable companion log written beside the full
// candidate JSONL. It intentionally omits chat, features, and checksums.
type CandidateReview struct {
	CandidateID       string  `json:"candidate_id"`
	Streamer          string  `json:"streamer"`
	Score             float32 `json:"score"`
	StreamOffsetStamp string  `json:"stream_offset_stamp"`
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

var reviewCSVHeader = []string{
	"candidate_id",
	"streamer",
	"score",
	"stream_offset_stamp",
	"review_label",
	"reason",
}

type JSONL struct {
	mu         sync.Mutex
	candidates *os.File
	sessions   *os.File
	reviews    *os.File
	reviewCSV  *os.File
}

func Open(candidatePath, sessionPath, reviewPath, reviewCSVPath string) (*JSONL, error) {
	candidates, err := openAppend(candidatePath)
	if err != nil {
		return nil, fmt.Errorf("open candidates JSONL: %w", err)
	}
	sessions, err := openAppend(sessionPath)
	if err != nil {
		_ = candidates.Close()
		return nil, fmt.Errorf("open sessions JSONL: %w", err)
	}
	reviews, err := openAppend(reviewPath)
	if err != nil {
		_ = candidates.Close()
		_ = sessions.Close()
		return nil, fmt.Errorf("open candidates review JSONL: %w", err)
	}
	reviewCSV, err := openReviewCSV(reviewCSVPath)
	if err != nil {
		_ = candidates.Close()
		_ = sessions.Close()
		_ = reviews.Close()
		return nil, fmt.Errorf("open candidates review CSV: %w", err)
	}
	return &JSONL{
		candidates: candidates,
		sessions:   sessions,
		reviews:    reviews,
		reviewCSV:  reviewCSV,
	}, nil
}

func openAppend(path string) (*os.File, error) {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return nil, err
	}
	return os.OpenFile(path, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0o644)
}

func openReviewCSV(path string) (*os.File, error) {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return nil, err
	}
	file, err := os.OpenFile(path, os.O_CREATE|os.O_RDWR, 0o644)
	if err != nil {
		return nil, err
	}
	info, err := file.Stat()
	if err != nil {
		_ = file.Close()
		return nil, err
	}
	if info.Size() == 0 {
		writer := csv.NewWriter(file)
		if err := writer.Write(reviewCSVHeader); err != nil {
			_ = file.Close()
			return nil, err
		}
		writer.Flush()
		if err := writer.Error(); err != nil {
			_ = file.Close()
			return nil, err
		}
		if err := file.Sync(); err != nil {
			_ = file.Close()
			return nil, err
		}
	}
	if _, err := file.Seek(0, 2); err != nil {
		_ = file.Close()
		return nil, err
	}
	return file, nil
}

func (s *JSONL) AppendCandidate(candidate Candidate) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if err := appendAndSync(s.candidates, candidate); err != nil {
		return err
	}
	review := CandidateReview{
		CandidateID:       candidate.CandidateID,
		Streamer:          candidate.Streamer,
		Score:             candidate.Score,
		StreamOffsetStamp: StreamOffsetStamp(candidate.StreamOffsetSecond),
	}
	if err := appendAndSync(s.reviews, review); err != nil {
		return err
	}
	return appendReviewCSV(s.reviewCSV, review)
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

func appendReviewCSV(file *os.File, review CandidateReview) error {
	writer := csv.NewWriter(file)
	if err := writer.Write([]string{
		review.CandidateID,
		review.Streamer,
		fmt.Sprintf("%.8g", review.Score),
		review.StreamOffsetStamp,
		"",
		"",
	}); err != nil {
		return err
	}
	writer.Flush()
	if err := writer.Error(); err != nil {
		return err
	}
	return file.Sync()
}

func (s *JSONL) Close() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	first := s.candidates.Close()
	second := s.sessions.Close()
	third := s.reviews.Close()
	fourth := s.reviewCSV.Close()
	if first != nil {
		return first
	}
	if second != nil {
		return second
	}
	if third != nil {
		return third
	}
	return fourth
}

// StreamOffsetStamp formats a stream offset for Twitch seek boxes / URLs.
func StreamOffsetStamp(seconds float64) string {
	total := int(math.Floor(seconds))
	if total < 0 {
		total = 0
	}
	hours := total / 3600
	minutes := (total % 3600) / 60
	secs := total % 60
	return fmt.Sprintf("%dh%dm%ds", hours, minutes, secs)
}

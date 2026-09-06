package store

import (
	"encoding/csv"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"slices"
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

const LiveSchemaVersion = 1

type InferenceTelemetry struct {
	SchemaVersion         int       `json:"schema_version"`
	SessionID             string    `json:"session_id"`
	Streamer              string    `json:"streamer"`
	BroadcasterID         string    `json:"broadcaster_id,omitempty"`
	StreamID              string    `json:"stream_id,omitempty"`
	ManifestSHA256        string    `json:"model_manifest_sha256"`
	InferenceAt           time.Time `json:"inference_at"`
	TargetAt              time.Time `json:"target_at"`
	Score                 float32   `json:"score"`
	Threshold             float32   `json:"threshold"`
	Crossed               bool      `json:"crossed"`
	Triggered             bool      `json:"triggered"`
	AboveThreshold        bool      `json:"above_threshold"`
	Armed                 bool      `json:"armed"`
	InCooldown            bool      `json:"in_cooldown"`
	CooldownUntil         time.Time `json:"cooldown_until,omitempty"`
	RawFeatures           []float32 `json:"raw_features"`
	CumulativeDroppedChat uint64    `json:"cumulative_dropped_chat"`
}

type Episode struct {
	SchemaVersion    int       `json:"schema_version"`
	EpisodeID        string    `json:"episode_id"`
	RecordType       string    `json:"record_type"`
	SessionID        string    `json:"session_id"`
	Streamer         string    `json:"streamer"`
	BroadcasterID    string    `json:"broadcaster_id,omitempty"`
	StreamID         string    `json:"stream_id,omitempty"`
	ManifestSHA256   string    `json:"model_manifest_sha256"`
	OnsetAt          time.Time `json:"onset_at"`
	OnsetScore       float32   `json:"onset_score"`
	PeakAt           time.Time `json:"peak_at"`
	PeakTargetAt     time.Time `json:"peak_target_at"`
	PeakWindowStart  time.Time `json:"peak_window_start"`
	PeakWindowEnd    time.Time `json:"peak_window_end"`
	PeakStreamOffset float64   `json:"peak_stream_offset_seconds"`
	PeakScore        float32   `json:"peak_score"`
	ClosedAt         time.Time `json:"closed_at"`
	DurationSeconds  float64   `json:"duration_seconds"`
	MinimumScore     float32   `json:"minimum_score"`
	MaximumScore     float32   `json:"maximum_score"`
	Threshold        float32   `json:"threshold"`
	CloseReason      string    `json:"close_reason"`
	RawFeatures      []float32 `json:"raw_features"`
	ScaledFeatures   []float32 `json:"scaled_features"`
	Messages         []Message `json:"messages"`
}

type EpisodeReview struct {
	EpisodeID         string  `json:"episode_id"`
	RecordType        string  `json:"record_type"`
	SessionID         string  `json:"session_id"`
	Streamer          string  `json:"streamer"`
	OnsetScore        float32 `json:"onset_score"`
	PeakScore         float32 `json:"peak_score"`
	StreamOffsetStamp string  `json:"stream_offset_stamp"`
}

// CandidateReview is the scrollable companion log written beside the full
// candidate JSONL. It intentionally omits chat, features, and checksums.
// SessionID joins to sessions.jsonl (where vod_id is stored once resolved).
type CandidateReview struct {
	CandidateID       string  `json:"candidate_id"`
	SessionID         string  `json:"session_id"`
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
	VODID           string    `json:"vod_id,omitempty"`
	StreamStartedAt time.Time `json:"stream_started_at"`
	StartedAt       time.Time `json:"started_at"`
	EndedAt         time.Time `json:"ended_at"`
	MessagesSeen    uint64    `json:"messages_seen"`
	InferencesRun   uint64    `json:"inferences_run"`
	CandidatesFound uint64    `json:"candidates_found"`
	InferenceErrors uint64    `json:"inference_errors"`
	EpisodesFound   uint64    `json:"episodes_found"`
	LocalPeaksFound uint64    `json:"local_peaks_found"`
	DroppedChat     uint64    `json:"dropped_chat"`
	UsefulSeconds   float64   `json:"useful_seconds"`
}

var reviewCSVHeader = []string{
	"candidate_id",
	"session_id",
	"streamer",
	"score",
	"stream_offset_stamp",
	"review_label",
	"reason",
}

var episodeReviewCSVHeader = []string{
	"episode_id",
	"record_type",
	"session_id",
	"streamer",
	"onset_score",
	"peak_score",
	"stream_offset_stamp",
	"review_label",
	"reason",
}

type JSONL struct {
	mu               sync.Mutex
	candidates       *os.File
	sessions         *os.File
	reviews          *os.File
	reviewCSV        *os.File
	telemetry        *os.File
	episodes         *os.File
	episodeReviews   *os.File
	episodeReviewCSV *os.File
}

type Paths struct {
	Candidates         string
	Sessions           string
	CandidateReviews   string
	CandidateReviewCSV string
	Telemetry          string
	Episodes           string
	EpisodeReviews     string
	EpisodeReviewCSV   string
}

func Open(paths Paths) (*JSONL, error) {
	candidates, err := openAppend(paths.Candidates)
	if err != nil {
		return nil, fmt.Errorf("open candidates JSONL: %w", err)
	}
	sessions, err := openAppend(paths.Sessions)
	if err != nil {
		_ = candidates.Close()
		return nil, fmt.Errorf("open sessions JSONL: %w", err)
	}
	reviews, err := openAppend(paths.CandidateReviews)
	if err != nil {
		_ = candidates.Close()
		_ = sessions.Close()
		return nil, fmt.Errorf("open candidates review JSONL: %w", err)
	}
	reviewCSV, err := openCSV(paths.CandidateReviewCSV, reviewCSVHeader)
	if err != nil {
		_ = candidates.Close()
		_ = sessions.Close()
		_ = reviews.Close()
		return nil, fmt.Errorf("open candidates review CSV: %w", err)
	}
	telemetry, err := openAppend(paths.Telemetry)
	if err != nil {
		closeFiles(candidates, sessions, reviews, reviewCSV)
		return nil, fmt.Errorf("open telemetry JSONL: %w", err)
	}
	episodes, err := openAppend(paths.Episodes)
	if err != nil {
		closeFiles(candidates, sessions, reviews, reviewCSV, telemetry)
		return nil, fmt.Errorf("open episodes JSONL: %w", err)
	}
	episodeReviews, err := openAppend(paths.EpisodeReviews)
	if err != nil {
		closeFiles(candidates, sessions, reviews, reviewCSV, telemetry, episodes)
		return nil, fmt.Errorf("open episode reviews JSONL: %w", err)
	}
	episodeReviewCSV, err := openCSV(paths.EpisodeReviewCSV, episodeReviewCSVHeader)
	if err != nil {
		closeFiles(candidates, sessions, reviews, reviewCSV, telemetry, episodes, episodeReviews)
		return nil, fmt.Errorf("open episode reviews CSV: %w", err)
	}
	return &JSONL{
		candidates:       candidates,
		sessions:         sessions,
		reviews:          reviews,
		reviewCSV:        reviewCSV,
		telemetry:        telemetry,
		episodes:         episodes,
		episodeReviews:   episodeReviews,
		episodeReviewCSV: episodeReviewCSV,
	}, nil
}

func openAppend(path string) (*os.File, error) {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return nil, err
	}
	return os.OpenFile(path, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0o644)
}

func openCSV(path string, header []string) (*os.File, error) {
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
		if err := writer.Write(header); err != nil {
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
	} else {
		if _, err := file.Seek(0, 0); err != nil {
			_ = file.Close()
			return nil, err
		}
		existing, err := csv.NewReader(file).Read()
		if err != nil {
			_ = file.Close()
			return nil, fmt.Errorf("read existing CSV header: %w", err)
		}
		if !slices.Equal(existing, header) {
			_ = file.Close()
			return nil, fmt.Errorf("existing CSV header %v does not match schema %v", existing, header)
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
		SessionID:         candidate.SessionID,
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

func (s *JSONL) AppendTelemetry(telemetry InferenceTelemetry) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if telemetry.SchemaVersion != LiveSchemaVersion {
		return fmt.Errorf("telemetry schema_version must be %d", LiveSchemaVersion)
	}
	return appendAndSync(s.telemetry, telemetry)
}

func (s *JSONL) AppendEpisode(episode Episode) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if episode.SchemaVersion != LiveSchemaVersion {
		return fmt.Errorf("episode schema_version must be %d", LiveSchemaVersion)
	}
	if err := appendAndSync(s.episodes, episode); err != nil {
		return err
	}
	review := EpisodeReview{
		EpisodeID: episode.EpisodeID, RecordType: episode.RecordType,
		SessionID: episode.SessionID, Streamer: episode.Streamer,
		OnsetScore: episode.OnsetScore, PeakScore: episode.PeakScore,
		StreamOffsetStamp: StreamOffsetStamp(episode.PeakStreamOffset),
	}
	if err := appendAndSync(s.episodeReviews, review); err != nil {
		return err
	}
	return appendEpisodeReviewCSV(s.episodeReviewCSV, review)
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
		review.SessionID,
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

func appendEpisodeReviewCSV(file *os.File, review EpisodeReview) error {
	writer := csv.NewWriter(file)
	if err := writer.Write([]string{
		review.EpisodeID,
		review.RecordType,
		review.SessionID,
		review.Streamer,
		fmt.Sprintf("%.8g", review.OnsetScore),
		fmt.Sprintf("%.8g", review.PeakScore),
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
	fifth := s.telemetry.Close()
	sixth := s.episodes.Close()
	seventh := s.episodeReviews.Close()
	eighth := s.episodeReviewCSV.Close()
	if first != nil {
		return first
	}
	if second != nil {
		return second
	}
	if third != nil {
		return third
	}
	if fourth != nil {
		return fourth
	}
	if fifth != nil {
		return fifth
	}
	if sixth != nil {
		return sixth
	}
	if seventh != nil {
		return seventh
	}
	return eighth
}

func closeFiles(files ...*os.File) {
	for _, file := range files {
		if file != nil {
			_ = file.Close()
		}
	}
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

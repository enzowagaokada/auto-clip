package core

import (
	"crypto/rand"
	"encoding/hex"
	"errors"
	"fmt"
	"sync"
	"time"

	"auto-clip/clipper/detection"
	"auto-clip/clipper/preprocess"
	"auto-clip/clipper/store"
	"auto-clip/clipper/window"
)

type Scorer interface {
	Score(preprocess.Encoded) (float32, error)
}

type Recorder interface {
	AppendCandidate(store.Candidate) error
	AppendSession(store.SessionCounters) error
	AppendTelemetry(store.InferenceTelemetry) error
	AppendEpisode(store.Episode) error
}

type Options struct {
	Streamer               string
	ReviewPartition        string
	BroadcasterID          string
	StreamID               string
	StreamStarted          time.Time
	ObservedAt             time.Time
	Window                 time.Duration
	TargetLag              time.Duration
	ManifestSHA256         string
	EpisodeCloseBelowTicks int
	EpisodeMaxDuration     time.Duration
	LocalPeaksPerHour      int
}

type Evaluation struct {
	At        time.Time
	TargetAt  time.Time
	Score     float32
	Threshold float32
	Triggered bool
	Candidate *store.Candidate
}

type Session struct {
	mu       sync.Mutex
	options  Options
	id       string
	started  time.Time
	buffer   *window.Rolling
	encoder  *preprocess.Encoder
	scorer   Scorer
	machine  *detection.Machine
	recorder Recorder
	episodes *episodeTracker
	counters store.SessionCounters
	closed   bool
}

func NewSession(options Options, encoder *preprocess.Encoder, scorer Scorer,
	machine *detection.Machine, recorder Recorder) (*Session, error) {
	if options.Streamer == "" || options.StreamStarted.IsZero() {
		return nil, errors.New("streamer and stream start time are required")
	}
	if options.ReviewPartition != "calibration" &&
		options.ReviewPartition != "confirmation" {
		return nil, errors.New("review partition must be calibration or confirmation")
	}
	if options.Window <= 0 {
		return nil, errors.New("window duration must be positive")
	}
	if options.TargetLag < 0 || options.TargetLag >= options.Window {
		return nil, errors.New("target lag must be in [0, window duration)")
	}
	if options.EpisodeCloseBelowTicks <= 0 || options.EpisodeMaxDuration <= 0 ||
		options.LocalPeaksPerHour <= 0 || options.LocalPeaksPerHour > 5 {
		return nil, errors.New("episode closure and local-maximum settings are invalid")
	}
	if encoder == nil || scorer == nil || machine == nil || recorder == nil {
		return nil, errors.New("encoder, scorer, state machine, and recorder are required")
	}
	id, err := newID()
	if err != nil {
		return nil, err
	}
	now := options.ObservedAt.UTC()
	if now.IsZero() {
		now = time.Now().UTC()
	}
	session := &Session{
		options:  options,
		id:       id,
		started:  now,
		buffer:   window.New(options.Window),
		encoder:  encoder,
		scorer:   scorer,
		machine:  machine,
		recorder: recorder,
		counters: store.SessionCounters{
			SessionID: id, ReviewPartition: options.ReviewPartition,
			Streamer:      options.Streamer,
			BroadcasterID: options.BroadcasterID, StreamID: options.StreamID,
			StreamStartedAt: options.StreamStarted.UTC(), StartedAt: now,
		},
	}
	session.episodes = newEpisodeTracker(options, id, recorder)
	return session, nil
}

func (s *Session) ID() string { return s.id }

func (s *Session) Warm(at time.Time) bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	return !s.closed &&
		!at.Before(s.started.Add(s.options.Window)) &&
		s.buffer.Len(at) > 0
}

func (s *Session) AddMessage(message preprocess.Message) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed {
		return errors.New("session is closed")
	}
	s.buffer.Add(message)
	s.counters.MessagesSeen++
	return nil
}

func (s *Session) Evaluate(at time.Time) (*store.Candidate, error) {
	evaluation, err := s.EvaluateDetailed(at)
	if err != nil {
		return nil, err
	}
	return evaluation.Candidate, nil
}

func (s *Session) EvaluateDetailed(at time.Time) (Evaluation, error) {
	return s.EvaluateDetailedWithDropped(at, 0)
}

func (s *Session) EvaluateDetailedWithDropped(
	at time.Time,
	cumulativeDroppedChat uint64,
) (Evaluation, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed {
		return Evaluation{}, errors.New("session is closed")
	}
	messages := s.buffer.Snapshot(at)
	targetAt := at.Add(-s.options.TargetLag)
	streamOffset := targetAt.Sub(s.options.StreamStarted).Seconds()
	if streamOffset < 0 {
		streamOffset = 0
	}
	encoded, err := s.encoder.Encode(preprocess.Input{
		Messages:           messages,
		WindowStart:        at.Add(-s.options.Window),
		WindowEnd:          at,
		StreamOffsetSecond: streamOffset,
	})
	if err != nil {
		s.counters.InferenceErrors++
		return Evaluation{}, err
	}
	score, err := s.scorer.Score(encoded)
	s.counters.InferencesRun++
	if err != nil {
		s.counters.InferenceErrors++
		return Evaluation{}, err
	}
	decision := s.machine.Observe(at, score)
	s.counters.DroppedChat = cumulativeDroppedChat
	evaluation := Evaluation{
		At: at.UTC(), TargetAt: targetAt.UTC(), Score: score,
		Threshold: decision.Threshold, Triggered: decision.Triggered,
	}
	telemetry := store.InferenceTelemetry{
		SchemaVersion: store.LiveSchemaVersion,
		SessionID:     s.id, ReviewPartition: s.options.ReviewPartition,
		Streamer:      s.options.Streamer,
		BroadcasterID: s.options.BroadcasterID, StreamID: s.options.StreamID,
		ManifestSHA256: s.options.ManifestSHA256,
		InferenceAt:    at.UTC(), TargetAt: targetAt.UTC(),
		Score: score, Threshold: decision.Threshold,
		Crossed: decision.Crossed, Triggered: decision.Triggered,
		AboveThreshold: decision.AboveThreshold, Armed: decision.Armed,
		InCooldown: decision.InCooldown, CooldownUntil: decision.CooldownUntil.UTC(),
		RawFeatures:           append([]float32(nil), encoded.RawFeatures...),
		CumulativeDroppedChat: cumulativeDroppedChat,
	}
	if err := s.recorder.AppendTelemetry(telemetry); err != nil {
		return Evaluation{}, fmt.Errorf("persist inference telemetry: %w", err)
	}
	rawMessages := make([]store.Message, len(messages))
	for i, message := range messages {
		rawMessages[i] = store.Message{
			Time: message.Time.UTC(), User: message.User, Text: message.Text,
		}
	}
	if decision.Triggered {
		candidateID, err := newID()
		if err != nil {
			return Evaluation{}, err
		}
		candidate := store.Candidate{
			SessionID: s.id, CandidateID: candidateID,
			Streamer: s.options.Streamer, BroadcasterID: s.options.BroadcasterID,
			StreamID:   s.options.StreamID,
			DetectedAt: at.UTC(), StreamOffsetSecond: streamOffset,
			TargetAt: targetAt.UTC(),
			Score:    score, Threshold: decision.Threshold, MessageCount: len(messages),
			UniqueUsers:    int(encoded.RawFeatures[1]),
			ManifestSHA256: s.options.ManifestSHA256,
			RawFeatures:    append([]float32(nil), encoded.RawFeatures...),
			ScaledFeatures: append([]float32(nil), encoded.Features...),
			Messages:       rawMessages,
		}
		if err := s.recorder.AppendCandidate(candidate); err != nil {
			// Persistence is part of the trigger contract. Rearm so a transient
			// storage failure does not silently discard an otherwise valid moment.
			s.machine.Reset()
			return Evaluation{}, fmt.Errorf("persist candidate: %w", err)
		}
		s.counters.CandidatesFound++
		evaluation.Candidate = &candidate
	}
	recordType, err := s.episodes.observe(episodePoint{
		at: at.UTC(), targetAt: targetAt.UTC(),
		windowStart: at.Add(-s.options.Window).UTC(), windowEnd: at.UTC(),
		streamOffsetSecond: streamOffset,
		score:              score, threshold: decision.Threshold, triggered: decision.Triggered,
		rawFeatures:    append([]float32(nil), encoded.RawFeatures...),
		scaledFeatures: append([]float32(nil), encoded.Features...),
		messages:       rawMessages,
	})
	if err != nil {
		return Evaluation{}, err
	}
	if recordType == "triggered" {
		s.counters.EpisodesFound++
	} else if recordType == "local_maximum" {
		s.counters.LocalPeaksFound++
	}
	return evaluation, nil
}

func (s *Session) Counters() store.SessionCounters {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.counters
}

func (s *Session) Close(at time.Time) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.closeLocked(at, s.counters.DroppedChat)
}

func (s *Session) CloseWithDropped(at time.Time, cumulativeDroppedChat uint64) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.closeLocked(at, cumulativeDroppedChat)
}

func (s *Session) closeLocked(at time.Time, cumulativeDroppedChat uint64) error {
	if s.closed {
		return nil
	}
	flushed, err := s.episodes.flush(at.UTC())
	if err != nil {
		return err
	}
	if flushed {
		s.counters.EpisodesFound++
	}
	s.counters.DroppedChat = cumulativeDroppedChat
	s.counters.EndedAt = at.UTC()
	s.counters.UsefulSeconds = at.Sub(s.started.Add(s.options.Window)).Seconds()
	if s.counters.UsefulSeconds < 0 {
		s.counters.UsefulSeconds = 0
	}
	if err := s.recorder.AppendSession(s.counters); err != nil {
		return err
	}
	s.closed = true
	return nil
}

func newID() (string, error) {
	var bytes [16]byte
	if _, err := rand.Read(bytes[:]); err != nil {
		return "", fmt.Errorf("generate id: %w", err)
	}
	return hex.EncodeToString(bytes[:]), nil
}

package core

import (
	"fmt"
	"time"

	"auto-clip/clipper/store"
)

type episodePoint struct {
	at                 time.Time
	targetAt           time.Time
	windowStart        time.Time
	windowEnd          time.Time
	streamOffsetSecond float64
	score              float32
	threshold          float32
	triggered          bool
	inEpisode          bool
	rawFeatures        []float32
	scaledFeatures     []float32
	messages           []store.Message
}

type openEpisode struct {
	id         string
	onset      episodePoint
	peak       episodePoint
	minimum    float32
	maximum    float32
	belowTicks int
}

type episodeTracker struct {
	options      Options
	sessionID    string
	recorder     Recorder
	open         *openEpisode
	previous     []episodePoint
	localPerHour map[int]int
}

func newEpisodeTracker(options Options, sessionID string, recorder Recorder) *episodeTracker {
	return &episodeTracker{
		options: options, sessionID: sessionID, recorder: recorder,
		localPerHour: make(map[int]int),
	}
}

func (t *episodeTracker) observe(point episodePoint) (string, error) {
	wasOpen := t.open != nil
	if t.open == nil && point.triggered {
		id, err := newID()
		if err != nil {
			return "", err
		}
		t.open = &openEpisode{
			id: id, onset: point, peak: point,
			minimum: point.score, maximum: point.score,
		}
	}
	point.inEpisode = wasOpen || t.open != nil

	if t.open != nil {
		current := t.open
		if wasOpen {
			if point.score > current.peak.score {
				current.peak = point
			}
			if point.score < current.minimum {
				current.minimum = point.score
			}
			if point.score > current.maximum {
				current.maximum = point.score
			}
		}
		if point.score < point.threshold {
			current.belowTicks++
		} else {
			current.belowTicks = 0
		}
		reason := ""
		if current.belowTicks >= t.options.EpisodeCloseBelowTicks {
			reason = "consecutive_below_threshold"
		} else if point.at.Sub(current.onset.at) >= t.options.EpisodeMaxDuration {
			reason = "maximum_duration"
		}
		if reason != "" {
			if err := t.appendEpisode(current, point.at, reason); err != nil {
				return "", err
			}
			t.open = nil
			if _, err := t.observeLocalMaximum(point); err != nil {
				return "", err
			}
			return "triggered", nil
		}
	}

	local, err := t.observeLocalMaximum(point)
	if err != nil {
		return "", err
	}
	if local {
		return "local_maximum", nil
	}
	return "", nil
}

func (t *episodeTracker) observeLocalMaximum(point episodePoint) (bool, error) {
	t.previous = append(t.previous, point)
	if len(t.previous) < 3 {
		return false, nil
	}
	if len(t.previous) > 3 {
		t.previous = t.previous[len(t.previous)-3:]
	}
	left, middle, right := t.previous[0], t.previous[1], t.previous[2]
	if middle.inEpisode || middle.score >= middle.threshold ||
		middle.score <= left.score || middle.score < right.score {
		return false, nil
	}
	usefulStart := t.options.ObservedAt.Add(t.options.Window)
	if middle.at.Before(usefulStart) {
		return false, nil
	}
	hour := int(middle.at.Sub(usefulStart) / time.Hour)
	if t.localPerHour[hour] >= t.options.LocalPeaksPerHour {
		return false, nil
	}
	id, err := newID()
	if err != nil {
		return false, err
	}
	episode := t.record(id, "local_maximum", middle, middle, middle.at, "local_maximum")
	if err := t.recorder.AppendEpisode(episode); err != nil {
		return false, fmt.Errorf("persist local maximum: %w", err)
	}
	t.localPerHour[hour]++
	return true, nil
}

func (t *episodeTracker) flush(at time.Time) (bool, error) {
	if t.open == nil {
		return false, nil
	}
	if err := t.appendEpisode(t.open, at, "session_closed"); err != nil {
		return false, err
	}
	t.open = nil
	return true, nil
}

func (t *episodeTracker) appendEpisode(open *openEpisode, closedAt time.Time, reason string) error {
	episode := t.record(open.id, "triggered", open.onset, open.peak, closedAt, reason)
	episode.MinimumScore = open.minimum
	episode.MaximumScore = open.maximum
	if err := t.recorder.AppendEpisode(episode); err != nil {
		return fmt.Errorf("persist episode: %w", err)
	}
	return nil
}

func (t *episodeTracker) record(
	id, recordType string,
	onset, peak episodePoint,
	closedAt time.Time,
	reason string,
) store.Episode {
	duration := closedAt.Sub(onset.at).Seconds()
	if duration < 0 {
		duration = 0
	}
	return store.Episode{
		SchemaVersion: store.LiveSchemaVersion,
		EpisodeID:     id, RecordType: recordType,
		SessionID: t.sessionID, Streamer: t.options.Streamer,
		BroadcasterID: t.options.BroadcasterID, StreamID: t.options.StreamID,
		ManifestSHA256: t.options.ManifestSHA256,
		OnsetAt:        onset.at, OnsetScore: onset.score,
		PeakAt: peak.at, PeakTargetAt: peak.targetAt,
		PeakWindowStart: peak.windowStart, PeakWindowEnd: peak.windowEnd,
		PeakStreamOffset: peak.streamOffsetSecond, PeakScore: peak.score,
		ClosedAt: closedAt.UTC(), DurationSeconds: duration,
		MinimumScore: onset.score, MaximumScore: peak.score,
		Threshold: onset.threshold, CloseReason: reason,
		RawFeatures:    append([]float32(nil), peak.rawFeatures...),
		ScaledFeatures: append([]float32(nil), peak.scaledFeatures...),
		Messages:       append([]store.Message(nil), peak.messages...),
	}
}

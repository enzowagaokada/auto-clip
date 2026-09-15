package core

import (
	"errors"
	"testing"
	"time"

	"auto-clip/clipper/store"
)

type flakyEpisodeRecorder struct {
	memoryRecorder
	failures int
}

func (r *flakyEpisodeRecorder) AppendEpisode(episode store.Episode) error {
	if r.failures > 0 {
		r.failures--
		return errors.New("temporary write failure")
	}
	return r.memoryRecorder.AppendEpisode(episode)
}

func trackerForTest(t *testing.T, recorder *memoryRecorder, start time.Time) *episodeTracker {
	t.Helper()
	return newEpisodeTracker(Options{
		Streamer: "example", ReviewPartition: "calibration",
		StreamID: "stream", ObservedAt: start,
		Window: 35 * time.Second, ManifestSHA256: "manifest",
		EpisodeCloseBelowTicks: 2, EpisodeMaxDuration: 60 * time.Second,
		LocalPeaksPerHour: 5,
	}, "session", recorder)
}

func point(at time.Time, score float32, triggered bool) episodePoint {
	return episodePoint{
		at: at, targetAt: at.Add(-30 * time.Second),
		windowStart: at.Add(-35 * time.Second), windowEnd: at,
		streamOffsetSecond: at.Sub(time.Unix(0, 0)).Seconds(),
		score:              score, threshold: 0.5, triggered: triggered,
		rawFeatures: []float32{score}, scaledFeatures: []float32{score},
		messages: []store.Message{{Time: at, Text: "chat"}},
	}
}

func TestEpisodeRetainsPeakAndClosesAfterTwoBelowTicks(t *testing.T) {
	start := time.Unix(100, 0).UTC()
	recorder := &memoryRecorder{}
	tracker := trackerForTest(t, recorder, start)
	for _, sample := range []episodePoint{
		point(start.Add(35*time.Second), 0.51, true),
		point(start.Add(40*time.Second), 0.83, false),
		point(start.Add(45*time.Second), 0.49, false),
		point(start.Add(50*time.Second), 0.48, false),
	} {
		if _, err := tracker.observe(sample); err != nil {
			t.Fatal(err)
		}
	}
	if len(recorder.episodes) != 1 {
		t.Fatalf("episodes = %d, want 1", len(recorder.episodes))
	}
	got := recorder.episodes[0]
	if got.PeakScore != 0.83 || got.OnsetScore != 0.51 ||
		got.CloseReason != "consecutive_below_threshold" {
		t.Fatalf("episode = %#v", got)
	}
	if got.Messages[0].Text != "chat" || len(got.RawFeatures) != 1 {
		t.Fatalf("peak window not retained: %#v", got)
	}
}

func TestEpisodeMaximumDurationAndSessionFlush(t *testing.T) {
	start := time.Unix(100, 0).UTC()
	recorder := &memoryRecorder{}
	tracker := trackerForTest(t, recorder, start)
	if _, err := tracker.observe(point(start.Add(35*time.Second), 0.6, true)); err != nil {
		t.Fatal(err)
	}
	if _, err := tracker.observe(point(start.Add(95*time.Second), 0.7, false)); err != nil {
		t.Fatal(err)
	}
	if recorder.episodes[0].CloseReason != "maximum_duration" {
		t.Fatalf("close reason = %q", recorder.episodes[0].CloseReason)
	}

	tracker = trackerForTest(t, recorder, start)
	if _, err := tracker.observe(point(start.Add(35*time.Second), 0.6, true)); err != nil {
		t.Fatal(err)
	}
	flushed, err := tracker.flush(start.Add(50 * time.Second))
	if err != nil || !flushed {
		t.Fatalf("flush = %v, %v", flushed, err)
	}
	if recorder.episodes[1].CloseReason != "session_closed" {
		t.Fatalf("close reason = %q", recorder.episodes[1].CloseReason)
	}
}

func TestEpisodeFlushRetriesAfterPersistenceError(t *testing.T) {
	start := time.Unix(100, 0).UTC()
	recorder := &flakyEpisodeRecorder{failures: 1}
	tracker := newEpisodeTracker(Options{
		Streamer: "example", ReviewPartition: "calibration",
		ObservedAt: start, Window: 35 * time.Second,
		EpisodeCloseBelowTicks: 2, EpisodeMaxDuration: 60 * time.Second,
		LocalPeaksPerHour: 5,
	}, "session", recorder)
	if _, err := tracker.observe(point(start.Add(35*time.Second), 0.6, true)); err != nil {
		t.Fatal(err)
	}
	if _, err := tracker.flush(start.Add(40 * time.Second)); err == nil {
		t.Fatal("flush error = nil, want temporary persistence error")
	}
	flushed, err := tracker.flush(start.Add(41 * time.Second))
	if err != nil || !flushed || len(recorder.episodes) != 1 {
		t.Fatalf("retry flush = %v, %v, episodes=%d", flushed, err, len(recorder.episodes))
	}
}

func TestBelowThresholdLocalMaximumLimit(t *testing.T) {
	start := time.Unix(100, 0).UTC()
	recorder := &memoryRecorder{}
	tracker := trackerForTest(t, recorder, start)
	at := start.Add(35 * time.Second)
	for i := 0; i < 7; i++ {
		for _, score := range []float32{0.1, 0.4, 0.2} {
			if _, err := tracker.observe(point(at, score, false)); err != nil {
				t.Fatal(err)
			}
			at = at.Add(time.Second)
		}
	}
	if len(recorder.episodes) != 5 {
		t.Fatalf("local maxima = %d, want capped 5", len(recorder.episodes))
	}
	for _, episode := range recorder.episodes {
		if episode.RecordType != "local_maximum" {
			t.Fatalf("record type = %q", episode.RecordType)
		}
	}
}

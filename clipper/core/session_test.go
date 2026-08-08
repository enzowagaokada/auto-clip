package core

import (
	"testing"
	"time"

	"auto-clip/clipper/detection"
	"auto-clip/clipper/modelmeta"
	"auto-clip/clipper/preprocess"
	"auto-clip/clipper/store"
)

type fixedScorer struct {
	score float32
}

func (s fixedScorer) Score(preprocess.Encoded) (float32, error) {
	return s.score, nil
}

type memoryRecorder struct {
	candidate *store.Candidate
}

func (r *memoryRecorder) AppendCandidate(candidate store.Candidate) error {
	r.candidate = &candidate
	return nil
}

func (*memoryRecorder) AppendSession(store.SessionCounters) error {
	return nil
}

func TestEvaluateUsesThirtySecondTargetLag(t *testing.T) {
	bundle := &modelmeta.Bundle{
		Vocabulary: map[string]int{
			modelmeta.PAD: 0,
			modelmeta.UNK: 1,
			modelmeta.SEP: 2,
		},
		Metadata: modelmeta.Metadata{
			VocabSize:                3,
			MaxSequenceLength:        8,
			NumberOfFeatures:         len(modelmeta.FeatureNames),
			FeatureNames:             append([]string(nil), modelmeta.FeatureNames...),
			FeatureMean:              make([]float32, len(modelmeta.FeatureNames)),
			FeatureStandardDeviation: []float32{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
			WindowSeconds:            35,
			TargetLagSeconds:         30,
			WindowGeometry:           "clip_start_minus_5_plus_30",
			WindowGeometryVersion:    2,
			StreamTimeScaleSeconds:   100,
			Threshold:                0.5,
		},
	}
	encoder, err := preprocess.New(bundle)
	if err != nil {
		t.Fatal(err)
	}
	machine, err := detection.New(0.5, 0)
	if err != nil {
		t.Fatal(err)
	}
	recorder := &memoryRecorder{}
	streamStarted := time.Unix(1_000, 0).UTC()
	session, err := NewSession(Options{
		Streamer:      "example",
		StreamID:      "stream",
		StreamStarted: streamStarted,
		ObservedAt:    streamStarted,
		Window:        35 * time.Second,
		TargetLag:     30 * time.Second,
	}, encoder, fixedScorer{score: 0.75}, machine, recorder)
	if err != nil {
		t.Fatal(err)
	}

	at := streamStarted.Add(100 * time.Second)
	for _, message := range []preprocess.Message{
		{Time: at.Add(-36 * time.Second), User: "old", Text: "old"},
		{Time: at.Add(-35 * time.Second), User: "boundary", Text: "boundary"},
		{Time: at, User: "now", Text: "now"},
	} {
		if err := session.AddMessage(message); err != nil {
			t.Fatal(err)
		}
	}

	evaluation, err := session.EvaluateDetailed(at)
	if err != nil {
		t.Fatal(err)
	}
	wantTarget := at.Add(-30 * time.Second)
	if !evaluation.TargetAt.Equal(wantTarget) {
		t.Fatalf("TargetAt = %s, want %s", evaluation.TargetAt, wantTarget)
	}
	if evaluation.Candidate == nil || recorder.candidate == nil {
		t.Fatal("expected threshold crossing candidate")
	}
	if evaluation.Candidate.StreamOffsetSecond != 70 {
		t.Fatalf(
			"StreamOffsetSecond = %v, want 70",
			evaluation.Candidate.StreamOffsetSecond,
		)
	}
	if evaluation.Candidate.MessageCount != 2 {
		t.Fatalf("MessageCount = %d, want 2", evaluation.Candidate.MessageCount)
	}
}

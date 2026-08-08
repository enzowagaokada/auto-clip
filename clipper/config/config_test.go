package config

import (
	"os"
	"path/filepath"
	"testing"
)

func TestLoadRootYAMLUsesDefaultsAndStreamerOverrides(t *testing.T) {
	root := t.TempDir()
	path := filepath.Join(root, "config.yaml")
	data := []byte(`
twitch:
  streamers:
    - name: example
      broadcaster_id: "123"
      active: true
      clip_threshold: 0.8
clipper:
  window_seconds: 35
  target_lag_seconds: 30
`)
	if err := os.WriteFile(path, data, 0o644); err != nil {
		t.Fatal(err)
	}
	cfg, err := Load(path, root)
	if err != nil {
		t.Fatal(err)
	}
	streamer := cfg.ActiveStreamers()[0]
	if got := cfg.ThresholdFor(streamer, 0.57); got != 0.8 {
		t.Fatalf("threshold = %v, want 0.8", got)
	}
	if cfg.Clipper.BundleDir != "models/exports/window-v2-vod-seed0" {
		t.Fatalf("bundle_dir = %q", cfg.Clipper.BundleDir)
	}
	if cfg.Clipper.CandidatesPath != "data/live/shadow/window-v2/candidates.jsonl" {
		t.Fatalf("candidates_path = %q", cfg.Clipper.CandidatesPath)
	}
	if cfg.Clipper.CandidatesReviewPath != "data/live/shadow/window-v2/candidates_review.jsonl" {
		t.Fatalf("candidates_review_path = %q", cfg.Clipper.CandidatesReviewPath)
	}
	if cfg.Clipper.CandidatesReviewCSVPath != "data/live/shadow/window-v2/candidates_review.csv" {
		t.Fatalf("candidates_review_csv_path = %q", cfg.Clipper.CandidatesReviewCSVPath)
	}
}

func TestValidateRejectsChangedLiveParity(t *testing.T) {
	cfg := Defaults(t.TempDir())
	cfg.Twitch.Streamers = []Streamer{{Name: "example", BroadcasterID: "123", Active: true}}
	cfg.Clipper.TargetLagSeconds = 29
	if err := cfg.Validate(); err == nil {
		t.Fatal("Validate() error = nil, want immutable target lag error")
	}
}

func TestValidateRejectsNonShadowMode(t *testing.T) {
	cfg := Defaults(t.TempDir())
	cfg.Twitch.Streamers = []Streamer{{Name: "example", BroadcasterID: "123", Active: true}}
	cfg.Clipper.Mode = "live"
	if err := cfg.Validate(); err == nil {
		t.Fatal("Validate() error = nil, want shadow-only mode error")
	}
}

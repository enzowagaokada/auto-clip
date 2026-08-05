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
  target_lag_seconds: 5
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
	if cfg.Clipper.BundleDir != "models/exports/reviewed-vod-seed0" {
		t.Fatalf("bundle_dir = %q", cfg.Clipper.BundleDir)
	}
}

func TestValidateRejectsChangedLiveParity(t *testing.T) {
	cfg := Defaults(t.TempDir())
	cfg.Twitch.Streamers = []Streamer{{Name: "example", BroadcasterID: "123", Active: true}}
	cfg.Clipper.TargetLagSeconds = 4
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

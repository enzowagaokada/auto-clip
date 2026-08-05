package config

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"time"

	"gopkg.in/yaml.v3"
)

type Config struct {
	RepoRoot       string        `yaml:"-"`
	Twitch         Twitch        `yaml:"twitch"`
	Clipper        Clipper       `yaml:"clipper"`
	InferenceEvery time.Duration `yaml:"-"`
	PollEvery      time.Duration `yaml:"-"`
}

type Twitch struct {
	Streamers []Streamer `yaml:"streamers"`
}

type Streamer struct {
	Name            string   `yaml:"name"`
	BroadcasterID   string   `yaml:"broadcaster_id"`
	Active          bool     `yaml:"active"`
	Threshold       *float32 `yaml:"clip_threshold,omitempty"`
	CooldownSeconds *int     `yaml:"cooldown_seconds,omitempty"`
}

type Clipper struct {
	Mode              string   `yaml:"mode"`
	BundleDir         string   `yaml:"bundle_dir"`
	RuntimeDLLPath    string   `yaml:"runtime_dll_path"`
	Threshold         *float32 `yaml:"clip_threshold,omitempty"`
	CooldownSeconds   int      `yaml:"cooldown_seconds"`
	WindowSeconds     int      `yaml:"window_seconds"`
	TargetLagSeconds  int      `yaml:"target_lag_seconds"`
	InferenceMillis   int      `yaml:"inference_millis"`
	StreamPollSeconds int      `yaml:"stream_poll_seconds"`
	ChatBufferSize    int      `yaml:"chat_buffer_size"`
	CandidatesPath    string   `yaml:"candidates_path"`
	SessionsPath      string   `yaml:"sessions_path"`
	TokensInputName   string   `yaml:"tokens_input_name"`
	FeaturesInputName string   `yaml:"features_input_name"`
	OutputName        string   `yaml:"output_name"`
	OutputIsLogit     bool     `yaml:"output_is_logit"`
}

func Defaults(repoRoot string) Config {
	return Config{
		RepoRoot: repoRoot,
		Clipper: Clipper{
			Mode:              "shadow",
			BundleDir:         "models/exports/reviewed-vod-seed0",
			RuntimeDLLPath:    "clipper/runtime/onnxruntime.dll",
			CooldownSeconds:   75,
			WindowSeconds:     35,
			TargetLagSeconds:  5,
			InferenceMillis:   2500,
			StreamPollSeconds: 30,
			ChatBufferSize:    4096,
			CandidatesPath:    "data/live/shadow/candidates.jsonl",
			SessionsPath:      "data/live/shadow/sessions.jsonl",
			TokensInputName:   "tokens",
			FeaturesInputName: "features",
			OutputName:        "logits",
			OutputIsLogit:     true,
		},
	}
}

func Load(path, repoRoot string) (Config, error) {
	cfg := Defaults(repoRoot)
	data, err := os.ReadFile(path)
	if err != nil {
		return Config{}, fmt.Errorf("read root config: %w", err)
	}
	if err := yaml.Unmarshal(data, &cfg); err != nil {
		return Config{}, fmt.Errorf("decode root config YAML: %w", err)
	}
	cfg.RepoRoot = repoRoot
	if err := cfg.Validate(); err != nil {
		return Config{}, err
	}
	cfg.InferenceEvery = time.Duration(cfg.Clipper.InferenceMillis) * time.Millisecond
	cfg.PollEvery = time.Duration(cfg.Clipper.StreamPollSeconds) * time.Second
	return cfg, nil
}

func (c Config) Validate() error {
	if c.RepoRoot == "" {
		return errors.New("repository root is required")
	}
	if c.Clipper.BundleDir == "" {
		return errors.New("clipper.bundle_dir is required")
	}
	if c.Clipper.Mode != "shadow" {
		return errors.New(`clipper.mode must be "shadow"; public clip creation is not implemented`)
	}
	if c.Clipper.RuntimeDLLPath == "" {
		return errors.New("clipper.runtime_dll_path is required")
	}
	if c.Clipper.TokensInputName == "" || c.Clipper.FeaturesInputName == "" || c.Clipper.OutputName == "" {
		return errors.New("ONNX input and output names are required")
	}
	if !c.Clipper.OutputIsLogit {
		return errors.New("clipper.output_is_logit must be true for the exported model contract")
	}
	if c.Clipper.Threshold != nil && (*c.Clipper.Threshold < 0 || *c.Clipper.Threshold > 1) {
		return errors.New("threshold must be in [0, 1]")
	}
	if c.Clipper.WindowSeconds != 35 {
		return errors.New("clipper.window_seconds is immutable and must be 35")
	}
	if c.Clipper.TargetLagSeconds != 5 {
		return errors.New("clipper.target_lag_seconds is immutable and must be 5")
	}
	if c.Clipper.CooldownSeconds < 0 || c.Clipper.InferenceMillis <= 0 ||
		c.Clipper.StreamPollSeconds <= 0 || c.Clipper.ChatBufferSize <= 0 {
		return errors.New("clipper intervals and chat_buffer_size must be positive; cooldown must be non-negative")
	}
	active := 0
	ids := make(map[string]struct{})
	for i, streamer := range c.Twitch.Streamers {
		if !streamer.Active {
			continue
		}
		active++
		if streamer.Name == "" || streamer.BroadcasterID == "" {
			return fmt.Errorf("active twitch.streamers[%d] requires name and broadcaster_id", i)
		}
		if _, exists := ids[streamer.BroadcasterID]; exists {
			return fmt.Errorf("duplicate active broadcaster_id %q", streamer.BroadcasterID)
		}
		ids[streamer.BroadcasterID] = struct{}{}
		if streamer.Threshold != nil && (*streamer.Threshold < 0 || *streamer.Threshold > 1) {
			return fmt.Errorf("streamer %q clip_threshold must be in [0, 1]", streamer.Name)
		}
		if streamer.CooldownSeconds != nil && *streamer.CooldownSeconds < 0 {
			return fmt.Errorf("streamer %q cooldown_seconds must be non-negative", streamer.Name)
		}
	}
	if active == 0 {
		return errors.New("at least one twitch.streamers entry must be active")
	}
	return nil
}

// Resolve makes a configured bundle/output path absolute relative to RepoRoot.
func (c Config) Resolve(path string) string {
	if filepath.IsAbs(path) {
		return filepath.Clean(path)
	}
	return filepath.Join(c.RepoRoot, filepath.FromSlash(path))
}

func (c Config) ActiveStreamers() []Streamer {
	result := make([]Streamer, 0, len(c.Twitch.Streamers))
	for _, streamer := range c.Twitch.Streamers {
		if streamer.Active {
			result = append(result, streamer)
		}
	}
	return result
}

func (c Config) ThresholdFor(streamer Streamer, metadataDefault float32) float32 {
	if streamer.Threshold != nil {
		return *streamer.Threshold
	}
	if c.Clipper.Threshold != nil {
		return *c.Clipper.Threshold
	}
	return metadataDefault
}

func (c Config) CooldownFor(streamer Streamer) time.Duration {
	seconds := c.Clipper.CooldownSeconds
	if streamer.CooldownSeconds != nil {
		seconds = *streamer.CooldownSeconds
	}
	return time.Duration(seconds) * time.Second
}

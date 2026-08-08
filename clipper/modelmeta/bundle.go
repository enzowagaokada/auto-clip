package modelmeta

import (
	"crypto/sha256"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
)

const (
	PAD = "[PAD]"
	UNK = "[UNK]"
	SEP = "[SEP]"
)

var FeatureNames = []string{
	"messages_per_second",
	"unique_users",
	"normalized_stream_time",
	"message_rate_0_5s",
	"message_rate_5_10s",
	"message_rate_10_15s",
	"message_rate_15_20s",
	"message_rate_20_25s",
	"message_rate_25_30s",
	"message_rate_30_35s",
	"message_rate_change",
	"peak_5s_rate",
	"repeat_message_ratio",
}

type Metadata struct {
	VocabSize                int       `json:"vocab_size"`
	MaxSequenceLength        int       `json:"max_seq_len"`
	NumberOfFeatures         int       `json:"num_features"`
	FeatureNames             []string  `json:"feature_names"`
	FeatureMean              []float32 `json:"feature_mean"`
	FeatureStandardDeviation []float32 `json:"feature_std"`
	WindowSeconds            int       `json:"window_seconds"`
	TargetLagSeconds         int       `json:"target_lag_seconds"`
	WindowGeometry           string    `json:"window_geometry"`
	WindowGeometryVersion    int       `json:"window_geometry_version"`
	StreamTimeScaleSeconds   float64   `json:"stream_time_scale_seconds"`
	Threshold                float32   `json:"threshold"`
}

type Bundle struct {
	Vocabulary       map[string]int
	Metadata         Metadata
	Manifest         Manifest
	ManifestChecksum string
}

type Manifest struct {
	SchemaVersion int               `json:"schema_version"`
	Artifacts     map[string]string `json:"artifacts"`
}

const (
	ModelFilename    = "chat_classifier.onnx"
	VocabFilename    = "vocab.json"
	MetadataFilename = "inference_meta.json"
	ManifestFilename = "manifest.json"
)

// Load verifies the export manifest and all deployable artifacts before
// decoding vocabulary or metadata.
func Load(bundleDir string) (*Bundle, error) {
	manifestPath := filepath.Join(bundleDir, ManifestFilename)
	var manifest Manifest
	if err := decode(manifestPath, &manifest); err != nil {
		return nil, fmt.Errorf("load manifest: %w", err)
	}
	if manifest.SchemaVersion != 1 {
		return nil, fmt.Errorf("unsupported manifest schema_version %d", manifest.SchemaVersion)
	}
	expected := []string{ModelFilename, VocabFilename, MetadataFilename}
	if len(manifest.Artifacts) != len(expected) {
		return nil, errors.New("manifest artifacts must contain exactly ONNX, vocab, and metadata")
	}
	for _, name := range expected {
		want, ok := manifest.Artifacts[name]
		if !ok || len(want) != sha256.Size*2 {
			return nil, fmt.Errorf("manifest is missing a valid SHA-256 for %s", name)
		}
		got, err := fileSHA256(filepath.Join(bundleDir, name))
		if err != nil {
			return nil, fmt.Errorf("checksum %s: %w", name, err)
		}
		if got != want {
			return nil, fmt.Errorf("checksum mismatch for %s: expected %s, got %s", name, want, got)
		}
	}
	manifestChecksum, err := fileSHA256(manifestPath)
	if err != nil {
		return nil, fmt.Errorf("checksum manifest: %w", err)
	}

	var vocab map[string]int
	if err := decode(filepath.Join(bundleDir, VocabFilename), &vocab); err != nil {
		return nil, fmt.Errorf("load vocabulary: %w", err)
	}
	var meta Metadata
	if err := decode(filepath.Join(bundleDir, MetadataFilename), &meta); err != nil {
		return nil, fmt.Errorf("load inference metadata: %w", err)
	}
	bundle := &Bundle{
		Vocabulary: vocab, Metadata: meta, Manifest: manifest,
		ManifestChecksum: manifestChecksum,
	}
	if err := bundle.Validate(); err != nil {
		return nil, err
	}
	return bundle, nil
}

func (b *Bundle) ModelPath(bundleDir string) string {
	return filepath.Join(bundleDir, ModelFilename)
}

func fileSHA256(path string) (string, error) {
	file, err := os.Open(path)
	if err != nil {
		return "", err
	}
	defer file.Close()
	digest := sha256.New()
	if _, err := io.Copy(digest, file); err != nil {
		return "", err
	}
	return fmt.Sprintf("%x", digest.Sum(nil)), nil
}

func decode(path string, value any) error {
	data, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	if err := json.Unmarshal(data, value); err != nil {
		return err
	}
	return nil
}

func (b *Bundle) Validate() error {
	if b.Vocabulary[PAD] != 0 || b.Vocabulary[UNK] != 1 || b.Vocabulary[SEP] != 2 {
		return errors.New("vocabulary must reserve [PAD]=0, [UNK]=1, [SEP]=2")
	}
	m := b.Metadata
	if m.MaxSequenceLength <= 0 {
		return errors.New("max_seq_len must be positive")
	}
	if m.NumberOfFeatures != len(FeatureNames) ||
		len(m.FeatureMean) != len(FeatureNames) ||
		len(m.FeatureStandardDeviation) != len(FeatureNames) ||
		len(m.FeatureNames) != len(FeatureNames) {
		return fmt.Errorf("metadata must contain exactly %d features", len(FeatureNames))
	}
	for i, expected := range FeatureNames {
		if m.FeatureNames[i] != expected {
			return fmt.Errorf("feature %d is %q, expected %q", i, m.FeatureNames[i], expected)
		}
		if m.FeatureStandardDeviation[i] == 0 {
			return fmt.Errorf("feature_std[%d] is zero", i)
		}
	}
	if m.StreamTimeScaleSeconds <= 0 {
		return errors.New("stream_time_scale_seconds must be positive")
	}
	if m.WindowSeconds != 35 ||
		m.TargetLagSeconds != 30 ||
		m.WindowGeometry != "clip_start_minus_5_plus_30" ||
		m.WindowGeometryVersion != 2 {
		return errors.New(
			"model metadata must use the 35-second " +
				"[clip start - 5s, clip start + 30s] window contract",
		)
	}
	if m.Threshold < 0 || m.Threshold > 1 {
		return errors.New("metadata threshold must be in [0, 1]")
	}
	if m.VocabSize > 0 && len(b.Vocabulary) != m.VocabSize {
		return fmt.Errorf("vocabulary has %d entries, metadata expects %d", len(b.Vocabulary), m.VocabSize)
	}
	ids := make([]bool, len(b.Vocabulary))
	for token, id := range b.Vocabulary {
		if id < 0 || id >= len(ids) || ids[id] {
			return fmt.Errorf("vocabulary token %q has invalid or duplicate id %d", token, id)
		}
		ids[id] = true
	}
	return nil
}

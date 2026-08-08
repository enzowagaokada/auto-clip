package modelmeta

import "testing"

func testBundle() *Bundle {
	return &Bundle{
		Vocabulary: map[string]int{PAD: 0, UNK: 1, SEP: 2},
		Metadata: Metadata{
			VocabSize:                3,
			MaxSequenceLength:        8,
			NumberOfFeatures:         len(FeatureNames),
			FeatureNames:             append([]string(nil), FeatureNames...),
			FeatureMean:              make([]float32, len(FeatureNames)),
			FeatureStandardDeviation: []float32{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
			WindowSeconds:            35,
			TargetLagSeconds:         30,
			WindowGeometry:           "clip_start_minus_5_plus_30",
			WindowGeometryVersion:    2,
			StreamTimeScaleSeconds:   43200,
			Threshold:                0.57,
		},
	}
}

func TestValidateAcceptsCurrentWindowGeometry(t *testing.T) {
	if err := testBundle().Validate(); err != nil {
		t.Fatalf("Validate() error = %v", err)
	}
}

func TestValidateRejectsLegacyWindowGeometry(t *testing.T) {
	bundle := testBundle()
	bundle.Metadata.TargetLagSeconds = 5
	bundle.Metadata.WindowGeometry = ""
	bundle.Metadata.WindowGeometryVersion = 0
	if err := bundle.Validate(); err == nil {
		t.Fatal("Validate() error = nil, want stale geometry error")
	}
}

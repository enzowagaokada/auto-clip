package preprocess

import (
	"encoding/json"
	"math"
	"os"
	"reflect"
	"testing"
	"time"

	"auto-clip/clipper/modelmeta"
)

type golden struct {
	Tokens      []int32   `json:"tokens"`
	RawFeatures []float32 `json:"raw_features"`
}

func TestPythonGolden(t *testing.T) {
	data, err := os.ReadFile("testdata/python_golden.json")
	if err != nil {
		t.Fatal(err)
	}
	var want golden
	if err := json.Unmarshal(data, &want); err != nil {
		t.Fatal(err)
	}
	bundle := &modelmeta.Bundle{
		Vocabulary: map[string]int{
			"[PAD]": 0, "[UNK]": 1, "[SEP]": 2,
			"Hello": 3, "HELLO": 4, "world": 5,
		},
		Metadata: modelmeta.Metadata{
			VocabSize: 6, MaxSequenceLength: 8, NumberOfFeatures: 13,
			FeatureNames:             modelmeta.FeatureNames,
			FeatureMean:              make([]float32, 13),
			FeatureStandardDeviation: []float32{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
			StreamTimeScaleSeconds:   43200,
		},
	}
	encoder, err := New(bundle)
	if err != nil {
		t.Fatal(err)
	}
	start := time.Date(2026, 1, 2, 3, 4, 5, 0, time.UTC)
	got, err := encoder.Encode(Input{
		WindowStart: start, WindowEnd: start.Add(35 * time.Second),
		StreamOffsetSecond: 21600,
		Messages: []Message{
			{Time: start.Add(time.Second), User: "a", Text: "Hello world"},
			{Time: start.Add(6 * time.Second), User: "b", Text: "HELLO"},
			{Time: start.Add(31 * time.Second), User: "a", Text: "hello"},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(got.Tokens, want.Tokens) {
		t.Fatalf("tokens = %v, want %v", got.Tokens, want.Tokens)
	}
	for i := range want.RawFeatures {
		if math.Abs(float64(got.RawFeatures[i]-want.RawFeatures[i])) > 1e-6 {
			t.Errorf("raw feature %d = %v, want %v", i, got.RawFeatures[i], want.RawFeatures[i])
		}
		if math.Abs(float64(got.Features[i]-want.RawFeatures[i])) > 1e-6 {
			t.Errorf("scaled feature %d = %v, want %v", i, got.Features[i], want.RawFeatures[i])
		}
	}
}

func TestRecentTokenTruncationPreservesCase(t *testing.T) {
	bundle := &modelmeta.Bundle{
		Vocabulary: map[string]int{"[PAD]": 0, "[UNK]": 1, "[SEP]": 2, "A": 3, "a": 4},
		Metadata: modelmeta.Metadata{
			VocabSize: 5, MaxSequenceLength: 3, NumberOfFeatures: 13,
			FeatureNames: modelmeta.FeatureNames, FeatureMean: make([]float32, 13),
			FeatureStandardDeviation: []float32{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
			StreamTimeScaleSeconds:   1,
		},
	}
	encoder, _ := New(bundle)
	start := time.Now()
	got, err := encoder.Encode(Input{
		WindowStart: start, WindowEnd: start.Add(time.Second),
		Messages: []Message{{Time: start, Text: "A A"}, {Time: start, Text: "a"}},
	})
	if err != nil {
		t.Fatal(err)
	}
	if want := []int32{3, 2, 4}; !reflect.DeepEqual(got.Tokens, want) {
		t.Fatalf("tokens = %v, want %v", got.Tokens, want)
	}
}

func TestTokenizeMatchesPythonControlWhitespace(t *testing.T) {
	got := Tokenize("one\x1ctwo\u202fthree")
	want := []string{"one", "two", "three"}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("Tokenize() = %v, want %v", got, want)
	}
}

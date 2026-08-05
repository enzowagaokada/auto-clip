package preprocess

import (
	"errors"
	"math"
	"sort"
	"strings"
	"time"

	"auto-clip/clipper/modelmeta"
	"golang.org/x/text/cases"
)

const (
	BucketSeconds = 5
	BucketCount   = 7
)

type Message struct {
	Time time.Time `json:"time"`
	User string    `json:"user"`
	Text string    `json:"text"`
}

type Input struct {
	Messages           []Message
	WindowStart        time.Time
	WindowEnd          time.Time
	StreamOffsetSecond float64
}

type Encoded struct {
	Tokens      []int32
	RawFeatures []float32
	Features    []float32
}

type Encoder struct {
	bundle *modelmeta.Bundle
}

func New(bundle *modelmeta.Bundle) (*Encoder, error) {
	if bundle == nil {
		return nil, errors.New("model bundle is nil")
	}
	if err := bundle.Validate(); err != nil {
		return nil, err
	}
	return &Encoder{bundle: bundle}, nil
}

// Tokenize matches Python str.split(): Python whitespace, no lowercasing.
func Tokenize(text string) []string {
	return strings.FieldsFunc(text, pythonWhitespace)
}

func pythonWhitespace(r rune) bool {
	switch {
	case r >= '\t' && r <= '\r':
		return true
	case r >= '\x1c' && r <= '\x1f':
		return true
	case r == ' ', r == '\x85', r == '\xa0', r == '\u1680',
		r == '\u2028', r == '\u2029', r == '\u202f', r == '\u205f',
		r == '\u3000':
		return true
	case r >= '\u2000' && r <= '\u200a':
		return true
	default:
		return false
	}
}

func (e *Encoder) Encode(input Input) (Encoded, error) {
	if input.WindowEnd.Before(input.WindowStart) {
		return Encoded{}, errors.New("window end precedes window start")
	}
	messages := append([]Message(nil), input.Messages...)
	sort.SliceStable(messages, func(i, j int) bool {
		return messages[i].Time.Before(messages[j].Time)
	})
	tokens := e.encodeTokens(messages)
	raw := rawFeatures(messages, input.WindowStart, input.WindowEnd,
		input.StreamOffsetSecond, e.bundle.Metadata.StreamTimeScaleSeconds)
	scaled := make([]float32, len(raw))
	for i := range raw {
		scaled[i] = (raw[i] - e.bundle.Metadata.FeatureMean[i]) /
			e.bundle.Metadata.FeatureStandardDeviation[i]
	}
	return Encoded{Tokens: tokens, RawFeatures: raw, Features: scaled}, nil
}

func (e *Encoder) encodeTokens(messages []Message) []int32 {
	vocab := e.bundle.Vocabulary
	ids := make([]int32, 0, e.bundle.Metadata.MaxSequenceLength)
	for i, message := range messages {
		if i > 0 {
			ids = append(ids, int32(vocab[modelmeta.SEP]))
		}
		for _, token := range Tokenize(message.Text) {
			id, ok := vocab[token]
			if !ok {
				id = vocab[modelmeta.UNK]
			}
			ids = append(ids, int32(id))
		}
	}
	maxLength := e.bundle.Metadata.MaxSequenceLength
	if len(ids) > maxLength {
		return append([]int32(nil), ids[len(ids)-maxLength:]...)
	}
	result := make([]int32, maxLength)
	padding := maxLength - len(ids)
	for i := 0; i < padding; i++ {
		result[i] = int32(vocab[modelmeta.PAD])
	}
	copy(result[padding:], ids)
	return result
}

func rawFeatures(messages []Message, start, end time.Time, streamOffset, streamScale float64) []float32 {
	duration := end.Sub(start).Seconds()
	if duration < 1 {
		duration = 1
	}
	bucketCounts := [BucketCount]int{}
	users := make(map[string]struct{})
	normalized := make([]string, 0, len(messages))
	folder := cases.Fold()

	for _, message := range messages {
		if message.User != "" {
			users[message.User] = struct{}{}
		}
		relative := message.Time.Sub(start).Seconds()
		if relative < 0 {
			relative = 0
		}
		bucket := int(math.Floor(relative / BucketSeconds))
		if bucket >= BucketCount {
			bucket = BucketCount - 1
		}
		bucketCounts[bucket]++
		text := strings.TrimSpace(message.Text)
		if text != "" {
			normalized = append(normalized, folder.String(text))
		}
	}

	bucketRates := make([]float64, BucketCount)
	for i, count := range bucketCounts {
		bucketRates[i] = round4(float64(count) / BucketSeconds)
	}
	early := (bucketRates[0] + bucketRates[1]) / 2
	recent := (bucketRates[BucketCount-2] + bucketRates[BucketCount-1]) / 2
	repeats := 0.0
	if len(normalized) > 0 {
		unique := make(map[string]struct{}, len(normalized))
		for _, message := range normalized {
			unique[message] = struct{}{}
		}
		repeats = 1 - float64(len(unique))/float64(len(normalized))
	}
	normalizedTime := streamOffset / math.Max(1, streamScale)
	normalizedTime = math.Max(0, math.Min(1, normalizedTime))

	raw := []float32{
		float32(round4(float64(len(messages)) / duration)),
		float32(len(users)),
		float32(normalizedTime),
	}
	for _, rate := range bucketRates {
		raw = append(raw, float32(rate))
	}
	raw = append(raw,
		float32(round4(recent-early)),
		float32(maxRate(bucketRates)),
		float32(round4(repeats)),
	)
	return raw
}

func maxRate(values []float64) float64 {
	result := 0.0
	for _, value := range values {
		if value > result {
			result = value
		}
	}
	return result
}

// Python round() and NumPy use ties-to-even; the historical feature builder
// rounds derived rates to four decimal places.
func round4(value float64) float64 {
	return math.RoundToEven(value*10000) / 10000
}

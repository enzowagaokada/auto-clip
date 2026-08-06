package replay

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strconv"
	"time"

	"auto-clip/clipper/core"
	"auto-clip/clipper/detection"
	"auto-clip/clipper/preprocess"
	"auto-clip/clipper/store"
)

type stringOrNumber string

func (value *stringOrNumber) UnmarshalJSON(data []byte) error {
	var text string
	if err := json.Unmarshal(data, &text); err == nil {
		*value = stringOrNumber(text)
		return nil
	}

	var number json.Number
	if err := json.Unmarshal(data, &number); err != nil {
		return errors.New("must be a string or integer")
	}
	if _, err := strconv.ParseUint(number.String(), 10, 64); err != nil {
		return fmt.Errorf("must be an unsigned integer: %w", err)
	}
	*value = stringOrNumber(number.String())
	return nil
}

type RawWindow struct {
	StreamerName string         `json:"streamer_name"`
	ClipID       string         `json:"clip_id"`
	VODID        stringOrNumber `json:"vod_id"`
	TargetOffset float64        `json:"target_offset"`
	WindowStart  float64        `json:"window_start"`
	WindowEnd    float64        `json:"window_end"`
	Messages     []RawMessage   `json:"messages"`
}

type RawMessage struct {
	OffsetSeconds float64 `json:"offset_seconds"`
	CreatedAt     string  `json:"created_at"`
	User          string  `json:"user"`
	Message       string  `json:"message"`
}

type Result struct {
	Path         string  `json:"path"`
	Streamer     string  `json:"streamer"`
	VODID        string  `json:"vod_id"`
	ClipID       string  `json:"clip_id,omitempty"`
	TargetOffset float64 `json:"target_offset"`
	Score        float32 `json:"score"`
	Threshold    float32 `json:"threshold"`
	Candidate    bool    `json:"candidate"`
}

type Options struct {
	Encoder        *preprocess.Encoder
	Scorer         core.Scorer
	Threshold      float32
	Cooldown       time.Duration
	ManifestSHA256 string
	Output         io.Writer
}

type discardRecorder struct{}

func (discardRecorder) AppendCandidate(store.Candidate) error     { return nil }
func (discardRecorder) AppendSession(store.SessionCounters) error { return nil }

func Run(paths []string, options Options) error {
	if len(paths) == 0 {
		return errors.New("replay requires at least one input file")
	}
	if options.Encoder == nil || options.Scorer == nil || options.Output == nil {
		return errors.New("replay encoder, scorer, and output are required")
	}
	encoder := json.NewEncoder(options.Output)
	encoder.SetEscapeHTML(false)
	for _, path := range paths {
		result, err := runFile(path, options)
		if err != nil {
			return err
		}
		if err := encoder.Encode(result); err != nil {
			return fmt.Errorf("write replay result: %w", err)
		}
	}
	return nil
}

func runFile(path string, options Options) (Result, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return Result{}, fmt.Errorf("read replay %s: %w", path, err)
	}
	var raw RawWindow
	if err := json.Unmarshal(data, &raw); err != nil {
		return Result{}, fmt.Errorf("decode replay %s: %w", path, err)
	}
	if raw.WindowEnd-raw.WindowStart != 35 ||
		raw.WindowEnd-raw.TargetOffset != 5 ||
		raw.TargetOffset-raw.WindowStart != 30 {
		return Result{}, fmt.Errorf("%s does not contain the immutable [target-30s, target+5s] window", path)
	}

	base := time.Unix(0, 0).UTC()
	streamer := raw.StreamerName
	if streamer == "" {
		streamer = filepath.Base(path)
	}
	vodID := string(raw.VODID)
	machine, err := detection.New(options.Threshold, options.Cooldown)
	if err != nil {
		return Result{}, err
	}
	session, err := core.NewSession(core.Options{
		Streamer: streamer, StreamID: vodID, StreamStarted: base,
		ObservedAt: base.Add(time.Duration(raw.WindowStart * float64(time.Second))),
		Window:     35 * time.Second, TargetLag: 5 * time.Second,
		ManifestSHA256: options.ManifestSHA256,
	}, options.Encoder, options.Scorer, machine, discardRecorder{})
	if err != nil {
		return Result{}, fmt.Errorf("create replay session for %s: %w", path, err)
	}
	for _, message := range raw.Messages {
		if err := session.AddMessage(preprocess.Message{
			Time: base.Add(time.Duration(message.OffsetSeconds * float64(time.Second))),
			User: message.User, Text: message.Message,
		}); err != nil {
			return Result{}, err
		}
	}
	at := base.Add(time.Duration(raw.WindowEnd * float64(time.Second)))
	evaluation, err := session.EvaluateDetailed(at)
	if err != nil {
		return Result{}, fmt.Errorf("evaluate replay %s: %w", path, err)
	}
	_ = session.Close(at)
	return Result{
		Path: path, Streamer: streamer, VODID: vodID, ClipID: raw.ClipID,
		TargetOffset: raw.TargetOffset, Score: evaluation.Score,
		Threshold: evaluation.Threshold, Candidate: evaluation.Triggered,
	}, nil
}

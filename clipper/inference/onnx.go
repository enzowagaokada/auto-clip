package inference

import (
	"errors"
	"fmt"
	"math"
	"sync"

	"auto-clip/clipper/modelmeta"
	"auto-clip/clipper/preprocess"
	ort "github.com/yalue/onnxruntime_go"
)

var environment struct {
	sync.Mutex
	path string
	refs int
}

type Names struct {
	Tokens   string
	Features string
	Output   string
}

type Engine struct {
	mu            sync.Mutex
	session       *ort.AdvancedSession
	tokens        *ort.Tensor[int32]
	features      *ort.Tensor[float32]
	output        *ort.Tensor[float32]
	outputIsLogit bool
	closed        bool
}

// New initializes the process-wide ONNX Runtime environment on first use.
// Engines sharing a process must use the same runtime DLL path.
func New(runtimeDLL, modelPath string, bundle *modelmeta.Bundle, names Names, outputIsLogit bool) (*Engine, error) {
	if bundle == nil {
		return nil, errors.New("model bundle is nil")
	}
	if names.Tokens == "" || names.Features == "" || names.Output == "" {
		return nil, errors.New("all ONNX tensor names are required")
	}
	if err := acquireEnvironment(runtimeDLL); err != nil {
		return nil, err
	}
	ok := false
	defer func() {
		if !ok {
			_ = releaseEnvironment()
		}
	}()

	tokens, err := ort.NewTensor(ort.NewShape(1, int64(bundle.Metadata.MaxSequenceLength)),
		make([]int32, bundle.Metadata.MaxSequenceLength))
	if err != nil {
		return nil, fmt.Errorf("create token tensor: %w", err)
	}
	features, err := ort.NewTensor(ort.NewShape(1, int64(bundle.Metadata.NumberOfFeatures)),
		make([]float32, bundle.Metadata.NumberOfFeatures))
	if err != nil {
		_ = tokens.Destroy()
		return nil, fmt.Errorf("create feature tensor: %w", err)
	}
	output, err := ort.NewEmptyTensor[float32](ort.NewShape(1))
	if err != nil {
		_ = features.Destroy()
		_ = tokens.Destroy()
		return nil, fmt.Errorf("create output tensor: %w", err)
	}
	session, err := ort.NewAdvancedSession(modelPath,
		[]string{names.Tokens, names.Features},
		[]string{names.Output},
		[]ort.Value{tokens, features},
		[]ort.Value{output},
		nil,
	)
	if err != nil {
		_ = output.Destroy()
		_ = features.Destroy()
		_ = tokens.Destroy()
		return nil, fmt.Errorf("create ONNX session: %w", err)
	}
	ok = true
	return &Engine{
		session: session, tokens: tokens, features: features, output: output,
		outputIsLogit: outputIsLogit,
	}, nil
}

func (e *Engine) Score(encoded preprocess.Encoded) (float32, error) {
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.closed {
		return 0, errors.New("ONNX engine is closed")
	}
	if len(encoded.Tokens) != len(e.tokens.GetData()) {
		return 0, fmt.Errorf("got %d tokens, expected %d", len(encoded.Tokens), len(e.tokens.GetData()))
	}
	if len(encoded.Features) != len(e.features.GetData()) {
		return 0, fmt.Errorf("got %d features, expected %d", len(encoded.Features), len(e.features.GetData()))
	}
	copy(e.tokens.GetData(), encoded.Tokens)
	copy(e.features.GetData(), encoded.Features)
	if err := e.session.Run(); err != nil {
		return 0, fmt.Errorf("run ONNX inference: %w", err)
	}
	value := e.output.GetData()[0]
	if e.outputIsLogit {
		value = float32(1 / (1 + math.Exp(-float64(value))))
	}
	if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
		return 0, errors.New("ONNX produced a non-finite score")
	}
	return value, nil
}

func (e *Engine) Close() error {
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.closed {
		return nil
	}
	e.closed = true
	return errors.Join(
		e.session.Destroy(),
		e.output.Destroy(),
		e.features.Destroy(),
		e.tokens.Destroy(),
		releaseEnvironment(),
	)
}

func acquireEnvironment(path string) error {
	environment.Lock()
	defer environment.Unlock()
	if environment.refs > 0 {
		if environment.path != path {
			return fmt.Errorf("ONNX Runtime already initialized from %q", environment.path)
		}
		environment.refs++
		return nil
	}
	ort.SetSharedLibraryPath(path)
	if err := ort.InitializeEnvironment(); err != nil {
		return fmt.Errorf("initialize ONNX Runtime: %w", err)
	}
	environment.path = path
	environment.refs = 1
	return nil
}

func releaseEnvironment() error {
	environment.Lock()
	defer environment.Unlock()
	if environment.refs == 0 {
		return nil
	}
	environment.refs--
	if environment.refs > 0 {
		return nil
	}
	environment.path = ""
	return ort.DestroyEnvironment()
}

package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"log"
	"os"
	"os/signal"
	"path/filepath"
	"strings"
	"sync/atomic"
	"syscall"
	"time"

	"auto-clip/clipper/config"
	"auto-clip/clipper/core"
	"auto-clip/clipper/detection"
	"auto-clip/clipper/inference"
	"auto-clip/clipper/internal/twitch"
	"auto-clip/clipper/modelmeta"
	"auto-clip/clipper/preprocess"
	"auto-clip/clipper/replay"
	"auto-clip/clipper/store"
	"github.com/joho/godotenv"
)

type pathsFlag []string

func (p *pathsFlag) String() string { return strings.Join(*p, ",") }
func (p *pathsFlag) Set(value string) error {
	for _, path := range strings.Split(value, ",") {
		if path = strings.TrimSpace(path); path != "" {
			*p = append(*p, path)
		}
	}
	return nil
}

type liveSession struct {
	session *core.Session
	stream  twitch.Stream
}

type liveApp struct {
	cfg       config.Config
	bundle    *modelmeta.Bundle
	encoder   *preprocess.Encoder
	engine    *inference.Engine
	recorder  *store.JSONL
	client    *twitch.Client
	streamers map[string]config.Streamer
	sessions  map[string]liveSession
	chat      chan twitch.ChatMessage
	dropped   atomic.Uint64
}

func main() {
	if err := run(); err != nil {
		log.Printf("autoclip stopped: %v", err)
		os.Exit(1)
	}
}

func run() error {
	defaultRepo, err := defaultRepoRoot()
	if err != nil {
		return err
	}
	var replayPaths pathsFlag
	repo := flag.String("repo", defaultRepo, "repository root")
	configPath := flag.String("config", "config.yaml", "root YAML config path (relative to repo)")
	flag.Var(&replayPaths, "replay", "raw historical chat JSON path; repeat or comma-separate")
	flag.Parse()

	repoRoot, err := filepath.Abs(*repo)
	if err != nil {
		return fmt.Errorf("resolve repository root: %w", err)
	}
	rootConfig := *configPath
	if !filepath.IsAbs(rootConfig) {
		rootConfig = filepath.Join(repoRoot, filepath.FromSlash(rootConfig))
	}
	cfg, err := config.Load(rootConfig, repoRoot)
	if err != nil {
		return err
	}
	bundleDir := cfg.Resolve(cfg.Clipper.BundleDir)
	bundle, err := modelmeta.Load(bundleDir)
	if err != nil {
		return fmt.Errorf("load verified model bundle: %w", err)
	}
	encoder, err := preprocess.New(bundle)
	if err != nil {
		return fmt.Errorf("create encoder: %w", err)
	}
	engine, err := inference.New(
		cfg.Resolve(cfg.Clipper.RuntimeDLLPath),
		bundle.ModelPath(bundleDir),
		bundle,
		inference.Names{
			Tokens:   cfg.Clipper.TokensInputName,
			Features: cfg.Clipper.FeaturesInputName,
			Output:   cfg.Clipper.OutputName,
		},
		cfg.Clipper.OutputIsLogit,
	)
	if err != nil {
		return fmt.Errorf("initialize ONNX engine: %w", err)
	}
	defer engine.Close()

	if len(replayPaths) > 0 {
		resolved := make([]string, len(replayPaths))
		for i, path := range replayPaths {
			if !filepath.IsAbs(path) {
				path = filepath.Join(repoRoot, filepath.FromSlash(path))
			}
			resolved[i] = path
		}
		return replay.Run(resolved, replay.Options{
			Encoder: encoder, Scorer: engine,
			Threshold:      cfg.ThresholdFor(config.Streamer{}, bundle.Metadata.Threshold),
			Cooldown:       time.Duration(cfg.Clipper.CooldownSeconds) * time.Second,
			ManifestSHA256: bundle.ManifestChecksum, Output: os.Stdout,
		})
	}

	_ = godotenv.Load(filepath.Join(repoRoot, ".env"))
	clientID := strings.TrimSpace(os.Getenv("TWITCH_CLIENT_ID"))
	userToken := strings.TrimSpace(os.Getenv("TWITCH_USER_ACCESS_TOKEN"))
	if clientID == "" || userToken == "" {
		return errors.New("live mode requires TWITCH_CLIENT_ID and TWITCH_USER_ACCESS_TOKEN")
	}

	recorder, err := store.Open(
		cfg.Resolve(cfg.Clipper.CandidatesPath),
		cfg.Resolve(cfg.Clipper.SessionsPath),
	)
	if err != nil {
		return err
	}
	defer recorder.Close()

	active := cfg.ActiveStreamers()
	ids := make([]string, 0, len(active))
	streamers := make(map[string]config.Streamer, len(active))
	for _, streamer := range active {
		ids = append(ids, streamer.BroadcasterID)
		streamers[streamer.BroadcasterID] = streamer
	}
	twitchClient, err := twitch.NewClient(twitch.Config{
		ClientID: clientID, UserToken: userToken, BroadcasterIDs: ids,
	})
	if err != nil {
		return err
	}
	app := &liveApp{
		cfg: cfg, bundle: bundle, encoder: encoder, engine: engine,
		recorder: recorder, client: twitchClient, streamers: streamers,
		sessions: make(map[string]liveSession),
		chat:     make(chan twitch.ChatMessage, cfg.Clipper.ChatBufferSize),
	}
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	return app.run(ctx)
}

func (a *liveApp) run(ctx context.Context) error {
	eventSubDone := make(chan error, 1)
	go func() {
		eventSubDone <- a.client.Run(ctx, twitch.Callbacks{
			OnChatMessage: func(_ context.Context, message twitch.ChatMessage) {
				select {
				case a.chat <- message:
				default:
					a.dropped.Add(1)
				}
			},
			OnValidated: func(_ context.Context, token twitch.TokenInfo) {
				log.Printf("Twitch user token validated for %s", token.Login)
			},
			OnRevocation: func(_ context.Context, revocation twitch.Revocation) {
				log.Printf("EventSub revoked for broadcaster %s: %s", revocation.BroadcasterID, revocation.Status)
			},
			OnError: func(_ context.Context, err error) {
				log.Printf("EventSub transient error: %v", err)
			},
		})
	}()

	defer func() { a.closeSessions(time.Now().UTC()) }()
	if err := a.reconcileStreams(ctx, time.Now().UTC()); err != nil {
		return err
	}
	pollTicker := time.NewTicker(a.cfg.PollEvery)
	inferenceTicker := time.NewTicker(a.cfg.InferenceEvery)
	defer pollTicker.Stop()
	defer inferenceTicker.Stop()

	for {
		select {
		case <-ctx.Done():
			return nil
		case err := <-eventSubDone:
			if ctx.Err() != nil || errors.Is(err, context.Canceled) {
				return nil
			}
			return fmt.Errorf("EventSub stopped: %w", err)
		case message := <-a.chat:
			current, ok := a.sessions[message.BroadcasterID]
			if !ok {
				continue
			}
			if err := current.session.AddMessage(preprocess.Message{
				Time: message.Timestamp.UTC(), User: message.ChatterLogin, Text: message.Text,
			}); err != nil {
				log.Printf("add chat message for %s: %v", current.stream.BroadcasterLogin, err)
			}
		case now := <-pollTicker.C:
			if err := a.reconcileStreams(ctx, now.UTC()); err != nil {
				log.Printf("poll live streams: %v", err)
			}
		case now := <-inferenceTicker.C:
			a.evaluateSessions(now.UTC())
		}
	}
}

func (a *liveApp) reconcileStreams(ctx context.Context, now time.Time) error {
	streams, err := a.client.GetStreams(ctx, nil)
	if err != nil {
		return err
	}
	live := make(map[string]twitch.Stream, len(streams))
	for _, stream := range streams {
		live[stream.BroadcasterID] = stream
	}
	for broadcasterID, current := range a.sessions {
		stream, online := live[broadcasterID]
		if !online || stream.ID != current.stream.ID || !stream.StartedAt.Equal(current.stream.StartedAt) {
			if err := current.session.Close(now); err != nil {
				log.Printf("close session for %s: %v", current.stream.BroadcasterLogin, err)
			}
			delete(a.sessions, broadcasterID)
		}
	}
	for broadcasterID, stream := range live {
		if _, exists := a.sessions[broadcasterID]; exists {
			continue
		}
		streamer, configured := a.streamers[broadcasterID]
		if !configured {
			continue
		}
		machine, err := detection.New(
			a.cfg.ThresholdFor(streamer, a.bundle.Metadata.Threshold),
			a.cfg.CooldownFor(streamer),
		)
		if err != nil {
			return err
		}
		session, err := core.NewSession(core.Options{
			Streamer: streamer.Name, BroadcasterID: broadcasterID,
			StreamID: stream.ID, StreamStarted: stream.StartedAt, ObservedAt: now,
			Window: 35 * time.Second, TargetLag: 5 * time.Second,
			ManifestSHA256: a.bundle.ManifestChecksum,
		}, a.encoder, a.engine, machine, a.recorder)
		if err != nil {
			return fmt.Errorf("create session for %s: %w", streamer.Name, err)
		}
		a.sessions[broadcasterID] = liveSession{session: session, stream: stream}
		log.Printf("started shadow session for %s stream %s", streamer.Name, stream.ID)
	}
	return nil
}

func (a *liveApp) evaluateSessions(now time.Time) {
	if dropped := a.dropped.Swap(0); dropped > 0 {
		log.Printf("chat queue dropped %d messages since last inference tick", dropped)
	}
	for _, current := range a.sessions {
		if !current.session.Warm(now) {
			continue
		}
		candidate, err := current.session.Evaluate(now)
		if err != nil {
			log.Printf("evaluate %s: %v", current.stream.BroadcasterLogin, err)
			continue
		}
		if candidate != nil {
			log.Printf(
				"shadow candidate streamer=%s stream=%s score=%.4f threshold=%.4f target=%s",
				candidate.Streamer, candidate.StreamID, candidate.Score,
				candidate.Threshold, candidate.TargetAt.Format(time.RFC3339Nano),
			)
		}
	}
}

func (a *liveApp) closeSessions(at time.Time) {
	for broadcasterID, current := range a.sessions {
		if err := current.session.Close(at); err != nil {
			log.Printf("close session for %s: %v", current.stream.BroadcasterLogin, err)
		}
		delete(a.sessions, broadcasterID)
	}
}

func defaultRepoRoot() (string, error) {
	working, err := os.Getwd()
	if err != nil {
		return "", fmt.Errorf("get working directory: %w", err)
	}
	if filepath.Base(working) == "clipper" {
		return filepath.Dir(working), nil
	}
	return working, nil
}

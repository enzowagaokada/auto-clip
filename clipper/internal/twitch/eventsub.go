package twitch

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"sync"
	"time"

	"github.com/coder/websocket"
)

const (
	welcomeTimeout       = 15 * time.Second
	keepaliveGrace       = 2 * time.Second
	reconnectHandoffTime = 25 * time.Second
	maxEventSubMessage   = 1 << 20
)

var (
	// ErrSubscriptionRevoked indicates Twitch permanently revoked an active
	// subscription. Run returns this error instead of reconnecting indefinitely.
	ErrSubscriptionRevoked = errors.New("twitch: EventSub subscription revoked")

	// ErrAlreadyRunning is returned when Run is called twice concurrently on
	// one Client, preserving the one-shared-socket contract.
	ErrAlreadyRunning = errors.New("twitch: client is already running")
)

type eventSubEnvelope struct {
	Metadata struct {
		MessageID           string    `json:"message_id"`
		MessageType         string    `json:"message_type"`
		MessageTimestamp    time.Time `json:"message_timestamp"`
		SubscriptionType    string    `json:"subscription_type"`
		SubscriptionVersion string    `json:"subscription_version"`
	} `json:"metadata"`
	Payload struct {
		Session      eventSubSession      `json:"session"`
		Subscription eventSubSubscription `json:"subscription"`
		Event        json.RawMessage      `json:"event"`
	} `json:"payload"`
}

type eventSubSession struct {
	ID                      string  `json:"id"`
	Status                  string  `json:"status"`
	KeepaliveTimeoutSeconds *int    `json:"keepalive_timeout_seconds"`
	ReconnectURL            *string `json:"reconnect_url"`
}

type eventSubSubscription struct {
	ID        string `json:"id"`
	Status    string `json:"status"`
	Type      string `json:"type"`
	Version   string `json:"version"`
	Condition struct {
		BroadcasterUserID string `json:"broadcaster_user_id"`
	} `json:"condition"`
}

type chatEvent struct {
	BroadcasterUserID    string `json:"broadcaster_user_id"`
	BroadcasterUserLogin string `json:"broadcaster_user_login"`
	BroadcasterUserName  string `json:"broadcaster_user_name"`
	ChatterUserID        string `json:"chatter_user_id"`
	ChatterUserLogin     string `json:"chatter_user_login"`
	ChatterUserName      string `json:"chatter_user_name"`
	MessageID            string `json:"message_id"`
	Message              struct {
		Text string `json:"text"`
	} `json:"message"`
	MessageType string `json:"message_type"`
}

// Run validates the user token, revalidates it hourly, and runs one EventSub
// WebSocket shared by every configured broadcaster until ctx is canceled or a
// terminal authentication/subscription error occurs.
func (c *Client) Run(ctx context.Context, callbacks Callbacks) error {
	c.runMu.Lock()
	if c.running {
		c.runMu.Unlock()
		return ErrAlreadyRunning
	}
	c.running = true
	c.runMu.Unlock()
	defer func() {
		c.runMu.Lock()
		c.running = false
		c.runMu.Unlock()
	}()

	token, err := c.ValidateToken(ctx)
	if err != nil {
		return err
	}
	callValidated(ctx, callbacks, token)

	runCtx, cancel := context.WithCancel(ctx)
	defer cancel()

	validationErr := make(chan error, 1)
	go c.validateHourly(runCtx, callbacks, validationErr, cancel)

	socketErr := c.runEventSub(runCtx, token.UserID, callbacks)
	select {
	case err := <-validationErr:
		return err
	default:
	}
	if ctx.Err() != nil {
		return ctx.Err()
	}
	return socketErr
}

func (c *Client) validateHourly(
	ctx context.Context,
	callbacks Callbacks,
	errs chan<- error,
	cancel context.CancelFunc,
) {
	ticker := time.NewTicker(c.validateEvery)
	defer ticker.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			token, err := c.ValidateToken(ctx)
			if err != nil {
				select {
				case errs <- fmt.Errorf("twitch: hourly token validation failed: %w", err):
				default:
				}
				cancel()
				return
			}
			callValidated(ctx, callbacks, token)
		}
	}
}

func (c *Client) runEventSub(ctx context.Context, tokenUserID string, callbacks Callbacks) error {
	backoff := time.Second
	for {
		conn, welcome, err := c.connectAndWelcome(ctx, c.eventSubURL)
		if err != nil {
			if ctx.Err() != nil {
				return ctx.Err()
			}
			callError(ctx, callbacks, err)
			if err := waitFor(ctx, backoff); err != nil {
				return err
			}
			backoff = nextBackoff(backoff)
			continue
		}

		if err := c.createChatSubscriptions(ctx, welcome.Payload.Session.ID, tokenUserID); err != nil {
			conn.CloseNow()
			return err
		}
		backoff = time.Second

		for {
			reconnectURL, err := c.consumeConnection(ctx, conn, welcome, callbacks)
			if reconnectURL == "" {
				conn.CloseNow()
				if errors.Is(err, ErrSubscriptionRevoked) {
					return err
				}
				if err != nil && ctx.Err() == nil {
					callError(ctx, callbacks, err)
				}
				break
			}

			nextConn, nextWelcome, handoffErr := c.handoff(ctx, reconnectURL, callbacks)
			if handoffErr != nil {
				conn.CloseNow()
				if ctx.Err() != nil {
					return ctx.Err()
				}
				callError(ctx, callbacks, handoffErr)
				break
			}

			// Twitch transfers existing subscriptions to reconnect_url sessions.
			// Close the old socket only after the replacement sends Welcome.
			conn.CloseNow()
			conn = nextConn
			welcome = nextWelcome
		}

		if err := waitFor(ctx, backoff); err != nil {
			return err
		}
		backoff = nextBackoff(backoff)
	}
}

func (c *Client) connectAndWelcome(
	ctx context.Context,
	endpoint string,
) (*websocket.Conn, eventSubEnvelope, error) {
	connectCtx, cancel := context.WithTimeout(ctx, welcomeTimeout)
	defer cancel()

	conn, resp, err := websocket.Dial(connectCtx, endpoint, &websocket.DialOptions{
		HTTPClient: c.httpClient,
	})
	if err != nil {
		if resp != nil {
			return nil, eventSubEnvelope{}, fmt.Errorf(
				"twitch: EventSub WebSocket handshake returned HTTP %d",
				resp.StatusCode,
			)
		}
		// Do not wrap the dial error: errors for reconnect URLs may contain the
		// URL's opaque session data.
		return nil, eventSubEnvelope{}, errors.New("twitch: EventSub WebSocket dial failed")
	}
	conn.SetReadLimit(maxEventSubMessage)

	messageType, data, err := conn.Read(connectCtx)
	if err != nil {
		conn.CloseNow()
		return nil, eventSubEnvelope{}, fmt.Errorf("twitch: read EventSub welcome: %w", err)
	}
	if messageType != websocket.MessageText {
		conn.CloseNow()
		return nil, eventSubEnvelope{}, errors.New("twitch: EventSub welcome was not a text message")
	}
	welcome, err := parseEnvelope(data)
	if err != nil {
		conn.CloseNow()
		return nil, eventSubEnvelope{}, err
	}
	if welcome.Metadata.MessageType != "session_welcome" || welcome.Payload.Session.ID == "" {
		conn.CloseNow()
		return nil, eventSubEnvelope{}, errors.New("twitch: first EventSub message was not a valid session_welcome")
	}
	c.seen.seenOrAdd(welcome.Metadata.MessageID)
	return conn, welcome, nil
}

func (c *Client) consumeConnection(
	ctx context.Context,
	conn *websocket.Conn,
	welcome eventSubEnvelope,
	callbacks Callbacks,
) (string, error) {
	keepalive := 10 * time.Second
	if seconds := welcome.Payload.Session.KeepaliveTimeoutSeconds; seconds != nil && *seconds > 0 {
		keepalive = time.Duration(*seconds) * time.Second
	}

	for {
		readCtx, cancel := context.WithTimeout(ctx, keepalive+keepaliveGrace)
		messageType, data, err := conn.Read(readCtx)
		deadlineExceeded := errors.Is(readCtx.Err(), context.DeadlineExceeded)
		cancel()
		if err != nil {
			if deadlineExceeded {
				return "", errors.New("twitch: EventSub keepalive deadline exceeded")
			}
			return "", fmt.Errorf("twitch: read EventSub message: %w", err)
		}
		if messageType != websocket.MessageText {
			continue
		}

		envelope, err := parseEnvelope(data)
		if err != nil {
			callError(ctx, callbacks, err)
			continue
		}

		switch envelope.Metadata.MessageType {
		case "session_keepalive":
			c.seen.seenOrAdd(envelope.Metadata.MessageID)
		case "notification":
			if envelope.Metadata.SubscriptionType != "channel.chat.message" {
				c.seen.seenOrAdd(envelope.Metadata.MessageID)
				continue
			}
			message, err := chatMessageFromEnvelope(envelope)
			if err != nil {
				c.seen.seenOrAdd(envelope.Metadata.MessageID)
				callError(ctx, callbacks, err)
				continue
			}
			if c.seen.seenOrAddAny(envelope.Metadata.MessageID, message.MessageID) {
				continue
			}
			if callbacks.OnChatMessage != nil {
				callbacks.OnChatMessage(ctx, message)
			}
		case "session_reconnect":
			if c.seen.seenOrAdd(envelope.Metadata.MessageID) {
				continue
			}
			if envelope.Payload.Session.ReconnectURL == nil || *envelope.Payload.Session.ReconnectURL == "" {
				return "", errors.New("twitch: session_reconnect omitted reconnect_url")
			}
			return *envelope.Payload.Session.ReconnectURL, nil
		case "revocation":
			if c.seen.seenOrAdd(envelope.Metadata.MessageID) {
				continue
			}
			revocation := revocationFromEnvelope(envelope)
			if callbacks.OnRevocation != nil {
				callbacks.OnRevocation(ctx, revocation)
			}
			return "", fmt.Errorf(
				"%w: type %s, status %s",
				ErrSubscriptionRevoked,
				revocation.Type,
				revocation.Status,
			)
		default:
			c.seen.seenOrAdd(envelope.Metadata.MessageID)
		}
	}
}

func (c *Client) handoff(
	ctx context.Context,
	reconnectURL string,
	callbacks Callbacks,
) (*websocket.Conn, eventSubEnvelope, error) {
	handoffCtx, cancel := context.WithTimeout(ctx, reconnectHandoffTime)
	defer cancel()

	backoff := time.Second
	for {
		conn, welcome, err := c.connectAndWelcome(handoffCtx, reconnectURL)
		if err == nil {
			return conn, welcome, nil
		}
		if handoffCtx.Err() != nil {
			return nil, eventSubEnvelope{}, errors.New("twitch: EventSub reconnect handoff timed out")
		}
		callError(ctx, callbacks, err)
		if err := waitFor(handoffCtx, backoff); err != nil {
			return nil, eventSubEnvelope{}, errors.New("twitch: EventSub reconnect handoff timed out")
		}
		backoff = nextBackoff(backoff)
	}
}

func parseEnvelope(data []byte) (eventSubEnvelope, error) {
	var envelope eventSubEnvelope
	if err := json.Unmarshal(data, &envelope); err != nil {
		return eventSubEnvelope{}, fmt.Errorf("twitch: decode EventSub message: %w", err)
	}
	if envelope.Metadata.MessageID == "" || envelope.Metadata.MessageType == "" ||
		envelope.Metadata.MessageTimestamp.IsZero() {
		return eventSubEnvelope{}, errors.New("twitch: EventSub message has invalid metadata")
	}
	return envelope, nil
}

func chatMessageFromEnvelope(envelope eventSubEnvelope) (ChatMessage, error) {
	var event chatEvent
	if err := json.Unmarshal(envelope.Payload.Event, &event); err != nil {
		return ChatMessage{}, fmt.Errorf("twitch: decode channel.chat.message event: %w", err)
	}
	if event.MessageID == "" || event.BroadcasterUserID == "" {
		return ChatMessage{}, errors.New("twitch: channel.chat.message event is missing required IDs")
	}
	return ChatMessage{
		EnvelopeID:       envelope.Metadata.MessageID,
		MessageID:        event.MessageID,
		Timestamp:        envelope.Metadata.MessageTimestamp,
		BroadcasterID:    event.BroadcasterUserID,
		BroadcasterLogin: event.BroadcasterUserLogin,
		BroadcasterName:  event.BroadcasterUserName,
		ChatterID:        event.ChatterUserID,
		ChatterLogin:     event.ChatterUserLogin,
		ChatterName:      event.ChatterUserName,
		Text:             event.Message.Text,
		MessageType:      event.MessageType,
	}, nil
}

func revocationFromEnvelope(envelope eventSubEnvelope) Revocation {
	subscription := envelope.Payload.Subscription
	return Revocation{
		EnvelopeID:     envelope.Metadata.MessageID,
		Timestamp:      envelope.Metadata.MessageTimestamp,
		SubscriptionID: subscription.ID,
		Type:           subscription.Type,
		Version:        subscription.Version,
		Status:         subscription.Status,
		BroadcasterID:  subscription.Condition.BroadcasterUserID,
	}
}

func callValidated(ctx context.Context, callbacks Callbacks, token TokenInfo) {
	if callbacks.OnValidated != nil {
		callbacks.OnValidated(ctx, token)
	}
}

func callError(ctx context.Context, callbacks Callbacks, err error) {
	if callbacks.OnError != nil && err != nil {
		callbacks.OnError(ctx, err)
	}
}

func waitFor(ctx context.Context, duration time.Duration) error {
	timer := time.NewTimer(duration)
	defer timer.Stop()
	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-timer.C:
		return nil
	}
}

func nextBackoff(current time.Duration) time.Duration {
	const maximum = 30 * time.Second
	current *= 2
	if current > maximum {
		return maximum
	}
	return current
}

type deduper struct {
	mu       sync.Mutex
	capacity int
	ids      map[string]struct{}
	order    []string
}

func newDeduper(capacity int) *deduper {
	return &deduper{
		capacity: capacity,
		ids:      make(map[string]struct{}, capacity),
		order:    make([]string, 0, capacity),
	}
}

func (d *deduper) seenOrAdd(id string) bool {
	return d.seenOrAddAny(id)
}

// seenOrAddAny atomically reports whether any non-empty ID was already seen,
// then records every supplied ID so both envelope and event IDs are retained.
func (d *deduper) seenOrAddAny(ids ...string) bool {
	d.mu.Lock()
	defer d.mu.Unlock()

	duplicate := false
	for _, id := range ids {
		if id == "" {
			continue
		}
		if _, exists := d.ids[id]; exists {
			duplicate = true
		}
	}
	for _, id := range ids {
		if id == "" {
			continue
		}
		if _, exists := d.ids[id]; exists {
			continue
		}
		d.ids[id] = struct{}{}
		d.order = append(d.order, id)
	}
	for len(d.order) > d.capacity {
		oldest := d.order[0]
		d.order = d.order[1:]
		delete(d.ids, oldest)
	}
	return duplicate
}

// Package twitch provides the standalone clipper's Twitch HTTP and EventSub
// connectivity. A Client is safe to run as the single shared EventSub connection
// for all configured broadcasters.
package twitch

import (
	"context"
	"net/http"
	"time"
)

const (
	defaultValidateURL = "https://id.twitch.tv/oauth2/validate"
	defaultHelixURL    = "https://api.twitch.tv/helix"
	defaultEventSubURL = "wss://eventsub.wss.twitch.tv/ws"
)

// Config contains credentials and channels used by Client.
// UserToken must be a user access token with user:read:chat.
type Config struct {
	ClientID       string
	UserToken      string
	BroadcasterIDs []string

	// HTTPClient is optional. http.DefaultClient is used when nil.
	HTTPClient *http.Client

	// ValidationInterval defaults to one hour. It is configurable for callers
	// that want shorter intervals in tests.
	ValidationInterval time.Duration

	// Endpoint overrides are intended for tests.
	ValidateURL string
	HelixURL    string
	EventSubURL string
}

// Callbacks are invoked synchronously by their producing goroutine. OnValidated
// may run concurrently with socket callbacks, so handlers must be thread-safe.
// Handlers should return quickly to avoid blocking keepalive reads.
type Callbacks struct {
	OnChatMessage func(context.Context, ChatMessage)
	OnRevocation  func(context.Context, Revocation)
	OnValidated   func(context.Context, TokenInfo)
	OnError       func(context.Context, error)
}

// TokenInfo is the verified identity associated with the configured user token.
type TokenInfo struct {
	ClientID  string
	Login     string
	UserID    string
	Scopes    []string
	ExpiresIn int
}

// Stream is an online stream returned by Helix Get Streams.
type Stream struct {
	ID               string
	BroadcasterID    string
	BroadcasterLogin string
	BroadcasterName  string
	Title            string
	GameID           string
	GameName         string
	StartedAt        time.Time
}

// ChatMessage is the normalized subset of channel.chat.message needed by the
// clipper. Timestamp comes from the EventSub envelope metadata.
type ChatMessage struct {
	EnvelopeID       string
	MessageID        string
	Timestamp        time.Time
	BroadcasterID    string
	BroadcasterLogin string
	BroadcasterName  string
	ChatterID        string
	ChatterLogin     string
	ChatterName      string
	Text             string
	MessageType      string
}

// Revocation describes an EventSub subscription Twitch has revoked.
type Revocation struct {
	EnvelopeID     string
	Timestamp      time.Time
	SubscriptionID string
	Type           string
	Version        string
	Status         string
	BroadcasterID  string
}

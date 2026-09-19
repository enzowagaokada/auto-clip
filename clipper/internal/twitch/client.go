package twitch

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strconv"
	"strings"
	"sync"
	"time"
)

const (
	requiredChatScope   = "user:read:chat"
	maxResponseBytes    = 1 << 20
	maxStreamBatch      = 100
	maxVideoPage        = 100
	defaultArchiveRetry = 15 * time.Second
	defaultArchiveSlack = 2 * time.Minute
	maxArchiveRetry     = 2 * time.Minute
)

// Client owns the Twitch API configuration and one shared EventSub connection.
type Client struct {
	clientID       string
	userToken      string
	broadcasterIDs []string
	httpClient     *http.Client
	validateURL    string
	helixURL       string
	eventSubURL    string
	validateEvery  time.Duration
	seen           *deduper
	archiveRetry   time.Duration
	archiveSlack   time.Duration
	runMu          sync.Mutex
	running        bool
}

// NewClient validates local configuration but does not make network requests.
func NewClient(cfg Config) (*Client, error) {
	if strings.TrimSpace(cfg.ClientID) == "" {
		return nil, errors.New("twitch: client ID is required")
	}
	if strings.TrimSpace(cfg.UserToken) == "" {
		return nil, errors.New("twitch: user token is required")
	}

	ids := uniqueNonEmpty(cfg.BroadcasterIDs)
	if len(ids) == 0 {
		return nil, errors.New("twitch: at least one broadcaster ID is required")
	}
	if len(ids) > 300 {
		return nil, fmt.Errorf("twitch: %d broadcaster IDs exceeds the 300 subscriptions-per-socket limit", len(ids))
	}

	httpClient := cfg.HTTPClient
	if httpClient == nil {
		httpClient = &http.Client{Timeout: 20 * time.Second}
	}
	validateEvery := cfg.ValidationInterval
	if validateEvery <= 0 {
		validateEvery = time.Hour
	}

	return &Client{
		clientID:       cfg.ClientID,
		userToken:      cfg.UserToken,
		broadcasterIDs: ids,
		httpClient:     httpClient,
		validateURL:    valueOr(cfg.ValidateURL, defaultValidateURL),
		helixURL:       strings.TrimRight(valueOr(cfg.HelixURL, defaultHelixURL), "/"),
		eventSubURL:    valueOr(cfg.EventSubURL, defaultEventSubURL),
		validateEvery:  validateEvery,
		seen:           newDeduper(20_000),
		archiveRetry:   valueDuration(cfg.ArchiveRetryWait, defaultArchiveRetry),
		archiveSlack:   valueDuration(cfg.ArchiveCreatedSlack, defaultArchiveSlack),
	}, nil
}

// ValidateToken verifies that the token belongs to the configured client, is a
// user token, and grants user:read:chat.
func (c *Client) ValidateToken(ctx context.Context) (TokenInfo, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, c.validateURL, nil)
	if err != nil {
		return TokenInfo{}, fmt.Errorf("twitch: create token validation request: %w", err)
	}
	req.Header.Set("Authorization", "OAuth "+c.userToken)

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return TokenInfo{}, fmt.Errorf("twitch: validate token: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return TokenInfo{}, apiStatusError("token validation", resp)
	}

	var wire struct {
		ClientID  string   `json:"client_id"`
		Login     string   `json:"login"`
		UserID    string   `json:"user_id"`
		Scopes    []string `json:"scopes"`
		ExpiresIn int      `json:"expires_in"`
	}
	if err := decodeJSON(resp.Body, &wire); err != nil {
		return TokenInfo{}, fmt.Errorf("twitch: decode token validation response: %w", err)
	}
	if wire.ClientID != c.clientID {
		return TokenInfo{}, errors.New("twitch: token client ID does not match configured client ID")
	}
	if wire.UserID == "" {
		return TokenInfo{}, errors.New("twitch: token is not a user access token")
	}
	if !contains(wire.Scopes, requiredChatScope) {
		return TokenInfo{}, fmt.Errorf("twitch: token is missing required scope %q", requiredChatScope)
	}

	return TokenInfo{
		ClientID:  wire.ClientID,
		Login:     wire.Login,
		UserID:    wire.UserID,
		Scopes:    append([]string(nil), wire.Scopes...),
		ExpiresIn: wire.ExpiresIn,
	}, nil
}

// GetStreams returns currently live streams for broadcasterIDs. Twitch accepts
// at most 100 user_id parameters per request, so larger inputs are batched.
// If broadcasterIDs is empty, the IDs configured on Client are used.
func (c *Client) GetStreams(ctx context.Context, broadcasterIDs []string) ([]Stream, error) {
	ids := uniqueNonEmpty(broadcasterIDs)
	if len(ids) == 0 {
		ids = append([]string(nil), c.broadcasterIDs...)
	}

	var streams []Stream
	for start := 0; start < len(ids); start += maxStreamBatch {
		end := start + maxStreamBatch
		if end > len(ids) {
			end = len(ids)
		}
		batch, err := c.getStreamsBatch(ctx, ids[start:end])
		if err != nil {
			return nil, err
		}
		streams = append(streams, batch...)
	}
	return streams, nil
}

func (c *Client) getStreamsBatch(ctx context.Context, ids []string) ([]Stream, error) {
	query := make(url.Values, len(ids))
	for _, id := range ids {
		query.Add("user_id", id)
	}
	endpoint := c.helixURL + "/streams?" + query.Encode()
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, endpoint, nil)
	if err != nil {
		return nil, fmt.Errorf("twitch: create Get Streams request: %w", err)
	}
	c.setHelixHeaders(req)

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("twitch: Get Streams: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return nil, apiStatusError("Get Streams", resp)
	}

	var wire struct {
		Data []struct {
			ID        string    `json:"id"`
			UserID    string    `json:"user_id"`
			UserLogin string    `json:"user_login"`
			UserName  string    `json:"user_name"`
			GameID    string    `json:"game_id"`
			GameName  string    `json:"game_name"`
			Title     string    `json:"title"`
			StartedAt time.Time `json:"started_at"`
		} `json:"data"`
	}
	if err := decodeJSON(resp.Body, &wire); err != nil {
		return nil, fmt.Errorf("twitch: decode Get Streams response: %w", err)
	}

	result := make([]Stream, 0, len(wire.Data))
	for _, item := range wire.Data {
		result = append(result, Stream{
			ID:               item.ID,
			BroadcasterID:    item.UserID,
			BroadcasterLogin: item.UserLogin,
			BroadcasterName:  item.UserName,
			Title:            item.Title,
			GameID:           item.GameID,
			GameName:         item.GameName,
			StartedAt:        item.StartedAt,
		})
	}
	return result, nil
}

type helixVideo struct {
	ID        string    `json:"id"`
	StreamID  string    `json:"stream_id"`
	UserID    string    `json:"user_id"`
	CreatedAt time.Time `json:"created_at"`
	Type      string    `json:"type"`
}

// FindArchiveVOD paginates Helix Get Videos for a broadcaster and returns the
// archive whose stream_id matches. If Helix omits stream_id, a unique created_at
// match within ArchiveCreatedSlack of startedAt is accepted.
func (c *Client) FindArchiveVOD(ctx context.Context, userID, streamID string, startedAt time.Time) (string, error) {
	userID = strings.TrimSpace(userID)
	streamID = strings.TrimSpace(streamID)
	if userID == "" {
		return "", errors.New("twitch: user ID is required")
	}
	var cursor string
	var timeMatches []string
	for {
		videos, next, err := c.getVideosPage(ctx, userID, cursor)
		if err != nil {
			return "", err
		}
		for _, video := range videos {
			if streamID != "" && strings.TrimSpace(video.StreamID) == streamID && video.ID != "" {
				return video.ID, nil
			}
			if strings.TrimSpace(video.StreamID) != "" {
				continue
			}
			if startedAt.IsZero() || video.CreatedAt.IsZero() {
				continue
			}
			delta := video.CreatedAt.Sub(startedAt)
			if delta < 0 {
				delta = -delta
			}
			if delta <= c.archiveSlack && video.ID != "" {
				timeMatches = append(timeMatches, video.ID)
			}
		}
		if next == "" {
			break
		}
		cursor = next
	}
	unique := uniqueNonEmpty(timeMatches)
	if len(unique) == 1 {
		return unique[0], nil
	}
	return "", nil
}

// WaitForArchiveVOD polls FindArchiveVOD until a VOD appears or ctx ends.
func (c *Client) WaitForArchiveVOD(ctx context.Context, userID, streamID string, startedAt time.Time) (string, error) {
	delay := c.archiveRetry
	if delay <= 0 {
		delay = defaultArchiveRetry
	}
	for {
		vodID, err := c.FindArchiveVOD(ctx, userID, streamID, startedAt)
		if err != nil {
			return "", err
		}
		if vodID != "" {
			return vodID, nil
		}
		timer := time.NewTimer(delay)
		select {
		case <-ctx.Done():
			timer.Stop()
			return "", ctx.Err()
		case <-timer.C:
		}
		if delay < maxArchiveRetry {
			delay *= 2
			if delay > maxArchiveRetry {
				delay = maxArchiveRetry
			}
		}
	}
}

func (c *Client) getVideosPage(ctx context.Context, userID, cursor string) ([]helixVideo, string, error) {
	query := url.Values{}
	query.Set("user_id", userID)
	query.Set("type", "archive")
	query.Set("first", strconv.Itoa(maxVideoPage))
	if cursor != "" {
		query.Set("after", cursor)
	}
	endpoint := c.helixURL + "/videos?" + query.Encode()
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, endpoint, nil)
	if err != nil {
		return nil, "", fmt.Errorf("twitch: create Get Videos request: %w", err)
	}
	c.setHelixHeaders(req)
	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, "", fmt.Errorf("twitch: Get Videos: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return nil, "", apiStatusError("Get Videos", resp)
	}
	var wire struct {
		Data       []helixVideo `json:"data"`
		Pagination struct {
			Cursor string `json:"cursor"`
		} `json:"pagination"`
	}
	if err := decodeJSON(resp.Body, &wire); err != nil {
		return nil, "", fmt.Errorf("twitch: decode Get Videos response: %w", err)
	}
	return wire.Data, strings.TrimSpace(wire.Pagination.Cursor), nil
}

func (c *Client) createChatSubscriptions(ctx context.Context, sessionID, tokenUserID string) error {
	for _, broadcasterID := range c.broadcasterIDs {
		body := struct {
			Type      string `json:"type"`
			Version   string `json:"version"`
			Condition struct {
				BroadcasterUserID string `json:"broadcaster_user_id"`
				UserID            string `json:"user_id"`
			} `json:"condition"`
			Transport struct {
				Method    string `json:"method"`
				SessionID string `json:"session_id"`
			} `json:"transport"`
		}{
			Type:    "channel.chat.message",
			Version: "1",
		}
		body.Condition.BroadcasterUserID = broadcasterID
		body.Condition.UserID = tokenUserID
		body.Transport.Method = "websocket"
		body.Transport.SessionID = sessionID

		encoded, err := json.Marshal(body)
		if err != nil {
			return fmt.Errorf("twitch: encode EventSub subscription: %w", err)
		}
		req, err := http.NewRequestWithContext(
			ctx,
			http.MethodPost,
			c.helixURL+"/eventsub/subscriptions",
			bytes.NewReader(encoded),
		)
		if err != nil {
			return fmt.Errorf("twitch: create EventSub subscription request: %w", err)
		}
		c.setHelixHeaders(req)
		req.Header.Set("Content-Type", "application/json")

		resp, err := c.httpClient.Do(req)
		if err != nil {
			return fmt.Errorf("twitch: create EventSub subscription for broadcaster %s: %w", broadcasterID, err)
		}
		if resp.StatusCode != http.StatusAccepted {
			statusErr := apiStatusError("create EventSub subscription", resp)
			resp.Body.Close()
			return fmt.Errorf("twitch: broadcaster %s: %w", broadcasterID, statusErr)
		}
		io.Copy(io.Discard, io.LimitReader(resp.Body, maxResponseBytes))
		resp.Body.Close()
	}
	return nil
}

func (c *Client) setHelixHeaders(req *http.Request) {
	req.Header.Set("Authorization", "Bearer "+c.userToken)
	req.Header.Set("Client-Id", c.clientID)
}

func apiStatusError(operation string, resp *http.Response) error {
	var payload struct {
		Message string `json:"message"`
	}
	_ = decodeJSON(resp.Body, &payload)
	if payload.Message == "" {
		return fmt.Errorf("twitch: %s returned HTTP %d", operation, resp.StatusCode)
	}
	return fmt.Errorf("twitch: %s returned HTTP %d: %s", operation, resp.StatusCode, payload.Message)
}

func decodeJSON(r io.Reader, target any) error {
	decoder := json.NewDecoder(io.LimitReader(r, maxResponseBytes))
	return decoder.Decode(target)
}

func uniqueNonEmpty(values []string) []string {
	seen := make(map[string]struct{}, len(values))
	result := make([]string, 0, len(values))
	for _, value := range values {
		value = strings.TrimSpace(value)
		if value == "" {
			continue
		}
		if _, ok := seen[value]; ok {
			continue
		}
		seen[value] = struct{}{}
		result = append(result, value)
	}
	return result
}

func contains(values []string, wanted string) bool {
	for _, value := range values {
		if value == wanted {
			return true
		}
	}
	return false
}

func valueDuration(value, fallback time.Duration) time.Duration {
	if value > 0 {
		return value
	}
	return fallback
}

func valueOr(value, fallback string) string {
	if value != "" {
		return value
	}
	return fallback
}

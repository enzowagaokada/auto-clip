package twitch

import (
	"testing"
	"time"
)

func TestChatMessageFromEnvelope(t *testing.T) {
	const payload = `{
		"metadata": {
			"message_id": "envelope-1",
			"message_type": "notification",
			"message_timestamp": "2026-08-02T20:31:02.123456789Z",
			"subscription_type": "channel.chat.message",
			"subscription_version": "1"
		},
		"payload": {
			"subscription": {
				"id": "subscription-1",
				"status": "enabled",
				"type": "channel.chat.message",
				"version": "1"
			},
			"event": {
				"broadcaster_user_id": "100",
				"broadcaster_user_login": "channel",
				"broadcaster_user_name": "Channel",
				"chatter_user_id": "200",
				"chatter_user_login": "viewer",
				"chatter_user_name": "Viewer",
				"message_id": "chat-1",
				"message": {"text": "KEKW no way"},
				"message_type": "text"
			}
		}
	}`

	envelope, err := parseEnvelope([]byte(payload))
	if err != nil {
		t.Fatalf("parseEnvelope() error = %v", err)
	}
	message, err := chatMessageFromEnvelope(envelope)
	if err != nil {
		t.Fatalf("chatMessageFromEnvelope() error = %v", err)
	}

	wantTime := time.Date(2026, time.August, 2, 20, 31, 2, 123456789, time.UTC)
	if message.EnvelopeID != "envelope-1" || message.MessageID != "chat-1" {
		t.Fatalf("unexpected IDs: envelope=%q message=%q", message.EnvelopeID, message.MessageID)
	}
	if !message.Timestamp.Equal(wantTime) {
		t.Fatalf("Timestamp = %s, want %s", message.Timestamp, wantTime)
	}
	if message.BroadcasterID != "100" || message.ChatterID != "200" {
		t.Fatalf("unexpected users: broadcaster=%q chatter=%q", message.BroadcasterID, message.ChatterID)
	}
	if message.Text != "KEKW no way" || message.MessageType != "text" {
		t.Fatalf("unexpected message: text=%q type=%q", message.Text, message.MessageType)
	}
}

func TestParseEnvelopeRequiresMetadata(t *testing.T) {
	_, err := parseEnvelope([]byte(`{"metadata":{"message_type":"notification"}}`))
	if err == nil {
		t.Fatal("parseEnvelope() error = nil, want invalid metadata error")
	}
}

func TestDeduperChecksEnvelopeAndEventIDsAtomically(t *testing.T) {
	d := newDeduper(4)
	if d.seenOrAddAny("envelope-1", "chat-1") {
		t.Fatal("first message was reported as duplicate")
	}
	if !d.seenOrAddAny("envelope-2", "chat-1") {
		t.Fatal("duplicate chat message ID was not detected")
	}
	if !d.seenOrAddAny("envelope-2", "chat-2") {
		t.Fatal("envelope ID from duplicate event was not retained")
	}
}

func TestRevocationFromEnvelope(t *testing.T) {
	const payload = `{
		"metadata": {
			"message_id": "envelope-r",
			"message_type": "revocation",
			"message_timestamp": "2026-08-02T20:31:02Z",
			"subscription_type": "channel.chat.message",
			"subscription_version": "1"
		},
		"payload": {
			"subscription": {
				"id": "subscription-r",
				"status": "authorization_revoked",
				"type": "channel.chat.message",
				"version": "1",
				"condition": {"broadcaster_user_id": "100"}
			}
		}
	}`

	envelope, err := parseEnvelope([]byte(payload))
	if err != nil {
		t.Fatalf("parseEnvelope() error = %v", err)
	}
	revocation := revocationFromEnvelope(envelope)
	if revocation.Status != "authorization_revoked" ||
		revocation.SubscriptionID != "subscription-r" ||
		revocation.BroadcasterID != "100" {
		t.Fatalf("unexpected revocation: %+v", revocation)
	}
}

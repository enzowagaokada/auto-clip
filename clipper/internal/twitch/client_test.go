package twitch

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
	"time"
)

func TestFindArchiveVODPaginatesAndMatchesStreamID(t *testing.T) {
	var pages int
	server := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		if request.URL.Path != "/videos" {
			http.NotFound(writer, request)
			return
		}
		pages++
		after := request.URL.Query().Get("after")
		if after == "" {
			writeVideos(writer, []map[string]string{
				{"id": "old", "stream_id": "other", "user_id": "u1"},
			}, "cursor-1")
			return
		}
		if after != "cursor-1" {
			t.Errorf("unexpected cursor %q", after)
		}
		writeVideos(writer, []map[string]string{
			{"id": "wanted", "stream_id": "live-1", "user_id": "u1"},
		}, "")
	}))
	defer server.Close()

	client := testHelixClient(t, server)
	vodID, err := client.FindArchiveVOD(context.Background(), "u1", "live-1", time.Time{})
	if err != nil {
		t.Fatal(err)
	}
	if vodID != "wanted" {
		t.Fatalf("vod ID = %q, want wanted", vodID)
	}
	if pages != 2 {
		t.Fatalf("pages = %d, want 2", pages)
	}
}

func TestFindArchiveVODUsesUniqueCreatedAtFallback(t *testing.T) {
	started := time.Date(2026, 1, 1, 0, 0, 0, 0, time.UTC)
	server := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		writeVideos(writer, []map[string]string{
			{
				"id": "time-only", "user_id": "u1",
				"created_at": "2026-01-01T00:00:30Z",
			},
		}, "")
	}))
	defer server.Close()
	client := testHelixClient(t, server)
	vodID, err := client.FindArchiveVOD(context.Background(), "u1", "live-missing", started)
	if err != nil {
		t.Fatal(err)
	}
	if vodID != "time-only" {
		t.Fatalf("vod ID = %q, want time-only", vodID)
	}
}

func TestWaitForArchiveVODRetriesUntilMatch(t *testing.T) {
	var calls atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		n := calls.Add(1)
		if n == 1 {
			writeVideos(writer, nil, "")
			return
		}
		writeVideos(writer, []map[string]string{
			{"id": "later", "stream_id": "live-1", "user_id": "u1"},
		}, "")
	}))
	defer server.Close()
	client := testHelixClient(t, server)
	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()
	vodID, err := client.WaitForArchiveVOD(ctx, "u1", "live-1", time.Time{})
	if err != nil {
		t.Fatal(err)
	}
	if vodID != "later" {
		t.Fatalf("vod ID = %q, want later", vodID)
	}
	if calls.Load() < 2 {
		t.Fatalf("calls = %d, want at least 2", calls.Load())
	}
}

func testHelixClient(t *testing.T, server *httptest.Server) *Client {
	t.Helper()
	client, err := NewClient(Config{
		ClientID: "client", UserToken: "token", BroadcasterIDs: []string{"u1"},
		HTTPClient: server.Client(), HelixURL: server.URL,
		ArchiveRetryWait: time.Millisecond,
	})
	if err != nil {
		t.Fatal(err)
	}
	return client
}

func writeVideos(writer http.ResponseWriter, videos []map[string]string, cursor string) {
	payload := map[string]any{"data": videos, "pagination": map[string]string{}}
	if cursor != "" {
		payload["pagination"] = map[string]string{"cursor": cursor}
	}
	writer.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(writer).Encode(payload)
}

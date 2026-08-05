package store

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestJSONLAppendsWithoutReplacingPriorRecords(t *testing.T) {
	directory := t.TempDir()
	candidates := filepath.Join(directory, "candidates.jsonl")
	sessions := filepath.Join(directory, "sessions.jsonl")
	if err := os.WriteFile(candidates, []byte("{\"existing\":true}\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	writer, err := Open(candidates, sessions)
	if err != nil {
		t.Fatal(err)
	}
	if err := writer.AppendCandidate(Candidate{
		SessionID: "s", CandidateID: "c", Streamer: "example",
		DetectedAt: time.Unix(1, 0).UTC(),
	}); err != nil {
		t.Fatal(err)
	}
	if err := writer.AppendSession(SessionCounters{SessionID: "s", Streamer: "example"}); err != nil {
		t.Fatal(err)
	}
	if err := writer.Close(); err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile(candidates)
	if err != nil {
		t.Fatal(err)
	}
	lines := strings.Split(strings.TrimSpace(string(data)), "\n")
	if len(lines) != 2 || lines[0] != `{"existing":true}` {
		t.Fatalf("candidate records = %q", data)
	}
}

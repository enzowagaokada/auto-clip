package store

import (
	"encoding/csv"
	"encoding/json"
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
	reviews := filepath.Join(directory, "candidates_review.jsonl")
	reviewCSV := filepath.Join(directory, "candidates_review.csv")
	if err := os.WriteFile(candidates, []byte("{\"existing\":true}\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	writer, err := Open(candidates, sessions, reviews, reviewCSV)
	if err != nil {
		t.Fatal(err)
	}
	if err := writer.AppendCandidate(Candidate{
		SessionID: "s", CandidateID: "c", Streamer: "example",
		DetectedAt: time.Unix(1, 0).UTC(), StreamOffsetSecond: 3664.9, Score: 0.5053,
	}); err != nil {
		t.Fatal(err)
	}
	if err := writer.AppendSession(SessionCounters{
		SessionID: "s", Streamer: "example", VODID: "2840052504",
	}); err != nil {
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

	reviewData, err := os.ReadFile(reviews)
	if err != nil {
		t.Fatal(err)
	}
	var review CandidateReview
	if err := json.Unmarshal(reviewData, &review); err != nil {
		t.Fatal(err)
	}
	if review != (CandidateReview{
		CandidateID: "c", SessionID: "s", Streamer: "example", Score: 0.5053, StreamOffsetStamp: "1h1m4s",
	}) {
		t.Fatalf("review = %#v", review)
	}

	sessionData, err := os.ReadFile(sessions)
	if err != nil {
		t.Fatal(err)
	}
	var session SessionCounters
	if err := json.Unmarshal(sessionData, &session); err != nil {
		t.Fatal(err)
	}
	if session.VODID != "2840052504" {
		t.Fatalf("session vod_id = %q", session.VODID)
	}

	csvFile, err := os.Open(reviewCSV)
	if err != nil {
		t.Fatal(err)
	}
	defer csvFile.Close()
	rows, err := csv.NewReader(csvFile).ReadAll()
	if err != nil {
		t.Fatal(err)
	}
	if len(rows) != 2 {
		t.Fatalf("review CSV rows = %d, want 2", len(rows))
	}
	wantHeader := []string{
		"candidate_id", "session_id", "streamer", "score", "stream_offset_stamp", "review_label", "reason",
	}
	if strings.Join(rows[0], ",") != strings.Join(wantHeader, ",") {
		t.Fatalf("review CSV header = %v, want %v", rows[0], wantHeader)
	}
	if rows[1][0] != "c" || rows[1][1] != "s" || rows[1][2] != "example" || rows[1][4] != "1h1m4s" {
		t.Fatalf("review CSV row = %v", rows[1])
	}
	if rows[1][5] != "" || rows[1][6] != "" {
		t.Fatalf("review_label/reason should be empty for machine writes, got %v", rows[1])
	}
}

func TestReviewCSVPreservesExistingRows(t *testing.T) {
	directory := t.TempDir()
	candidates := filepath.Join(directory, "candidates.jsonl")
	sessions := filepath.Join(directory, "sessions.jsonl")
	reviews := filepath.Join(directory, "candidates_review.jsonl")
	reviewCSV := filepath.Join(directory, "candidates_review.csv")
	existing := "candidate_id,session_id,streamer,score,stream_offset_stamp,review_label,reason\n" +
		"old,sess1,arky,0.5,1h0m0s,hard_negative,stream start\n"
	if err := os.WriteFile(reviewCSV, []byte(existing), 0o644); err != nil {
		t.Fatal(err)
	}
	writer, err := Open(candidates, sessions, reviews, reviewCSV)
	if err != nil {
		t.Fatal(err)
	}
	if err := writer.AppendCandidate(Candidate{
		CandidateID: "new", SessionID: "sess2", Streamer: "marlon", StreamOffsetSecond: 44, Score: 0.4932,
	}); err != nil {
		t.Fatal(err)
	}
	if err := writer.Close(); err != nil {
		t.Fatal(err)
	}
	csvFile, err := os.Open(reviewCSV)
	if err != nil {
		t.Fatal(err)
	}
	defer csvFile.Close()
	rows, err := csv.NewReader(csvFile).ReadAll()
	if err != nil {
		t.Fatal(err)
	}
	if len(rows) != 3 {
		t.Fatalf("review CSV rows = %d, want 3", len(rows))
	}
	if rows[1][0] != "old" || rows[1][1] != "sess1" || rows[1][5] != "hard_negative" {
		t.Fatalf("existing reviewed row changed: %v", rows[1])
	}
	if rows[2][0] != "new" || rows[2][1] != "sess2" || rows[2][2] != "marlon" {
		t.Fatalf("appended row = %v", rows[2])
	}
}

func TestStreamOffsetStamp(t *testing.T) {
	if got := StreamOffsetStamp(3664.9); got != "1h1m4s" {
		t.Fatalf("StreamOffsetStamp(3664.9) = %q, want 1h1m4s", got)
	}
	if got := StreamOffsetStamp(-1); got != "0h0m0s" {
		t.Fatalf("StreamOffsetStamp(-1) = %q, want 0h0m0s", got)
	}
}

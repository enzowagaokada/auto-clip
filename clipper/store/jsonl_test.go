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
	telemetry := filepath.Join(directory, "telemetry.jsonl")
	episodes := filepath.Join(directory, "episodes.jsonl")
	episodeReviews := filepath.Join(directory, "episodes_review.jsonl")
	episodeReviewCSV := filepath.Join(directory, "episodes_review.csv")
	if err := os.WriteFile(episodes, []byte("{\"existing_episode\":true}\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	writer, err := Open(Paths{
		Candidates: candidates, Sessions: sessions,
		CandidateReviews: reviews, CandidateReviewCSV: reviewCSV,
		Telemetry: telemetry, Episodes: episodes,
		EpisodeReviews: episodeReviews, EpisodeReviewCSV: episodeReviewCSV,
	})
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
		SessionID: "s", ReviewPartition: "calibration",
		Streamer: "example", VODID: "2840052504",
	}); err != nil {
		t.Fatal(err)
	}
	if err := writer.AppendTelemetry(InferenceTelemetry{
		SchemaVersion: LiveSchemaVersion, ReviewPartition: "calibration",
		SessionID: "s", Streamer: "example",
		InferenceAt: time.Unix(2, 0).UTC(), Score: 0.7,
	}); err != nil {
		t.Fatal(err)
	}
	if err := writer.AppendEpisode(Episode{
		SchemaVersion: LiveSchemaVersion, ReviewPartition: "calibration",
		EpisodeID: "e", RecordType: "triggered",
		SessionID: "s", Streamer: "example", OnsetScore: 0.51, PeakScore: 0.8,
		PeakStreamOffset: 3665,
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
	telemetryData, err := os.ReadFile(telemetry)
	if err != nil || !strings.Contains(string(telemetryData), `"schema_version":2`) {
		t.Fatalf("telemetry record = %q, err=%v", telemetryData, err)
	}
	episodeCSVFile, err := os.Open(episodeReviewCSV)
	if err != nil {
		t.Fatal(err)
	}
	defer episodeCSVFile.Close()
	episodeRows, err := csv.NewReader(episodeCSVFile).ReadAll()
	if err != nil {
		t.Fatal(err)
	}
	if len(episodeRows) != 2 || episodeRows[1][0] != "e" ||
		episodeRows[1][1] != "triggered" ||
		episodeRows[1][2] != "calibration" || episodeRows[1][6] != "0.8" {
		t.Fatalf("episode review CSV rows = %v", episodeRows)
	}
	episodeData, err := os.ReadFile(episodes)
	if err != nil {
		t.Fatal(err)
	}
	episodeLines := strings.Split(strings.TrimSpace(string(episodeData)), "\n")
	if len(episodeLines) != 2 || episodeLines[0] != `{"existing_episode":true}` {
		t.Fatalf("episode records = %q", episodeData)
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
	writer, err := Open(Paths{
		Candidates: candidates, Sessions: sessions,
		CandidateReviews: reviews, CandidateReviewCSV: reviewCSV,
		Telemetry:        filepath.Join(directory, "telemetry.jsonl"),
		Episodes:         filepath.Join(directory, "episodes.jsonl"),
		EpisodeReviews:   filepath.Join(directory, "episodes_review.jsonl"),
		EpisodeReviewCSV: filepath.Join(directory, "episodes_review.csv"),
	})
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

func TestJSONLRejectsUnknownLiveSchema(t *testing.T) {
	directory := t.TempDir()
	writer, err := Open(Paths{
		Candidates:         filepath.Join(directory, "candidates.jsonl"),
		Sessions:           filepath.Join(directory, "sessions.jsonl"),
		CandidateReviews:   filepath.Join(directory, "candidates_review.jsonl"),
		CandidateReviewCSV: filepath.Join(directory, "candidates_review.csv"),
		Telemetry:          filepath.Join(directory, "telemetry.jsonl"),
		Episodes:           filepath.Join(directory, "episodes.jsonl"),
		EpisodeReviews:     filepath.Join(directory, "episodes_review.jsonl"),
		EpisodeReviewCSV:   filepath.Join(directory, "episodes_review.csv"),
	})
	if err != nil {
		t.Fatal(err)
	}
	defer writer.Close()
	if err := writer.AppendTelemetry(InferenceTelemetry{SchemaVersion: LiveSchemaVersion + 1}); err == nil {
		t.Fatal("AppendTelemetry() error = nil, want schema rejection")
	}
	if err := writer.AppendEpisode(Episode{SchemaVersion: LiveSchemaVersion + 1}); err == nil {
		t.Fatal("AppendEpisode() error = nil, want schema rejection")
	}
}

func TestJSONLRejectsMissingReviewPartition(t *testing.T) {
	directory := t.TempDir()
	writer, err := Open(Paths{
		Candidates:         filepath.Join(directory, "candidates.jsonl"),
		Sessions:           filepath.Join(directory, "sessions.jsonl"),
		CandidateReviews:   filepath.Join(directory, "candidates_review.jsonl"),
		CandidateReviewCSV: filepath.Join(directory, "candidates_review.csv"),
		Telemetry:          filepath.Join(directory, "telemetry.jsonl"),
		Episodes:           filepath.Join(directory, "episodes.jsonl"),
		EpisodeReviews:     filepath.Join(directory, "episodes_review.jsonl"),
		EpisodeReviewCSV:   filepath.Join(directory, "episodes_review.csv"),
	})
	if err != nil {
		t.Fatal(err)
	}
	defer writer.Close()
	if err := writer.AppendTelemetry(InferenceTelemetry{
		SchemaVersion: LiveSchemaVersion,
	}); err == nil {
		t.Fatal("AppendTelemetry() error = nil, want partition rejection")
	}
	if err := writer.AppendEpisode(Episode{
		SchemaVersion: LiveSchemaVersion,
	}); err == nil {
		t.Fatal("AppendEpisode() error = nil, want partition rejection")
	}
	if err := writer.AppendSession(SessionCounters{}); err == nil {
		t.Fatal("AppendSession() error = nil, want partition rejection")
	}
}

func TestJSONLRejectsMismatchedReviewCSVSchema(t *testing.T) {
	directory := t.TempDir()
	episodeCSV := filepath.Join(directory, "episodes_review.csv")
	if err := os.WriteFile(episodeCSV, []byte("wrong,header\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	_, err := Open(Paths{
		Candidates:         filepath.Join(directory, "candidates.jsonl"),
		Sessions:           filepath.Join(directory, "sessions.jsonl"),
		CandidateReviews:   filepath.Join(directory, "candidates_review.jsonl"),
		CandidateReviewCSV: filepath.Join(directory, "candidates_review.csv"),
		Telemetry:          filepath.Join(directory, "telemetry.jsonl"),
		Episodes:           filepath.Join(directory, "episodes.jsonl"),
		EpisodeReviews:     filepath.Join(directory, "episodes_review.jsonl"),
		EpisodeReviewCSV:   episodeCSV,
	})
	if err == nil {
		t.Fatal("Open() error = nil, want episode review schema rejection")
	}
}

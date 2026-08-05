package window

import (
	"testing"
	"time"

	"auto-clip/clipper/preprocess"
)

func TestSnapshotUsesClosedWindow(t *testing.T) {
	now := time.Unix(100, 0)
	buffer := New(35 * time.Second)
	buffer.Add(preprocess.Message{Time: now.Add(-36 * time.Second), Text: "old"})
	buffer.Add(preprocess.Message{Time: now.Add(-35 * time.Second), Text: "boundary"})
	buffer.Add(preprocess.Message{Time: now, Text: "now"})
	buffer.Add(preprocess.Message{Time: now.Add(time.Second), Text: "future"})

	got := buffer.Snapshot(now)
	if len(got) != 2 || got[0].Text != "boundary" || got[1].Text != "now" {
		t.Fatalf("snapshot = %#v", got)
	}
}

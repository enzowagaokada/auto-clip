package replay

import (
	"encoding/json"
	"testing"
	"time"
)

func TestRawWindowVODIDAcceptsStringAndNumber(t *testing.T) {
	tests := []struct {
		name     string
		vodID    string
		expected string
	}{
		{name: "string", vodID: `"2795916950"`, expected: "2795916950"},
		{name: "number", vodID: `2795916950`, expected: "2795916950"},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			var window RawWindow
			if err := json.Unmarshal([]byte(`{"vod_id":`+test.vodID+`}`), &window); err != nil {
				t.Fatalf("decode RawWindow: %v", err)
			}
			if actual := string(window.VODID); actual != test.expected {
				t.Fatalf("VODID = %q, want %q", actual, test.expected)
			}
		})
	}
}

func TestRawWindowVODIDRejectsInvalidType(t *testing.T) {
	var window RawWindow
	if err := json.Unmarshal([]byte(`{"vod_id":true}`), &window); err == nil {
		t.Fatal("expected invalid VOD ID type to fail")
	}
}

func TestExpectedGeometryUsesClipStartMinusFivePlusThirty(t *testing.T) {
	current := RawWindow{
		TargetOffset: 100,
		WindowStart:  95,
		WindowEnd:    130,
	}
	if !hasExpectedGeometry(current, 35*time.Second, 30*time.Second) {
		t.Fatal("current geometry was rejected")
	}

	old := RawWindow{
		TargetOffset: 100,
		WindowStart:  70,
		WindowEnd:    105,
	}
	if hasExpectedGeometry(old, 35*time.Second, 30*time.Second) {
		t.Fatal("old geometry was accepted")
	}
}

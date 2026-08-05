package detection

import (
	"testing"
	"time"
)

func TestCrossingCooldownAndRearm(t *testing.T) {
	machine, err := New(0.6, 10*time.Second)
	if err != nil {
		t.Fatal(err)
	}
	start := time.Unix(100, 0)
	cases := []struct {
		offset  time.Duration
		score   float32
		trigger bool
	}{
		{0, 0.5, false},
		{time.Second, 0.6, true},
		{2 * time.Second, 0.9, false},  // no crossing
		{3 * time.Second, 0.4, false},  // rearm during cooldown
		{4 * time.Second, 0.8, false},  // crossing blocked by cooldown
		{12 * time.Second, 0.4, false}, // return below after cooldown
		{13 * time.Second, 0.8, true},
	}
	for _, test := range cases {
		got := machine.Observe(start.Add(test.offset), test.score)
		if got.Triggered != test.trigger {
			t.Errorf("at %s score %.2f: triggered=%v, want %v",
				test.offset, test.score, got.Triggered, test.trigger)
		}
	}
}

func TestSustainedHighDoesNotRetriggerAfterCooldown(t *testing.T) {
	machine, _ := New(0.5, time.Second)
	start := time.Unix(100, 0)
	if !machine.Observe(start, 0.7).Triggered {
		t.Fatal("initial crossing did not trigger")
	}
	if machine.Observe(start.Add(2*time.Second), 0.7).Triggered {
		t.Fatal("sustained high score retriggered without below-threshold rearm")
	}
}

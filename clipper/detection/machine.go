package detection

import (
	"errors"
	"sync"
	"time"
)

type Decision struct {
	Triggered     bool
	Crossed       bool
	Score         float32
	Threshold     float32
	CooldownUntil time.Time
}

// Machine emits only on a below-to-at/above threshold crossing. A trigger
// starts cooldown and disarms the machine; observing a below-threshold score
// rearms it, while cooldown remains an independent gate.
type Machine struct {
	mu            sync.Mutex
	threshold     float32
	cooldown      time.Duration
	above         bool
	armed         bool
	cooldownUntil time.Time
}

func New(threshold float32, cooldown time.Duration) (*Machine, error) {
	if threshold < 0 || threshold > 1 {
		return nil, errors.New("threshold must be in [0, 1]")
	}
	if cooldown < 0 {
		return nil, errors.New("cooldown must be non-negative")
	}
	return &Machine{threshold: threshold, cooldown: cooldown, armed: true}, nil
}

func (m *Machine) Observe(at time.Time, score float32) Decision {
	m.mu.Lock()
	defer m.mu.Unlock()

	isAbove := score >= m.threshold
	crossed := isAbove && !m.above
	if !isAbove {
		m.armed = true
	}
	triggered := crossed && m.armed && !at.Before(m.cooldownUntil)
	if triggered {
		m.armed = false
		m.cooldownUntil = at.Add(m.cooldown)
	}
	m.above = isAbove
	return Decision{
		Triggered:     triggered,
		Crossed:       crossed,
		Score:         score,
		Threshold:     m.threshold,
		CooldownUntil: m.cooldownUntil,
	}
}

func (m *Machine) Reset() {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.above = false
	m.armed = true
	m.cooldownUntil = time.Time{}
}

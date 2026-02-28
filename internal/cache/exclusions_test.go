package cache

import (
	"testing"
)

func TestExclusionList_NilSafe(t *testing.T) {
	var el *ExclusionList
	if el.Matches("gpt-4o") {
		t.Fatal("nil ExclusionList must never match")
	}
	if el.Len() != 0 {
		t.Fatal("nil ExclusionList Len must be 0")
	}
}

func TestExclusionList_ExactMatch(t *testing.T) {
	el, err := NewExclusionList([]string{"gpt-4o", "gemini-pro"}, nil)
	if err != nil {
		t.Fatal(err)
	}

	cases := []struct {
		model string
		want  bool
	}{
		{"gpt-4o", true},
		{"gemini-pro", true},
		{"gpt-4-turbo", false},   // different model
		{"GPT-4O", false},         // case-sensitive
		{"gpt-4", false},          // prefix only
		{"claude-3-5-sonnet", false},
	}
	for _, c := range cases {
		if got := el.Matches(c.model); got != c.want {
			t.Errorf("Matches(%q) = %v, want %v", c.model, got, c.want)
		}
	}
}

func TestExclusionList_RegexMatch(t *testing.T) {
	el, err := NewExclusionList(nil, []string{`^gpt-4`, `claude-3-opus`})
	if err != nil {
		t.Fatal(err)
	}

	cases := []struct {
		model string
		want  bool
	}{
		{"gpt-4o", true},
		{"gpt-4-turbo", true},
		{"gpt-4", true},
		{"claude-3-opus", true},
		{"claude-3-5-sonnet", false}, // doesn't match either pattern
		{"gpt-3.5-turbo", false},
		{"gemini-1.5-pro", false},
	}
	for _, c := range cases {
		if got := el.Matches(c.model); got != c.want {
			t.Errorf("Matches(%q) = %v, want %v", c.model, got, c.want)
		}
	}
}

func TestExclusionList_ExactBeatsRegex(t *testing.T) {
	// Both exact and regex configured; exact should still work.
	el, err := NewExclusionList(
		[]string{"mistral-large"},
		[]string{`^gpt-4`},
	)
	if err != nil {
		t.Fatal(err)
	}

	if !el.Matches("mistral-large") {
		t.Error("exact match missed")
	}
	if !el.Matches("gpt-4o") {
		t.Error("regex match missed")
	}
	if el.Matches("mistral-medium") {
		t.Error("should not match")
	}
}

func TestExclusionList_InvalidPattern(t *testing.T) {
	_, err := NewExclusionList(nil, []string{`[invalid(`})
	if err == nil {
		t.Fatal("expected error for invalid regex")
	}
}

func TestExclusionList_EmptyStringsSkipped(t *testing.T) {
	el, err := NewExclusionList([]string{"", "gpt-4o", ""}, []string{"", `^claude`})
	if err != nil {
		t.Fatal(err)
	}
	if !el.Matches("gpt-4o") {
		t.Error("should match gpt-4o")
	}
	if !el.Matches("claude-3-opus") {
		t.Error("should match claude-3-opus via regex")
	}
	if el.Len() != 2 { // 1 exact + 1 regex
		t.Errorf("Len = %d, want 2", el.Len())
	}
}

package cache

import (
	"fmt"
	"regexp"
)

// ExclusionList decides whether a given model name should be excluded from
// caching. It supports two matching modes:
//
//   - Exact match: the model string must equal the rule exactly.
//   - Regex match: the model string is tested against a compiled regexp.
//
// A nil *ExclusionList is safe to call — Matches always returns false.
type ExclusionList struct {
	exact    map[string]struct{}
	patterns []*regexp.Regexp
}

// NewExclusionList compiles the given exact strings and regex patterns into an
// ExclusionList. Returns an error if any pattern fails to compile so that
// misconfiguration is caught at startup.
func NewExclusionList(exact, patterns []string) (*ExclusionList, error) {
	el := &ExclusionList{
		exact: make(map[string]struct{}, len(exact)),
	}

	for _, e := range exact {
		if e != "" {
			el.exact[e] = struct{}{}
		}
	}

	for _, p := range patterns {
		if p == "" {
			continue
		}
		re, err := regexp.Compile(p)
		if err != nil {
			return nil, fmt.Errorf("cache exclusion: invalid pattern %q: %w", p, err)
		}
		el.patterns = append(el.patterns, re)
	}

	return el, nil
}

// Matches reports whether the given model name is excluded from caching.
// Exact rules are checked first (O(1)), then regex patterns in order.
func (el *ExclusionList) Matches(model string) bool {
	if el == nil {
		return false
	}
	if _, ok := el.exact[model]; ok {
		return true
	}
	for _, re := range el.patterns {
		if re.MatchString(model) {
			return true
		}
	}
	return false
}

// Len returns the total number of exclusion rules configured.
func (el *ExclusionList) Len() int {
	if el == nil {
		return 0
	}
	return len(el.exact) + len(el.patterns)
}

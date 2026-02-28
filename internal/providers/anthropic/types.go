package anthropic

type messagesRequest struct {
	Model       string       `json:"model"`
	Messages    []apiMessage `json:"messages"`
	System      string       `json:"system,omitempty"`
	MaxTokens   int          `json:"max_tokens"`
	Stream      bool         `json:"stream,omitempty"`
	Temperature float64      `json:"temperature,omitempty"`
}

type apiMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type messagesResponse struct {
	ID      string         `json:"id"`
	Model   string         `json:"model"`
	Content []contentBlock `json:"content"`
	Usage   apiUsage       `json:"usage"`
}

type contentBlock struct {
	Type string `json:"type"`
	Text string `json:"text"`
}

type apiUsage struct {
	InputTokens  int `json:"input_tokens"`
	OutputTokens int `json:"output_tokens"`
}

// SSE event types for streaming
type streamEvent struct {
	Type  string       `json:"type"`
	Delta *streamDelta `json:"delta,omitempty"`
	Index int          `json:"index"`
}

type streamDelta struct {
	Type string `json:"type"`
	Text string `json:"text"`
}

type apiError struct {
	Type  string        `json:"type"`
	Error *apiErrDetail `json:"error"`
}

type apiErrDetail struct {
	Type    string `json:"type"`
	Message string `json:"message"`
}

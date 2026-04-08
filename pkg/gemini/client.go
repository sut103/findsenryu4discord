package gemini

import (
	"context"
	"encoding/json"
	"fmt"

	"github.com/google/generative-ai-go/genai"
	"google.golang.org/api/option"
)

// Client is a wrapper for Gemini API client
type Client struct {
	client       *genai.Client
	model        *genai.GenerativeModel
	systemPrompt string
}

// ScoreResult represents the output from Gemini
type ScoreResult struct {
	Score   int    `json:"score"`
	Comment string `json:"comment"`
}

// NewClient creates a new Gemini client
func NewClient(ctx context.Context, apiKey, modelName, systemPrompt string) (*Client, error) {
	if apiKey == "" {
		return nil, fmt.Errorf("gemini api key is empty")
	}

	client, err := genai.NewClient(ctx, option.WithAPIKey(apiKey))
	if err != nil {
		return nil, err
	}

	model := client.GenerativeModel(modelName)
	// Enable JSON response
	model.ResponseMIMEType = "application/json"
	model.SystemInstruction = genai.NewUserContent(genai.Text(systemPrompt))

	return &Client{
		client:       client,
		model:        model,
		systemPrompt: systemPrompt,
	}, nil
}

// Close closes the underlying client
func (c *Client) Close() {
	if c.client != nil {
		c.client.Close()
	}
}

// ScoreSenryu evaluates a senryu and returns a score and comment
func (c *Client) ScoreSenryu(ctx context.Context, senryuText string) (*ScoreResult, error) {
	resp, err := c.model.GenerateContent(ctx, genai.Text(senryuText))
	if err != nil {
		return nil, fmt.Errorf("failed to generate content: %w", err)
	}

	if len(resp.Candidates) == 0 || len(resp.Candidates[0].Content.Parts) == 0 {
		return nil, fmt.Errorf("empty response from gemini")
	}

	part := resp.Candidates[0].Content.Parts[0]
	text, ok := part.(genai.Text)
	if !ok {
		return nil, fmt.Errorf("unexpected response type from gemini")
	}

	var result ScoreResult
	if err := json.Unmarshal([]byte(text), &result); err != nil {
		return nil, fmt.Errorf("failed to parse json response: %w", err)
	}

	return &result, nil
}
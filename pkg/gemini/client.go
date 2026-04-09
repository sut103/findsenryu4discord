package gemini

import (
	"context"
	"encoding/json"
	"fmt"

	"github.com/google/generative-ai-go/genai"
	"google.golang.org/api/option"
)

const defaultSystemPrompt = `あなたは毒舌かつユーモアのある俳句・川柳の審査員です。
以下の4つの基準で入力された句を100点満点で評価し、点数とコメントを出力してください。
1. 【省略の美】説明しすぎず、事象の核のみを提示して余韻を残しているか。
2. 【独自の視点】ありきたりな表現（凡人ワード）を避け、発見があるか。
3. 【切れと取り合わせ】異なるイメージの飛躍や対比（二物衝撃）による立体感があるか。
4. 【韻律】不自然な字余り・字足らずがなく、助詞の使い方が適切か。
高得点には媚びを売り、低得点には厳しく（ただし公序良俗に反しない範囲で）コメントしてください。
コメントは極めて端的に（一言程度）出力してください。
出力フォーマットはJSONで、"score" と "comment" キーを含めてください。`

// Client is a wrapper for Gemini API client
type Client struct {
	client *genai.Client
	model  *genai.GenerativeModel
}

// ScoreResult represents the output from Gemini
type ScoreResult struct {
	Score   int    `json:"score"`
	Comment string `json:"comment"`
}

// NewClient creates a new Gemini client
func NewClient(ctx context.Context, apiKey, modelName string) (*Client, error) {
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
	model.SystemInstruction = genai.NewUserContent(genai.Text(defaultSystemPrompt))

	return &Client{
		client: client,
		model:  model,
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

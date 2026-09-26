package agent

import (
	"context"
	"fmt"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"charm.land/catwalk/pkg/catwalk"
	"charm.land/fantasy"
	"github.com/charmbracelet/crush/internal/agent/notify"
	"github.com/charmbracelet/crush/internal/db"
	"github.com/charmbracelet/crush/internal/message"
	"github.com/charmbracelet/crush/internal/pubsub"
	"github.com/charmbracelet/crush/internal/session"
	"github.com/stretchr/testify/require"
)

// xmemoryBoundaryModel substitutes only the provider. SessionAgent, its queue,
// message persistence, summarization and terminal notifications remain real.
type xmemoryBoundaryModel struct {
	fastModel
	stream func(context.Context, fantasy.Call) (fantasy.StreamResponse, error)
}

func (m *xmemoryBoundaryModel) Stream(ctx context.Context, call fantasy.Call) (fantasy.StreamResponse, error) {
	return m.stream(ctx, call)
}

func xmemoryBoundaryStream(text string, tool bool, usage fantasy.Usage) fantasy.StreamResponse {
	return func(yield func(fantasy.StreamPart) bool) {
		if tool {
			if !yield(fantasy.StreamPart{Type: fantasy.StreamPartTypeToolCall, ID: "probe-call", ToolCallName: "probe", ToolCallInput: `{}`}) {
				return
			}
			yield(fantasy.StreamPart{Type: fantasy.StreamPartTypeFinish, FinishReason: fantasy.FinishReasonToolCalls, Usage: usage})
			return
		}
		for _, part := range []fantasy.StreamPart{
			{Type: fantasy.StreamPartTypeTextStart, ID: "text"},
			{Type: fantasy.StreamPartTypeTextDelta, ID: "text", Delta: text},
			{Type: fantasy.StreamPartTypeTextEnd, ID: "text"},
			{Type: fantasy.StreamPartTypeFinish, FinishReason: fantasy.FinishReasonStop, Usage: usage},
		} {
			if !yield(part) {
				return
			}
		}
	}
}

func xmemoryBoundaryAgent(t *testing.T, model fantasy.LanguageModel, window int64, tools ...fantasy.AgentTool) (*sessionAgent, session.Service, message.Service, string, <-chan pubsub.Event[notify.RunComplete]) {
	t.Helper()
	conn, err := db.Connect(t.Context(), t.TempDir())
	require.NoError(t, err)
	t.Cleanup(func() { require.NoError(t, conn.Close()) })
	queries := db.New(conn)
	sessions := session.NewService(queries, conn)
	messages := message.NewService(queries)
	created, err := sessions.Create(t.Context(), "Synthetic lifecycle contract")
	require.NoError(t, err)
	broker := pubsub.NewBroker[notify.RunComplete]()
	t.Cleanup(broker.Shutdown)
	events := broker.Subscribe(t.Context())
	sa := NewSessionAgent(SessionAgentOptions{
		LargeModel:  Model{Model: model, CatwalkCfg: catwalk.Model{ContextWindow: window, DefaultMaxTokens: 100}},
		SmallModel:  Model{Model: fastModel{}, CatwalkCfg: catwalk.Model{ContextWindow: 200000, DefaultMaxTokens: 100}},
		IsYolo:      true,
		Sessions:    sessions,
		Messages:    messages,
		RunComplete: broker,
		Tools:       tools,
	}).(*sessionAgent)
	return sa, sessions, messages, created.ID, events
}

func xmemoryBoundaryTerminal(t *testing.T, events <-chan pubsub.Event[notify.RunComplete], runID string) notify.RunComplete {
	t.Helper()
	select {
	case event := <-events:
		require.Equal(t, runID, event.Payload.RunID)
		require.Empty(t, event.Payload.Error)
		require.False(t, event.Payload.Cancelled)
		return event.Payload
	case <-time.After(5 * time.Second):
		t.Fatal("No terminal event arrived")
		return notify.RunComplete{}
	}
}

func xmemoryBoundaryNoExtraTerminal(t *testing.T, events <-chan pubsub.Event[notify.RunComplete]) {
	t.Helper()
	select {
	case event := <-events:
		t.Fatalf("Unexpected additional terminal event: %+v", event.Payload)
	case <-time.After(100 * time.Millisecond):
	}
}

func xmemoryBoundaryPromptText(prompt fantasy.Prompt) string {
	var text strings.Builder
	for _, msg := range prompt {
		for _, part := range msg.Content {
			if part, ok := part.(fantasy.TextPart); ok {
				text.WriteString(part.Text)
				text.WriteByte('\n')
			}
		}
	}
	return text.String()
}

func TestXmemoryBoundaryEqualInputsKeepDistinctPersistedIdentities(t *testing.T) {
	t.Parallel()
	model := &xmemoryBoundaryModel{stream: func(context.Context, fantasy.Call) (fantasy.StreamResponse, error) {
		return xmemoryBoundaryStream("done", false, fantasy.Usage{InputTokens: 1, OutputTokens: 1, TotalTokens: 2}), nil
	}}
	sa, _, messages, sessionID, events := xmemoryBoundaryAgent(t, model, 200000)
	for i := range 2 {
		runID := fmt.Sprintf("run-%d", i)
		_, err := sa.Run(t.Context(), SessionAgentCall{SessionID: sessionID, RunID: runID, Prompt: "same user input"})
		require.NoError(t, err)
		terminal := xmemoryBoundaryTerminal(t, events, runID)
		stored, err := messages.Get(t.Context(), terminal.MessageID)
		require.NoError(t, err)
		require.Equal(t, terminal.Text, stored.Content().String())
	}
	users, err := messages.ListUserMessages(t.Context(), sessionID)
	require.NoError(t, err)
	require.Len(t, users, 2)
	require.Equal(t, users[0].Content().String(), users[1].Content().String())
	require.NotEqual(t, users[0].ID, users[1].ID, "Equal text must not become one source identity")
	xmemoryBoundaryNoExtraTerminal(t, events)
	t.Log("Observed two real Run calls, two distinct persisted user IDs, and one matching terminal event per RunID")
}

func TestXmemoryBoundaryFollowupFoldsAtNextActualStep(t *testing.T) {
	t.Parallel()
	var calls, executions atomic.Int64
	var sa *sessionAgent
	var sessionID string
	var messages message.Service
	var followedIDs []string
	model := &xmemoryBoundaryModel{stream: func(ctx context.Context, call fantasy.Call) (fantasy.StreamResponse, error) {
		if calls.Add(1) == 1 {
			queued, err := sa.Run(ctx, SessionAgentCall{SessionID: sessionID, Prompt: "same user input"})
			require.NoError(t, err)
			require.Nil(t, queued)
			require.Equal(t, 1, sa.QueuedPrompts(sessionID))
			users, err := messages.ListUserMessages(ctx, sessionID)
			require.NoError(t, err)
			require.Len(t, users, 1, "Queued text is not persisted before its actual step")
			return xmemoryBoundaryStream("", true, fantasy.Usage{InputTokens: 1, OutputTokens: 1, TotalTokens: 2}), nil
		}
		users, err := messages.ListUserMessages(ctx, sessionID)
		require.NoError(t, err)
		require.Len(t, users, 2)
		for _, user := range users {
			followedIDs = append(followedIDs, user.ID)
		}
		occurrences := 0
		for _, msg := range call.Prompt {
			if msg.Role != fantasy.MessageRoleUser {
				continue
			}
			for _, part := range msg.Content {
				if text, ok := part.(fantasy.TextPart); ok && text.Text == "same user input" {
					occurrences++
				}
			}
		}
		require.Equal(t, 2, occurrences, "Actual second provider step must receive both persisted inputs")
		return xmemoryBoundaryStream("folded-done", false, fantasy.Usage{InputTokens: 1, OutputTokens: 1, TotalTokens: 2}), nil
	}}
	tool := fantasy.NewAgentTool("probe", "Record one local test effect.", func(context.Context, struct{}, fantasy.ToolCall) (fantasy.ToolResponse, error) {
		executions.Add(1)
		return fantasy.NewTextResponse("probe-done"), nil
	})
	var events <-chan pubsub.Event[notify.RunComplete]
	sa, _, messages, sessionID, events = xmemoryBoundaryAgent(t, model, 200000, tool)
	_, err := sa.Run(t.Context(), SessionAgentCall{SessionID: sessionID, RunID: "main-run", Prompt: "same user input"})
	require.NoError(t, err)
	require.EqualValues(t, 2, calls.Load())
	require.EqualValues(t, 1, executions.Load())
	require.Len(t, followedIDs, 2)
	require.NotEqual(t, followedIDs[0], followedIDs[1])
	require.Equal(t, "folded-done", xmemoryBoundaryTerminal(t, events, "main-run").Text)
	xmemoryBoundaryNoExtraTerminal(t, events)
	t.Log("Observed queued followup absent from DB until step 2, then two user IDs in one run and one terminal event")
}

func TestXmemoryBoundarySummaryContinuationIsNotNewUserInput(t *testing.T) {
	t.Parallel()
	for _, runID := range []string{"", "summary-run"} {
		name := "without_external_run_id"
		if runID != "" {
			name = "with_external_run_id"
		}
		t.Run(name, func(t *testing.T) {
			t.Parallel()
			xmemoryBoundarySummaryContinuation(t, runID)
		})
	}
}

func xmemoryBoundarySummaryContinuation(t *testing.T, runID string) {
	t.Helper()
	var calls, executions atomic.Int64
	var promptMu sync.Mutex
	var actualPrompts []fantasy.Prompt
	model := &xmemoryBoundaryModel{stream: func(_ context.Context, call fantasy.Call) (fantasy.StreamResponse, error) {
		promptMu.Lock()
		actualPrompts = append(actualPrompts, call.Prompt)
		promptMu.Unlock()
		switch calls.Add(1) {
		case 1:
			return xmemoryBoundaryStream("", true, fantasy.Usage{InputTokens: 1950, OutputTokens: 10, TotalTokens: 1960}), nil
		case 2:
			return xmemoryBoundaryStream("synthetic summary", false, fantasy.Usage{InputTokens: 20, OutputTokens: 5, TotalTokens: 25}), nil
		case 3:
			return xmemoryBoundaryStream("resumed-done", false, fantasy.Usage{InputTokens: 20, OutputTokens: 5, TotalTokens: 25}), nil
		default:
			return nil, fmt.Errorf("Unexpected extra provider call")
		}
	}}
	tool := fantasy.NewAgentTool("probe", "Record one local test effect.", func(context.Context, struct{}, fantasy.ToolCall) (fantasy.ToolResponse, error) {
		executions.Add(1)
		return fantasy.NewTextResponse("probe-done"), nil
	})
	sa, sessions, messages, sessionID, events := xmemoryBoundaryAgent(t, model, 2000, tool)
	_, err := sa.Run(t.Context(), SessionAgentCall{SessionID: sessionID, RunID: runID, Prompt: "original request"})
	require.NoError(t, err)
	require.EqualValues(t, 3, calls.Load(), "Expected task step, actual summarize call, and resumed step")
	require.EqualValues(t, 1, executions.Load())
	terminal := xmemoryBoundaryTerminal(t, events, runID)
	require.Equal(t, "resumed-done", terminal.Text)
	xmemoryBoundaryNoExtraTerminal(t, events)
	users, err := messages.ListUserMessages(t.Context(), sessionID)
	require.NoError(t, err)
	require.Len(t, users, 2)
	originals, continuations := 0, 0
	for _, user := range users {
		text := user.Content().String()
		if text == "original request" {
			originals++
		}
		if strings.Contains(text, "The previous session was interrupted because it got too long") {
			continuations++
		}
	}
	require.Equal(t, 1, originals)
	require.Equal(t, 1, continuations, "Current Crush persists a synthetic continuation with role=user; S2 must explicitly exclude it from capture")
	storedSession, err := sessions.Get(t.Context(), sessionID)
	require.NoError(t, err)
	summary, err := messages.Get(t.Context(), storedSession.SummaryMessageID)
	require.NoError(t, err)
	require.True(t, summary.IsSummaryMessage)
	require.Equal(t, "synthetic summary", summary.Content().String())
	storedFinal, err := messages.Get(t.Context(), terminal.MessageID)
	require.NoError(t, err)
	require.Equal(t, terminal.Text, storedFinal.Content().String())
	require.NotEqual(t, summary.ID, terminal.MessageID)
	promptMu.Lock()
	defer promptMu.Unlock()
	require.Len(t, actualPrompts, 3)
	require.Contains(t, xmemoryBoundaryPromptText(actualPrompts[1]), "original request")
	require.Contains(t, xmemoryBoundaryPromptText(actualPrompts[2]), "synthetic summary")
	require.Contains(t, xmemoryBoundaryPromptText(actualPrompts[2]), "The previous session was interrupted because it got too long")
	t.Log("Observed tool step -> real summary -> synthetic continuation, one tool effect, and one terminal under the original RunID")
	t.Log("Boundary only: xmemory input/turn envelope and synthetic-source exclusion still require S2 implementation")
}

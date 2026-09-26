package memory

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"charm.land/fantasy"
	"charm.land/fantasy/providers/openai"
	"charm.land/fantasy/providers/openaicompat"
	"github.com/charmbracelet/openai-go/option"
	"github.com/stretchr/testify/require"
)

// These S0 contract probes use real provider adapters, not production memory
// integration. Only the dedicated HTTP/1 transport below is certified here.
// Pooled HTTP/2, OAuth/Copilot transports, and other providers need their own
// send-boundary verification before memory injection can be enabled there.
var errContractPackObsolete = errors.New("memory pack became obsolete before send")

type contractPackKey struct{}
type contractStepKey struct{}

type contractStep struct {
	generation int64
	observed   bool
}

type contractGuard struct {
	next       http.RoundTripper
	generation atomic.Int64
	checks     atomic.Int64
	blocked    atomic.Int64
}

func (g *contractGuard) RoundTrip(req *http.Request) (*http.Response, error) {
	g.checks.Add(1)
	if generation, ok := req.Context().Value(contractPackKey{}).(int64); ok && generation != g.generation.Load() {
		g.blocked.Add(1)
		if req.Body != nil {
			_ = req.Body.Close()
		}
		return nil, errContractPackObsolete
	}
	return g.next.RoundTrip(req)
}

// A fresh HTTP/1 connection cannot enter net/http's reused-connection replay
// path. Ordinary SDK/fantasy retries still pass through this guard each time.
func newContractClient(t *testing.T) (*http.Client, *contractGuard) {
	t.Helper()
	transport := http.DefaultTransport.(*http.Transport).Clone()
	transport.DisableKeepAlives = true
	transport.Protocols = new(http.Protocols)
	transport.Protocols.SetHTTP1(true)
	transport.ForceAttemptHTTP2 = false
	guard := &contractGuard{next: transport}
	guard.generation.Store(1)
	t.Cleanup(transport.CloseIdleConnections)
	return &http.Client{Transport: guard}, guard
}

func newContractModel(t *testing.T, protocol, endpoint string, client *http.Client, sdkRetries int) fantasy.LanguageModel {
	t.Helper()
	var provider fantasy.Provider
	var err error
	switch protocol {
	case "chat":
		provider, err = openaicompat.New(
			openaicompat.WithBaseURL(endpoint+"/v1/"),
			openaicompat.WithAPIKey("local-contract-key"),
			openaicompat.WithHTTPClient(client),
			openaicompat.WithSDKOptions(option.WithMaxRetries(sdkRetries)),
		)
	case "responses":
		provider, err = openai.New(
			openai.WithBaseURL(endpoint+"/v1/"),
			openai.WithAPIKey("local-contract-key"),
			openai.WithHTTPClient(client),
			openai.WithUseResponsesAPI(),
			openai.WithResponsesAPIFunc(func(string) bool { return true }),
			openai.WithSDKOptions(option.WithMaxRetries(sdkRetries)),
		)
	default:
		t.Fatalf("Unknown contract protocol %q", protocol)
	}
	require.NoError(t, err)
	model, err := provider.LanguageModel(context.Background(), "contract-model")
	require.NoError(t, err)
	return model
}

// contractModel rebuilds only a language-model request, never Agent.Stream or
// SessionAgent.Run. S2 must move equivalent behavior into production code and
// cover Generate/Object calls separately if those paths receive memory.
type contractModel struct {
	fantasy.LanguageModel
	guard    *contractGuard
	rebuilds atomic.Int64
}

func (m *contractModel) Stream(ctx context.Context, call fantasy.Call) (fantasy.StreamResponse, error) {
	step, ok := ctx.Value(contractStepKey{}).(*contractStep)
	if !ok {
		step = &contractStep{generation: m.guard.generation.Load()}
	}
	return func(yield func(fantasy.StreamPart) bool) {
		for attempt := range 2 {
			// Keep the caller's canonical prompt untouched across attempts.
			prepared := call
			prepared.Prompt = append([]fantasy.Message(nil), call.Prompt...)
			prepared.Prompt = append(prepared.Prompt, fantasy.NewSystemMessage(fmt.Sprintf("memory-generation-%d", step.generation)))
			attemptCtx := context.WithValue(ctx, contractPackKey{}, step.generation)
			stream, err := m.LanguageModel.Stream(attemptCtx, prepared)
			if err == nil {
				for part := range stream {
					if part.Type == fantasy.StreamPartTypeError {
						err = part.Error
						break
					}
					if part.Type != fantasy.StreamPartTypeWarnings && part.Type != fantasy.StreamPartTypeKeepalive {
						step.observed = true
					}
					if !yield(part) {
						return
					}
				}
			}
			if err == nil {
				return
			}
			if errors.Is(err, errContractPackObsolete) {
				if !step.observed && attempt == 0 && ctx.Err() == nil {
					m.rebuilds.Add(1)
					step.generation = m.guard.generation.Load()
					continue
				}
				// Strip the *url.Error net.Error wrapper so fantasy cannot
				// mistake a deterministic guard failure for a network retry.
				err = errContractPackObsolete
			}
			yield(fantasy.StreamPart{Type: fantasy.StreamPartTypeError, Error: err})
			return
		}
	}, nil
}

func (m *contractModel) prepareStep(ctx context.Context, options fantasy.PrepareStepFunctionOptions) (context.Context, fantasy.PrepareStepResult, error) {
	step := &contractStep{generation: m.guard.generation.Load()}
	return context.WithValue(ctx, contractStepKey{}, step), fantasy.PrepareStepResult{Messages: options.Messages}, nil
}

type contractRequests struct {
	sync.Mutex
	bodies []string
}

func (r *contractRequests) record(t *testing.T, req *http.Request) int {
	t.Helper()
	body, err := io.ReadAll(req.Body)
	if !assertContractRequest(t, req, err) {
		return 0
	}
	r.Lock()
	defer r.Unlock()
	r.bodies = append(r.bodies, string(body))
	return len(r.bodies)
}

func assertContractRequest(t *testing.T, req *http.Request, err error) bool {
	t.Helper()
	if err != nil || req.ProtoMajor != 1 || req.Method != http.MethodPost {
		t.Errorf("Unexpected request: protocol=%s method=%s error=%v", req.Proto, req.Method, err)
		return false
	}
	return true
}

func (r *contractRequests) snapshot() []string {
	r.Lock()
	defer r.Unlock()
	return append([]string(nil), r.bodies...)
}

func writeContractText(w http.ResponseWriter, protocol string) {
	w.Header().Set("Content-Type", "text/event-stream")
	if protocol == "responses" {
		fmt.Fprint(w, "data: {\"type\":\"response.output_item.added\",\"output_index\":0,\"item\":{\"type\":\"message\",\"id\":\"msg-1\",\"role\":\"assistant\"}}\n\n")
		fmt.Fprint(w, "data: {\"type\":\"response.output_text.delta\",\"item_id\":\"msg-1\",\"output_index\":0,\"content_index\":0,\"delta\":\"ok\"}\n\n")
		fmt.Fprint(w, "data: {\"type\":\"response.output_item.done\",\"output_index\":0,\"item\":{\"type\":\"message\",\"id\":\"msg-1\",\"role\":\"assistant\"}}\n\n")
		fmt.Fprint(w, "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp-1\",\"status\":\"completed\",\"output\":[],\"usage\":{\"input_tokens\":1,\"output_tokens\":1,\"total_tokens\":2}}}\n\n")
		return
	}
	fmt.Fprint(w, "data: {\"id\":\"completion-1\",\"object\":\"chat.completion.chunk\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"ok\"},\"finish_reason\":null}]}\n\n")
	fmt.Fprint(w, "data: {\"id\":\"completion-1\",\"object\":\"chat.completion.chunk\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n")
}

func TestGuardContractErrorSurvivesProviderStream(t *testing.T) {
	t.Parallel()
	for _, protocol := range []string{"chat", "responses"} {
		t.Run(protocol, func(t *testing.T) {
			t.Parallel()
			client, guard := newContractClient(t)
			var requests atomic.Int64
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
				requests.Add(1)
				writeContractText(w, protocol)
			}))
			defer server.Close()
			model := newContractModel(t, protocol, server.URL, client, 0)
			guard.generation.Store(2)
			ctx := context.WithValue(context.Background(), contractPackKey{}, int64(1))
			stream, err := model.Stream(ctx, fantasy.Call{Prompt: []fantasy.Message{fantasy.NewUserMessage("hello")}})
			require.NoError(t, err, "OpenAI returns transport errors through its iterator")
			var streamErr error
			for part := range stream {
				if part.Type == fantasy.StreamPartTypeError {
					streamErr = part.Error
				}
			}
			require.ErrorIs(t, streamErr, errContractPackObsolete)
			var urlErr *url.Error
			require.ErrorAs(t, streamErr, &urlErr)
			var netErr net.Error
			require.ErrorAs(t, streamErr, &netErr, "The unnormalized error looks retryable to fantasy")
			require.EqualValues(t, 0, requests.Load())
			require.EqualValues(t, 1, guard.checks.Load())
			require.EqualValues(t, 1, guard.blocked.Load())
		})
	}
}

func TestGuardContractAuthRefreshReassemblesOnlyModelRequest(t *testing.T) {
	t.Parallel()
	for _, protocol := range []string{"chat", "responses"} {
		t.Run(protocol, func(t *testing.T) {
			t.Parallel()
			client, guard := newContractClient(t)
			var requests contractRequests
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
				if requests.record(t, req) == 1 {
					guard.generation.Store(2)
					w.Header().Set("Content-Type", "application/json")
					w.WriteHeader(http.StatusUnauthorized)
					fmt.Fprint(w, `{"error":{"message":"expired","type":"authentication_error"}}`)
					return
				}
				writeContractText(w, protocol)
			}))
			defer server.Close()
			model := &contractModel{LanguageModel: newContractModel(t, protocol, server.URL, client, 0), guard: guard}
			var preparations, refreshes, selections int
			ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
			defer cancel()
			result, err := fantasy.NewAgent(model).Stream(ctx, fantasy.AgentStreamCall{
				Prompt: "hello",
				PrepareStep: func(ctx context.Context, options fantasy.PrepareStepFunctionOptions) (context.Context, fantasy.PrepareStepResult, error) {
					preparations++
					return model.prepareStep(ctx, options)
				},
				OnAuthRefresh: func(context.Context, *fantasy.ProviderError) error {
					refreshes++
					return nil
				},
				ModelProvider: func() fantasy.LanguageModel { selections++; return model },
			})
			require.NoError(t, err)
			require.Equal(t, "ok", result.Response.Content.Text())
			require.Equal(t, 1, preparations, "PrepareStep does not cover the auth retry")
			require.Equal(t, 1, refreshes)
			require.Equal(t, 2, selections)
			require.EqualValues(t, 3, guard.checks.Load())
			require.EqualValues(t, 1, guard.blocked.Load())
			require.EqualValues(t, 1, model.rebuilds.Load())
			bodies := requests.snapshot()
			require.Len(t, bodies, 2)
			require.Contains(t, bodies[0], "memory-generation-1")
			require.Contains(t, bodies[1], "memory-generation-2")
			require.NotContains(t, bodies[1], "memory-generation-1")
		})
	}
}

func TestGuardContractSDKRetryCannotSendObsoletePack(t *testing.T) {
	t.Parallel()
	for _, protocol := range []string{"chat", "responses"} {
		t.Run(protocol, func(t *testing.T) {
			t.Parallel()
			client, guard := newContractClient(t)
			var requests contractRequests
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
				if requests.record(t, req) == 1 {
					guard.generation.Store(2)
					w.Header().Set("Content-Type", "application/json")
					w.Header().Set("Retry-After-Ms", "1")
					w.WriteHeader(http.StatusTooManyRequests)
					fmt.Fprint(w, `{"error":{"message":"retry","type":"rate_limit_error"}}`)
					return
				}
				writeContractText(w, protocol)
			}))
			defer server.Close()
			// Fantasy normally disables SDK retries. Enabling one here proves
			// that an inner SDK attempt still passes through the actual guard.
			model := &contractModel{LanguageModel: newContractModel(t, protocol, server.URL, client, 1), guard: guard}
			ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
			defer cancel()
			result, err := fantasy.NewAgent(model).Stream(ctx, fantasy.AgentStreamCall{Prompt: "hello", PrepareStep: model.prepareStep})
			require.NoError(t, err)
			require.Equal(t, "ok", result.Response.Content.Text())
			require.EqualValues(t, 3, guard.checks.Load())
			require.EqualValues(t, 1, guard.blocked.Load())
			bodies := requests.snapshot()
			require.Len(t, bodies, 2)
			require.Contains(t, bodies[1], "memory-generation-2")
			require.NotContains(t, bodies[1], "memory-generation-1")
		})
	}
}

func TestGuardContractRecoveryDoesNotReplayCompletedTool(t *testing.T) {
	t.Parallel()
	client, guard := newContractClient(t)
	var requests contractRequests
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		switch requests.record(t, req) {
		case 1:
			w.Header().Set("Content-Type", "text/event-stream")
			fmt.Fprint(w, "data: {\"id\":\"c-1\",\"object\":\"chat.completion.chunk\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"tool_calls\":[{\"index\":0,\"id\":\"tool-1\",\"type\":\"function\",\"function\":{\"name\":\"count_once\",\"arguments\":\"{}\"}}]},\"finish_reason\":null}]}\n\n")
			fmt.Fprint(w, "data: {\"id\":\"c-1\",\"object\":\"chat.completion.chunk\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"tool_calls\"}]}\n\ndata: [DONE]\n\n")
		case 2:
			guard.generation.Store(2)
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusUnauthorized)
			fmt.Fprint(w, `{"error":{"message":"expired","type":"authentication_error"}}`)
		default:
			writeContractText(w, "chat")
		}
	}))
	defer server.Close()
	var executions atomic.Int64
	tool := fantasy.NewAgentTool("count_once", "Count an observable local effect.", func(context.Context, struct{}, fantasy.ToolCall) (fantasy.ToolResponse, error) {
		executions.Add(1)
		return fantasy.NewTextResponse("completed-once"), nil
	})
	model := &contractModel{LanguageModel: newContractModel(t, "chat", server.URL, client, 0), guard: guard}
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	result, err := fantasy.NewAgent(model, fantasy.WithTools(tool)).Stream(ctx, fantasy.AgentStreamCall{
		Prompt:        "count once",
		PrepareStep:   model.prepareStep,
		OnAuthRefresh: func(context.Context, *fantasy.ProviderError) error { return nil },
		StopWhen:      []fantasy.StopCondition{fantasy.StepCountIs(3)},
	})
	require.NoError(t, err)
	require.Equal(t, "ok", result.Response.Content.Text())
	require.EqualValues(t, 1, executions.Load())
	require.Len(t, result.Steps, 2)
	require.EqualValues(t, 4, guard.checks.Load())
	require.EqualValues(t, 1, model.rebuilds.Load())
	bodies := requests.snapshot()
	require.Len(t, bodies, 3)
	require.Contains(t, bodies[2], "completed-once", "Rebuilt request retains prior tool results")
	require.Contains(t, bodies[2], "memory-generation-2")
	require.NotContains(t, bodies[2], "memory-generation-1")
}

func TestGuardContractDoesNotRecoverAfterObservableOutput(t *testing.T) {
	t.Parallel()
	client, guard := newContractClient(t)
	var requests atomic.Int64
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		requests.Add(1)
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprint(w, "data: {\"id\":\"c-1\",\"object\":\"chat.completion.chunk\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"partial\"},\"finish_reason\":null}]}\n\n")
		fmt.Fprint(w, "data: {\"error\":{\"message\":\"retry\",\"type\":\"rate_limit_error\"}}\n\n")
	}))
	defer server.Close()
	model := &contractModel{LanguageModel: newContractModel(t, "chat", server.URL, client, 0), guard: guard}
	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()
	var text strings.Builder
	var retries int
	_, err := fantasy.NewAgent(model).Stream(ctx, fantasy.AgentStreamCall{
		Prompt:      "hello",
		PrepareStep: model.prepareStep,
		OnTextDelta: func(_ string, delta string) error {
			text.WriteString(delta)
			return nil
		},
		OnRetry: func(providerErr *fantasy.ProviderError, _ time.Duration) {
			retries++
			guard.generation.Store(2)
		},
	})
	require.ErrorIs(t, err, errContractPackObsolete)
	require.Equal(t, "partial", text.String())
	require.EqualValues(t, 1, requests.Load())
	require.Equal(t, 1, retries, "Guard failure must not become another network retry")
	require.EqualValues(t, 0, model.rebuilds.Load())
	require.EqualValues(t, 1, guard.blocked.Load())
}

func TestGuardContractTransportUsesFreshHTTP1Connections(t *testing.T) {
	t.Parallel()
	client, guard := newContractClient(t)
	var connections atomic.Int64
	server := httptest.NewUnstartedServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		if req.ProtoMajor != 1 {
			t.Errorf("Expected HTTP/1, got %s", req.Proto)
		}
		_, _ = io.Copy(io.Discard, req.Body)
		if req.URL.Path == "/close" {
			connection, _, err := w.(http.Hijacker).Hijack()
			if err != nil {
				t.Errorf("Failed to interrupt fresh connection: %v", err)
				return
			}
			_ = connection.Close()
			return
		}
		w.WriteHeader(http.StatusNoContent)
	}))
	server.EnableHTTP2 = true
	server.Config.ConnState = func(_ net.Conn, state http.ConnState) {
		if state == http.StateNew {
			connections.Add(1)
		}
	}
	server.StartTLS()
	defer server.Close()
	base := guard.next.(*http.Transport)
	base.TLSClientConfig = server.Client().Transport.(*http.Transport).TLSClientConfig.Clone()
	for range 2 {
		req, err := http.NewRequestWithContext(context.Background(), http.MethodPost, server.URL, strings.NewReader("body"))
		require.NoError(t, err)
		resp, err := client.Do(req)
		require.NoError(t, err)
		require.Equal(t, 1, resp.ProtoMajor)
		require.NoError(t, resp.Body.Close())
	}
	require.EqualValues(t, 2, connections.Load())
	require.EqualValues(t, 2, guard.checks.Load())
	// A dropped fresh connection must return to the model retry boundary,
	// never retry internally behind the guard, even with a replayable body.
	req, err := http.NewRequestWithContext(context.Background(), http.MethodPost, server.URL+"/close", strings.NewReader("body"))
	require.NoError(t, err)
	req.Header.Set("Idempotency-Key", "contract-replay-probe")
	resp, err := client.Do(req)
	require.Error(t, err)
	require.Nil(t, resp)
	require.EqualValues(t, 3, connections.Load())
	require.EqualValues(t, 3, guard.checks.Load())
}

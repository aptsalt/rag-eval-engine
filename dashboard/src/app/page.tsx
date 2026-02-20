'use client';

import { useState, useRef, useEffect, useCallback } from 'react';
import { api, type QueryResponse, type Collection, type OllamaModel } from '@/lib/api';
import { cn, formatMs, formatPercent, scoreBgColor } from '@/lib/utils';

interface Message {
  role: 'user' | 'assistant';
  content: string;
  sources?: QueryResponse['sources'];
  evalScores?: QueryResponse['eval_scores'];
  latencyMs?: number;
  model?: string;
  streaming?: boolean;
}

const SUGGESTED_QUERIES = [
  'What are the main topics covered in the documents?',
  'Summarize the key findings from the uploaded files.',
  'What technical concepts are discussed?',
  'Are there any recommendations or conclusions?',
];

export default function QueryPlayground() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [collections, setCollections] = useState<Collection[]>([]);
  const [models, setModels] = useState<OllamaModel[]>([]);
  const [selectedCollection, setSelectedCollection] = useState('documents');
  const [selectedModel, setSelectedModel] = useState<string>('');
  const [topK, setTopK] = useState(5);
  const [useStreaming, setUseStreaming] = useState(true);
  const [showSettings, setShowSettings] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    api.getCollections().then(setCollections).catch(() => {});
    api.getModels().then((m) => {
      setModels(m);
      if (m.length > 0 && !selectedModel) {
        setSelectedModel(m[0].name);
      }
    }).catch(() => {});
  }, []);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleStreamQuery = useCallback(async (userMessage: string) => {
    setMessages((prev) => [...prev, { role: 'assistant', content: '', streaming: true }]);

    try {
      const response = await api.queryStream({
        query: userMessage,
        collection: selectedCollection,
        top_k: topK,
        model: selectedModel || undefined,
      });

      if (!response.ok) throw new Error('Stream request failed');
      if (!response.body) throw new Error('No response body');

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let fullAnswer = '';
      let sources: QueryResponse['sources'] = [];

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const text = decoder.decode(value, { stream: true });
        const lines = text.split('\n');

        for (const line of lines) {
          if (!line.startsWith('data: ')) continue;
          try {
            const event = JSON.parse(line.slice(6));
            if (event.type === 'token') {
              fullAnswer += event.data;
              setMessages((prev) => {
                const updated = [...prev];
                const last = updated[updated.length - 1];
                if (last?.role === 'assistant') {
                  updated[updated.length - 1] = { ...last, content: fullAnswer };
                }
                return updated;
              });
            } else if (event.type === 'sources') {
              sources = event.data;
            } else if (event.type === 'done') {
              setMessages((prev) => {
                const updated = [...prev];
                const last = updated[updated.length - 1];
                if (last?.role === 'assistant') {
                  updated[updated.length - 1] = {
                    ...last,
                    content: fullAnswer,
                    sources,
                    streaming: false,
                  };
                }
                return updated;
              });
            }
          } catch {
            // skip malformed SSE
          }
        }
      }
    } catch (error) {
      setMessages((prev) => {
        const updated = [...prev];
        const last = updated[updated.length - 1];
        if (last?.role === 'assistant') {
          updated[updated.length - 1] = {
            ...last,
            content: `Error: ${error instanceof Error ? error.message : 'Stream failed'}`,
            streaming: false,
          };
        }
        return updated;
      });
    }
  }, [selectedCollection, topK, selectedModel]);

  const handleNonStreamQuery = useCallback(async (userMessage: string) => {
    try {
      const response = await api.query({
        query: userMessage,
        collection: selectedCollection,
        top_k: topK,
        model: selectedModel || undefined,
        evaluate: true,
      });

      setMessages((prev) => [
        ...prev,
        {
          role: 'assistant',
          content: response.answer,
          sources: response.sources,
          evalScores: response.eval_scores,
          latencyMs: response.latency_ms,
          model: response.model,
        },
      ]);
    } catch (error) {
      setMessages((prev) => [
        ...prev,
        {
          role: 'assistant',
          content: `Error: ${error instanceof Error ? error.message : 'Failed to get response'}`,
        },
      ]);
    }
  }, [selectedCollection, topK, selectedModel]);

  const handleSubmit = async (e?: React.FormEvent) => {
    e?.preventDefault();
    if (!input.trim() || loading) return;

    const userMessage = input.trim();
    setInput('');
    setMessages((prev) => [...prev, { role: 'user', content: userMessage }]);
    setLoading(true);

    if (useStreaming) {
      await handleStreamQuery(userMessage);
    } else {
      await handleNonStreamQuery(userMessage);
    }

    setLoading(false);
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
      e.preventDefault();
      handleSubmit();
    }
    if (e.key === 'Enter' && !e.shiftKey && !e.ctrlKey && !e.metaKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  const handleSuggestedQuery = (query: string) => {
    setInput(query);
    inputRef.current?.focus();
  };

  const clearChat = () => {
    setMessages([]);
  };

  return (
    <div className="flex h-full flex-col">
      <div className="border-b border-gray-200 dark:border-gray-700 pb-4">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold text-gray-900 dark:text-white">Query Playground</h1>
            <p className="mt-1 text-sm text-gray-500 dark:text-gray-400">
              Ask questions about your documents with real-time evaluation
            </p>
          </div>
          <div className="flex items-center gap-2">
            {messages.length > 0 && (
              <button
                onClick={clearChat}
                className="rounded-lg border border-gray-300 dark:border-gray-600 px-3 py-1.5 text-sm text-gray-600 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors"
              >
                Clear Chat
              </button>
            )}
            <button
              onClick={() => setShowSettings(!showSettings)}
              className={cn(
                'rounded-lg border px-3 py-1.5 text-sm transition-colors',
                showSettings
                  ? 'border-brand-500 bg-brand-50 dark:bg-brand-900/20 text-brand-700 dark:text-brand-300'
                  : 'border-gray-300 dark:border-gray-600 text-gray-600 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700'
              )}
            >
              Settings
            </button>
          </div>
        </div>

        {showSettings && (
          <div className="mt-3 grid grid-cols-2 gap-4 rounded-lg border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800 p-4 sm:grid-cols-4">
            <div>
              <label className="block text-xs font-medium text-gray-500 dark:text-gray-400 mb-1">Collection</label>
              <select
                value={selectedCollection}
                onChange={(e) => setSelectedCollection(e.target.value)}
                className="w-full rounded-lg border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-700 px-2.5 py-1.5 text-sm dark:text-white focus:border-brand-500 focus:outline-none focus:ring-1 focus:ring-brand-500"
              >
                {collections.length === 0 && <option value="documents">documents</option>}
                {collections.map((col) => (
                  <option key={col.name} value={col.name}>
                    {col.name} ({col.doc_count} docs)
                  </option>
                ))}
              </select>
            </div>
            <div>
              <label className="block text-xs font-medium text-gray-500 dark:text-gray-400 mb-1">Model</label>
              <select
                value={selectedModel}
                onChange={(e) => setSelectedModel(e.target.value)}
                className="w-full rounded-lg border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-700 px-2.5 py-1.5 text-sm dark:text-white focus:border-brand-500 focus:outline-none focus:ring-1 focus:ring-brand-500"
              >
                {models.map((m) => (
                  <option key={m.name} value={m.name}>{m.name}</option>
                ))}
                {models.length === 0 && <option value="">Default</option>}
              </select>
            </div>
            <div>
              <label className="block text-xs font-medium text-gray-500 dark:text-gray-400 mb-1">
                Top-K: {topK}
              </label>
              <input
                type="range"
                min={1}
                max={20}
                value={topK}
                onChange={(e) => setTopK(Number(e.target.value))}
                className="w-full accent-brand-600"
              />
            </div>
            <div>
              <label className="block text-xs font-medium text-gray-500 dark:text-gray-400 mb-1">Mode</label>
              <div className="flex items-center gap-2">
                <button
                  onClick={() => setUseStreaming(true)}
                  className={cn(
                    'rounded-md px-2.5 py-1.5 text-xs font-medium transition-colors',
                    useStreaming
                      ? 'bg-brand-600 text-white'
                      : 'bg-gray-100 dark:bg-gray-700 text-gray-600 dark:text-gray-300'
                  )}
                >
                  Stream
                </button>
                <button
                  onClick={() => setUseStreaming(false)}
                  className={cn(
                    'rounded-md px-2.5 py-1.5 text-xs font-medium transition-colors',
                    !useStreaming
                      ? 'bg-brand-600 text-white'
                      : 'bg-gray-100 dark:bg-gray-700 text-gray-600 dark:text-gray-300'
                  )}
                >
                  Eval
                </button>
              </div>
              <p className="mt-0.5 text-[10px] text-gray-400">
                {useStreaming ? 'Real-time tokens' : 'With quality scores'}
              </p>
            </div>
          </div>
        )}
      </div>

      <div className="flex-1 overflow-y-auto py-4">
        {messages.length === 0 && (
          <div className="flex h-full items-center justify-center">
            <div className="text-center max-w-lg">
              <div className="mx-auto h-14 w-14 rounded-2xl bg-gradient-to-br from-brand-500 to-brand-700 flex items-center justify-center shadow-lg shadow-brand-500/20">
                <svg className="h-7 w-7 text-white" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" d="M9.813 15.904 9 18.75l-.813-2.846a4.5 4.5 0 0 0-3.09-3.09L2.25 12l2.846-.813a4.5 4.5 0 0 0 3.09-3.09L9 5.25l.813 2.846a4.5 4.5 0 0 0 3.09 3.09L15.75 12l-2.846.813a4.5 4.5 0 0 0-3.09 3.09ZM18.259 8.715 18 9.75l-.259-1.035a3.375 3.375 0 0 0-2.455-2.456L14.25 6l1.036-.259a3.375 3.375 0 0 0 2.455-2.456L18 2.25l.259 1.035a3.375 3.375 0 0 0 2.455 2.456L21.75 6l-1.036.259a3.375 3.375 0 0 0-2.455 2.456ZM16.894 20.567 16.5 21.75l-.394-1.183a2.25 2.25 0 0 0-1.423-1.423L13.5 18.75l1.183-.394a2.25 2.25 0 0 0 1.423-1.423l.394-1.183.394 1.183a2.25 2.25 0 0 0 1.423 1.423l1.183.394-1.183.394a2.25 2.25 0 0 0-1.423 1.423Z" />
                </svg>
              </div>
              <h3 className="mt-5 text-lg font-semibold text-gray-900 dark:text-white">Ask a question</h3>
              <p className="mt-2 text-sm text-gray-500 dark:text-gray-400">
                Query your ingested documents. Each response includes source citations
                {!useStreaming && ' and evaluation scores for faithfulness and relevance'}.
              </p>
              <div className="mt-6 grid grid-cols-1 gap-2 sm:grid-cols-2">
                {SUGGESTED_QUERIES.map((q) => (
                  <button
                    key={q}
                    onClick={() => handleSuggestedQuery(q)}
                    className="rounded-xl border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800 px-4 py-3 text-left text-xs text-gray-600 dark:text-gray-300 hover:border-brand-300 dark:hover:border-brand-600 hover:bg-brand-50 dark:hover:bg-brand-900/20 transition-colors"
                  >
                    {q}
                  </button>
                ))}
              </div>
            </div>
          </div>
        )}

        <div className="space-y-6">
          {messages.map((msg, i) => (
            <div key={i} className={cn('flex gap-4', msg.role === 'user' ? 'justify-end' : '')}>
              {msg.role === 'assistant' && (
                <div className="h-8 w-8 shrink-0 rounded-full bg-gradient-to-br from-brand-500 to-brand-700 flex items-center justify-center">
                  <span className="text-white text-xs font-bold">AI</span>
                </div>
              )}
              <div className={cn('max-w-2xl', msg.role === 'user' ? 'order-first' : '')}>
                <div
                  className={cn(
                    'rounded-2xl px-4 py-3 text-sm',
                    msg.role === 'user'
                      ? 'bg-brand-600 text-white'
                      : 'bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 text-gray-800 dark:text-gray-200'
                  )}
                >
                  <div className="whitespace-pre-wrap prose prose-sm dark:prose-invert max-w-none">
                    {msg.content}
                    {msg.streaming && (
                      <span className="inline-block w-2 h-4 bg-brand-500 animate-pulse ml-0.5" />
                    )}
                  </div>
                </div>

                {msg.role === 'assistant' && msg.evalScores && (
                  <div className="mt-2 flex flex-wrap gap-2">
                    <ScoreBadge label="Faithfulness" value={msg.evalScores.faithfulness} />
                    <ScoreBadge label="Relevance" value={msg.evalScores.relevance} />
                    {msg.evalScores.hallucination_rate > 0 && (
                      <ScoreBadge label="Hallucination" value={msg.evalScores.hallucination_rate} invert />
                    )}
                    {msg.latencyMs && (
                      <span className="inline-flex items-center rounded-full bg-gray-100 dark:bg-gray-700 px-2.5 py-0.5 text-xs text-gray-600 dark:text-gray-300">
                        {formatMs(msg.latencyMs)}
                      </span>
                    )}
                    {msg.model && (
                      <span className="inline-flex items-center rounded-full bg-gray-100 dark:bg-gray-700 px-2.5 py-0.5 text-xs text-gray-600 dark:text-gray-300">
                        {msg.model}
                      </span>
                    )}
                  </div>
                )}

                {msg.role === 'assistant' && msg.sources && msg.sources.length > 0 && (
                  <details className="mt-2">
                    <summary className="cursor-pointer text-xs text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-300">
                      {msg.sources.length} source(s) used
                    </summary>
                    <div className="mt-2 space-y-2">
                      {msg.sources.map((source, j) => (
                        <div key={j} className="rounded-lg bg-gray-50 dark:bg-gray-800/50 border border-gray-100 dark:border-gray-700 p-3 text-xs">
                          <div className="flex items-center justify-between">
                            <span className="font-medium text-gray-700 dark:text-gray-300">[Source {source.index}]</span>
                            <div className="flex items-center gap-2">
                              <div className="h-1.5 w-16 rounded-full bg-gray-200 dark:bg-gray-600 overflow-hidden">
                                <div
                                  className="h-full rounded-full bg-brand-500"
                                  style={{ width: `${Math.min(100, source.score * 100)}%` }}
                                />
                              </div>
                              <span className="text-gray-400">{source.source}</span>
                            </div>
                          </div>
                          <p className="mt-1 text-gray-600 dark:text-gray-400">{source.text}</p>
                        </div>
                      ))}
                    </div>
                  </details>
                )}
              </div>
            </div>
          ))}
          {loading && !messages.some((m) => m.streaming) && (
            <div className="flex gap-4">
              <div className="h-8 w-8 shrink-0 rounded-full bg-gradient-to-br from-brand-500 to-brand-700 flex items-center justify-center">
                <span className="text-white text-xs font-bold">AI</span>
              </div>
              <div className="rounded-2xl bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 px-4 py-3">
                <div className="flex gap-1">
                  <div className="h-2 w-2 rounded-full bg-gray-400 animate-bounce" style={{ animationDelay: '0ms' }} />
                  <div className="h-2 w-2 rounded-full bg-gray-400 animate-bounce" style={{ animationDelay: '150ms' }} />
                  <div className="h-2 w-2 rounded-full bg-gray-400 animate-bounce" style={{ animationDelay: '300ms' }} />
                </div>
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>
      </div>

      <form onSubmit={handleSubmit} className="border-t border-gray-200 dark:border-gray-700 pt-4">
        <div className="flex gap-3">
          <div className="relative flex-1">
            <textarea
              ref={inputRef}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Ask a question about your documents..."
              disabled={loading}
              rows={1}
              className="w-full resize-none rounded-xl border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 px-4 py-3 pr-12 text-sm dark:text-white focus:border-brand-500 focus:outline-none focus:ring-1 focus:ring-brand-500 disabled:opacity-50 placeholder:text-gray-400"
            />
            <div className="absolute bottom-1.5 right-2 text-[10px] text-gray-400 pointer-events-none">
              Enter to send
            </div>
          </div>
          <button
            type="submit"
            disabled={loading || !input.trim()}
            className="rounded-xl bg-brand-600 px-6 py-3 text-sm font-medium text-white hover:bg-brand-700 focus:outline-none focus:ring-2 focus:ring-brand-500 focus:ring-offset-2 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            <svg className="h-5 w-5" fill="none" viewBox="0 0 24 24" strokeWidth={2} stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" d="M6 12 3.269 3.125A59.769 59.769 0 0 1 21.485 12 59.768 59.768 0 0 1 3.27 20.875L5.999 12Zm0 0h7.5" />
            </svg>
          </button>
        </div>
      </form>
    </div>
  );
}

function ScoreBadge({
  label,
  value,
  invert = false,
}: {
  label: string;
  value: number;
  invert?: boolean;
}) {
  const displayScore = invert ? 1 - value : value;
  return (
    <span className={cn('inline-flex items-center rounded-full px-2.5 py-0.5 text-xs font-medium', scoreBgColor(displayScore))}>
      {label}: {formatPercent(value)}
    </span>
  );
}

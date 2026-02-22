'use client';

import { useState, useEffect } from 'react';
import { api, type Collection, type OptimalParams } from '@/lib/api';
import { cn, formatNumber } from '@/lib/utils';

interface RetrievalResult {
  text: string;
  score: number;
  vector_score?: number;
  sparse_score?: number;
  chunk_index: number;
  metadata?: {
    source?: string;
    file_type?: string;
    page?: number;
    strategy?: string;
    token_count?: number;
  };
}

export default function RetrievalExplorer() {
  const [query, setQuery] = useState('');
  const [collections, setCollections] = useState<Collection[]>([]);
  const [selectedCollection, setSelectedCollection] = useState('documents');
  const [topK, setTopK] = useState(5);
  const [alpha, setAlpha] = useState(0.7);
  const [results, setResults] = useState<RetrievalResult[]>([]);
  const [loading, setLoading] = useState(false);
  const [searched, setSearched] = useState(false);
  const [latencyMs, setLatencyMs] = useState<number | null>(null);
  const [optimalParams, setOptimalParams] = useState<OptimalParams | null>(null);
  const [expandedChunks, setExpandedChunks] = useState<Set<number>>(new Set());
  const [sourceFilter, setSourceFilter] = useState('');

  useEffect(() => {
    api.getCollections().then(setCollections).catch(() => {});
  }, []);

  useEffect(() => {
    api.getOptimalParams(selectedCollection).then(setOptimalParams).catch(() => {});
  }, [selectedCollection]);

  const handleSearch = async (e?: React.FormEvent) => {
    e?.preventDefault();
    if (!query.trim() || loading) return;

    setLoading(true);
    setSearched(true);
    const start = performance.now();

    try {
      const data = await api.retrieve({
        query: query.trim(),
        collection: selectedCollection,
        top_k: topK,
        alpha,
        source_filter: sourceFilter || undefined,
      });
      setResults(data);
      setLatencyMs(performance.now() - start);
      setExpandedChunks(new Set());
    } catch {
      setResults([]);
      setLatencyMs(null);
    } finally {
      setLoading(false);
    }
  };

  const applyOptimal = () => {
    if (!optimalParams) return;
    if (optimalParams.optimal_alpha !== null && optimalParams.optimal_alpha !== undefined) {
      setAlpha(optimalParams.optimal_alpha);
    }
    if (optimalParams.optimal_top_k !== null && optimalParams.optimal_top_k !== undefined) {
      setTopK(optimalParams.optimal_top_k);
    }
  };

  const toggleExpand = (index: number) => {
    setExpandedChunks((prev) => {
      const next = new Set(prev);
      if (next.has(index)) next.delete(index);
      else next.add(index);
      return next;
    });
  };

  const uniqueSources = [...new Set(results.map((r) => r.metadata?.source?.split('_').pop()).filter(Boolean))];
  const maxScore = results.length > 0 ? Math.max(...results.map((r) => r.score)) : 1;
  const COLLAPSE_LENGTH = 300;

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-gray-900 dark:text-white">Retrieval Explorer</h1>
        <p className="mt-1 text-sm text-gray-500 dark:text-gray-400">
          Test hybrid retrieval independently — tune alpha, top-k, and inspect chunk scores
        </p>
      </div>

      {/* Search Controls */}
      <div className="rounded-xl border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800 p-6">
        <form onSubmit={handleSearch} className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">Query</label>
            <input
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Enter a search query to test retrieval..."
              className="w-full rounded-lg border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-700 px-4 py-2.5 text-sm dark:text-white focus:border-brand-500 focus:outline-none focus:ring-1 focus:ring-brand-500"
            />
          </div>

          <div className="grid gap-4 sm:grid-cols-4">
            <div>
              <label className="block text-xs font-medium text-gray-500 dark:text-gray-400 mb-1">Collection</label>
              <select
                value={selectedCollection}
                onChange={(e) => setSelectedCollection(e.target.value)}
                className="w-full rounded-lg border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-700 px-3 py-1.5 text-sm dark:text-white focus:border-brand-500 focus:outline-none focus:ring-1 focus:ring-brand-500"
              >
                {collections.length === 0 && <option value="documents">documents</option>}
                {collections.map((col) => (
                  <option key={col.name} value={col.name}>{col.name}</option>
                ))}
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
              <label className="block text-xs font-medium text-gray-500 dark:text-gray-400 mb-1">
                Alpha: {alpha.toFixed(1)} ({alpha > 0.5 ? 'vector-heavy' : alpha < 0.5 ? 'keyword-heavy' : 'balanced'})
              </label>
              <input
                type="range"
                min={0}
                max={1}
                step={0.1}
                value={alpha}
                onChange={(e) => setAlpha(Number(e.target.value))}
                className="w-full accent-brand-600"
              />
            </div>
            <div className="flex items-end">
              <button
                type="submit"
                disabled={loading || !query.trim()}
                className={cn(
                  'w-full rounded-lg px-4 py-2 text-sm font-medium text-white transition-colors',
                  loading || !query.trim()
                    ? 'cursor-not-allowed bg-gray-400'
                    : 'bg-brand-600 hover:bg-brand-700'
                )}
              >
                {loading ? 'Searching...' : 'Search'}
              </button>
            </div>
          </div>

          {/* Auto-tune recommendation */}
          {optimalParams && optimalParams.sufficient_data && (
            <div className="flex items-center gap-3 rounded-lg bg-brand-50 dark:bg-brand-900/20 border border-brand-200 dark:border-brand-800 px-4 py-2.5">
              <span className="text-xs text-brand-700 dark:text-brand-300">
                Recommended: alpha={optimalParams.optimal_alpha ?? 'N/A'}, top_k={optimalParams.optimal_top_k ?? 'N/A'}
                <span className="ml-1 text-brand-500">(based on {optimalParams.total_queries} queries)</span>
              </span>
              <button
                type="button"
                onClick={applyOptimal}
                className="rounded-md bg-brand-600 px-2.5 py-1 text-xs font-medium text-white hover:bg-brand-700 transition-colors"
              >
                Apply
              </button>
            </div>
          )}
        </form>
      </div>

      {/* Results */}
      {searched && (
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <h2 className="text-lg font-semibold text-gray-900 dark:text-white">
              Results ({results.length})
            </h2>
            <div className="flex items-center gap-3">
              {uniqueSources.length > 1 && (
                <select
                  value={sourceFilter}
                  onChange={(e) => setSourceFilter(e.target.value)}
                  className="rounded-lg border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-700 px-2 py-1 text-xs dark:text-white"
                >
                  <option value="">All sources</option>
                  {uniqueSources.map((s) => (
                    <option key={s} value={s}>{s}</option>
                  ))}
                </select>
              )}
              {latencyMs !== null && (
                <span className="text-sm text-gray-500 dark:text-gray-400">
                  {latencyMs.toFixed(0)}ms
                </span>
              )}
            </div>
          </div>

          {results.length === 0 ? (
            <div className="rounded-xl border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800 p-8 text-center">
              <p className="text-sm text-gray-500 dark:text-gray-400">No results found. Try a different query or collection.</p>
            </div>
          ) : (
            <div className="space-y-3">
              {results.map((result, i) => {
                const isLong = result.text.length > COLLAPSE_LENGTH;
                const isExpanded = expandedChunks.has(i);
                const displayText = isLong && !isExpanded
                  ? result.text.slice(0, COLLAPSE_LENGTH) + '...'
                  : result.text;

                return (
                  <div key={i} className="rounded-xl border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800 p-4">
                    <div className="flex items-center justify-between mb-2">
                      <div className="flex items-center gap-3">
                        <span className="rounded-full bg-brand-100 dark:bg-brand-900/30 px-2.5 py-0.5 text-xs font-bold text-brand-700 dark:text-brand-300">
                          #{i + 1}
                        </span>
                        <span className="text-xs text-gray-500 dark:text-gray-400">{result.metadata?.source?.split('_').pop() || 'unknown'}</span>
                        <span className="text-xs text-gray-400">chunk {result.chunk_index}</span>
                      </div>
                      <span className="text-sm font-mono font-medium text-gray-900 dark:text-white">
                        {result.score.toFixed(4)}
                      </span>
                    </div>

                    {/* Score bar */}
                    <div className="h-1.5 w-full rounded-full bg-gray-100 dark:bg-gray-700 mb-3 overflow-hidden">
                      <div
                        className="h-full rounded-full bg-gradient-to-r from-brand-400 to-brand-600 transition-all"
                        style={{ width: `${(result.score / maxScore) * 100}%` }}
                      />
                    </div>

                    <p className="text-sm text-gray-700 dark:text-gray-300 leading-relaxed whitespace-pre-wrap">
                      {displayText}
                    </p>
                    {isLong && (
                      <button
                        onClick={() => toggleExpand(i)}
                        className="mt-1 text-xs text-brand-600 dark:text-brand-400 hover:text-brand-700"
                      >
                        {isExpanded ? 'Show less' : 'Show more'}
                      </button>
                    )}
                  </div>
                );
              })}
            </div>
          )}
        </div>
      )}

      {!searched && (
        <div className="rounded-xl border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800 p-12 text-center">
          <svg className="mx-auto h-12 w-12 text-gray-400" fill="none" viewBox="0 0 24 24" strokeWidth={1} stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" d="m21 21-5.197-5.197m0 0A7.5 7.5 0 1 0 5.196 5.196a7.5 7.5 0 0 0 10.607 10.607Z" />
          </svg>
          <h3 className="mt-4 text-lg font-medium text-gray-900 dark:text-white">Test your retrieval pipeline</h3>
          <p className="mt-2 text-sm text-gray-500 dark:text-gray-400 max-w-md mx-auto">
            Enter a query above to see ranked chunks from your documents.
            Adjust alpha to blend between vector search (1.0) and keyword search (0.0).
          </p>
        </div>
      )}
    </div>
  );
}

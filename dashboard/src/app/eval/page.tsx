'use client';

import { useState, useEffect } from 'react';
import { api, type MetricsResponse, type Collection } from '@/lib/api';
import { cn, formatMs, formatPercent, scoreBgColor, scoreColor } from '@/lib/utils';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  BarChart,
  Bar,
  Legend,
  AreaChart,
  Area,
} from 'recharts';

export default function EvalDashboard() {
  const [metrics, setMetrics] = useState<MetricsResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [collections, setCollections] = useState<Collection[]>([]);
  const [selectedCollection, setSelectedCollection] = useState<string | undefined>(undefined);
  const [autoRefresh, setAutoRefresh] = useState(true);

  const loadMetrics = async () => {
    try {
      const data = await api.getMetrics(selectedCollection, 200);
      setMetrics(data);
    } catch {
      // API unavailable
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    api.getCollections().then(setCollections).catch(() => {});
  }, []);

  useEffect(() => {
    loadMetrics();
    if (!autoRefresh) return;
    const interval = setInterval(loadMetrics, 30000);
    return () => clearInterval(interval);
  }, [selectedCollection, autoRefresh]);

  if (loading) {
    return (
      <div className="flex h-full items-center justify-center">
        <div className="text-center">
          <div className="h-8 w-8 animate-spin rounded-full border-4 border-brand-200 border-t-brand-600 mx-auto" />
          <p className="mt-4 text-sm text-gray-500 dark:text-gray-400">Loading metrics...</p>
        </div>
      </div>
    );
  }

  if (!metrics || metrics.total_queries === 0) {
    return (
      <div className="space-y-6">
        <div>
          <h1 className="text-2xl font-bold text-gray-900 dark:text-white">Evaluation Dashboard</h1>
          <p className="mt-1 text-sm text-gray-500 dark:text-gray-400">Track RAG quality metrics over time</p>
        </div>
        <div className="flex h-64 items-center justify-center rounded-xl border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800">
          <div className="text-center">
            <svg className="mx-auto h-12 w-12 text-gray-400" fill="none" viewBox="0 0 24 24" strokeWidth={1} stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" d="M3 13.125C3 12.504 3.504 12 4.125 12h2.25c.621 0 1.125.504 1.125 1.125v6.75C7.5 20.496 6.996 21 6.375 21h-2.25A1.125 1.125 0 0 1 3 19.875v-6.75ZM9.75 8.625c0-.621.504-1.125 1.125-1.125h2.25c.621 0 1.125.504 1.125 1.125v11.25c0 .621-.504 1.125-1.125 1.125h-2.25a1.125 1.125 0 0 1-1.125-1.125V8.625ZM16.5 4.125c0-.621.504-1.125 1.125-1.125h2.25C20.496 3 21 3.504 21 4.125v15.75c0 .621-.504 1.125-1.125 1.125h-2.25a1.125 1.125 0 0 1-1.125-1.125V4.125Z" />
            </svg>
            <h3 className="mt-4 text-lg font-medium text-gray-900 dark:text-white">No evaluation data yet</h3>
            <p className="mt-2 text-sm text-gray-500 dark:text-gray-400">
              Start querying your documents to generate evaluation metrics.
            </p>
          </div>
        </div>
      </div>
    );
  }

  const chartData = metrics.time_series.map((point, i) => ({
    index: i + 1,
    faithfulness: point.faithfulness !== null ? point.faithfulness * 100 : null,
    relevance: point.relevance !== null ? point.relevance * 100 : null,
    hallucination: point.hallucination_rate !== null ? point.hallucination_rate * 100 : null,
    latency: point.latency_ms,
    tokens: point.tokens_used,
  }));

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900 dark:text-white">Evaluation Dashboard</h1>
          <p className="mt-1 text-sm text-gray-500 dark:text-gray-400">
            Track RAG quality metrics across {metrics.total_queries} queries
          </p>
        </div>
        <div className="flex items-center gap-3">
          <select
            value={selectedCollection || ''}
            onChange={(e) => setSelectedCollection(e.target.value || undefined)}
            className="rounded-lg border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-700 px-3 py-1.5 text-sm dark:text-white focus:border-brand-500 focus:outline-none focus:ring-1 focus:ring-brand-500"
          >
            <option value="">All collections</option>
            {collections.map((col) => (
              <option key={col.name} value={col.name}>{col.name}</option>
            ))}
          </select>
          <button
            onClick={() => setAutoRefresh(!autoRefresh)}
            className={cn(
              'rounded-lg border px-3 py-1.5 text-xs font-medium transition-colors',
              autoRefresh
                ? 'border-green-300 dark:border-green-700 bg-green-50 dark:bg-green-900/20 text-green-700 dark:text-green-400'
                : 'border-gray-300 dark:border-gray-600 text-gray-500 dark:text-gray-400'
            )}
          >
            {autoRefresh ? 'Auto-refresh ON' : 'Auto-refresh OFF'}
          </button>
          <button
            onClick={loadMetrics}
            className="rounded-lg border border-gray-300 dark:border-gray-600 px-3 py-1.5 text-xs font-medium text-gray-600 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors"
          >
            Refresh
          </button>
        </div>
      </div>

      {/* Summary Cards */}
      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <MetricCard
          label="Avg Faithfulness"
          value={metrics.avg_faithfulness}
          format="percent"
          description="Are answers grounded in context?"
        />
        <MetricCard
          label="Avg Relevance"
          value={metrics.avg_relevance}
          format="percent"
          description="Do answers address the question?"
        />
        <MetricCard
          label="Avg Hallucination"
          value={metrics.avg_hallucination_rate}
          format="percent"
          invert
          description="Ungrounded claims rate (lower is better)"
        />
        <MetricCard
          label="P95 Latency"
          value={metrics.p95_latency_ms}
          format="ms"
          description="95th percentile response time"
        />
      </div>

      {/* Quality Over Time - Area Chart */}
      <div className="rounded-xl border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800 p-6">
        <h2 className="text-lg font-semibold text-gray-900 dark:text-white">Quality Over Time</h2>
        <p className="mt-1 text-sm text-gray-500 dark:text-gray-400">Faithfulness, relevance, and hallucination rate per query</p>
        <div className="mt-4 h-80">
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={chartData}>
              <defs>
                <linearGradient id="faithGrad" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#16a34a" stopOpacity={0.1} />
                  <stop offset="95%" stopColor="#16a34a" stopOpacity={0} />
                </linearGradient>
                <linearGradient id="relGrad" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#2563eb" stopOpacity={0.1} />
                  <stop offset="95%" stopColor="#2563eb" stopOpacity={0} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
              <XAxis dataKey="index" label={{ value: 'Query #', position: 'bottom' }} tick={{ fontSize: 12 }} />
              <YAxis domain={[0, 100]} tick={{ fontSize: 12 }} label={{ value: '%', angle: -90, position: 'insideLeft' }} />
              <Tooltip
                contentStyle={{ fontSize: 12, borderRadius: 8, background: 'var(--tooltip-bg, #fff)', border: '1px solid #e5e7eb' }}
                formatter={(value: number) => [`${value.toFixed(1)}%`]}
              />
              <Legend />
              <Area type="monotone" dataKey="faithfulness" stroke="#16a34a" strokeWidth={2} fill="url(#faithGrad)" name="Faithfulness" />
              <Area type="monotone" dataKey="relevance" stroke="#2563eb" strokeWidth={2} fill="url(#relGrad)" name="Relevance" />
              <Line type="monotone" dataKey="hallucination" stroke="#dc2626" strokeWidth={2} dot={false} name="Hallucination" />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Latency & Tokens */}
      <div className="grid gap-6 lg:grid-cols-2">
        <div className="rounded-xl border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800 p-6">
          <h2 className="text-lg font-semibold text-gray-900 dark:text-white">Latency Distribution</h2>
          <p className="mt-1 text-sm text-gray-500 dark:text-gray-400">Response time per query</p>
          <div className="mt-4 h-64">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                <XAxis dataKey="index" tick={{ fontSize: 12 }} />
                <YAxis tick={{ fontSize: 12 }} label={{ value: 'ms', angle: -90, position: 'insideLeft' }} />
                <Tooltip contentStyle={{ fontSize: 12, borderRadius: 8 }} formatter={(value: number) => [`${value.toFixed(0)}ms`]} />
                <Bar dataKey="latency" fill="#0ea5e9" radius={[4, 4, 0, 0]} name="Latency" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="rounded-xl border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800 p-6">
          <h2 className="text-lg font-semibold text-gray-900 dark:text-white">Token Usage</h2>
          <p className="mt-1 text-sm text-gray-500 dark:text-gray-400">Tokens consumed per query</p>
          <div className="mt-4 h-64">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                <XAxis dataKey="index" tick={{ fontSize: 12 }} />
                <YAxis tick={{ fontSize: 12 }} />
                <Tooltip contentStyle={{ fontSize: 12, borderRadius: 8 }} />
                <Bar dataKey="tokens" fill="#8b5cf6" radius={[4, 4, 0, 0]} name="Tokens" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* Stats Summary */}
      <div className="rounded-xl border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800 p-6">
        <h2 className="text-lg font-semibold text-gray-900 dark:text-white">Performance Summary</h2>
        <div className="mt-4 grid gap-6 sm:grid-cols-3">
          <div>
            <p className="text-sm font-medium text-gray-500 dark:text-gray-400">Total Queries</p>
            <p className="mt-1 text-3xl font-bold text-gray-900 dark:text-white">{metrics.total_queries}</p>
          </div>
          <div>
            <p className="text-sm font-medium text-gray-500 dark:text-gray-400">P50 Latency</p>
            <p className="mt-1 text-3xl font-bold text-gray-900 dark:text-white">
              {metrics.p50_latency_ms !== null ? formatMs(metrics.p50_latency_ms) : 'N/A'}
            </p>
          </div>
          <div>
            <p className="text-sm font-medium text-gray-500 dark:text-gray-400">Avg Latency</p>
            <p className="mt-1 text-3xl font-bold text-gray-900 dark:text-white">
              {metrics.avg_latency_ms !== null ? formatMs(metrics.avg_latency_ms) : 'N/A'}
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}

function MetricCard({
  label,
  value,
  format,
  description,
  invert = false,
}: {
  label: string;
  value: number | null;
  format: 'percent' | 'ms';
  description: string;
  invert?: boolean;
}) {
  const displayValue = value !== null
    ? format === 'percent'
      ? formatPercent(value)
      : formatMs(value)
    : 'N/A';

  const colorScore = value !== null
    ? invert ? 1 - value : value
    : 0.5;

  return (
    <div className="rounded-xl border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800 p-5">
      <p className="text-sm font-medium text-gray-500 dark:text-gray-400">{label}</p>
      <p className={cn('mt-2 text-3xl font-bold', format === 'percent' ? scoreColor(colorScore) : 'text-gray-900 dark:text-white')}>
        {displayValue}
      </p>
      <p className="mt-1 text-xs text-gray-400">{description}</p>
    </div>
  );
}

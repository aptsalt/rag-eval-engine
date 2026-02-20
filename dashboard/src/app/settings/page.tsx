'use client';

import { useState, useEffect } from 'react';
import { api, type AppSettings, type HealthStatus, type OllamaModel } from '@/lib/api';
import { cn } from '@/lib/utils';

export default function SettingsPage() {
  const [settings, setSettings] = useState<AppSettings | null>(null);
  const [health, setHealth] = useState<HealthStatus | null>(null);
  const [models, setModels] = useState<OllamaModel[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const load = async () => {
      try {
        const [settingsData, healthData, modelsData] = await Promise.all([
          api.getSettings(),
          api.health(),
          api.getModels(),
        ]);
        setSettings(settingsData);
        setHealth(healthData);
        setModels(modelsData);
      } catch {
        // API unavailable
      } finally {
        setLoading(false);
      }
    };
    load();
  }, []);

  if (loading) {
    return (
      <div className="flex h-full items-center justify-center">
        <div className="h-8 w-8 animate-spin rounded-full border-4 border-brand-200 border-t-brand-600" />
      </div>
    );
  }

  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-2xl font-bold text-gray-900 dark:text-white">Settings</h1>
        <p className="mt-1 text-sm text-gray-500 dark:text-gray-400">System configuration and status</p>
      </div>

      {/* System Status */}
      <div className="rounded-xl border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800 p-6">
        <h2 className="text-lg font-semibold text-gray-900 dark:text-white">System Status</h2>
        <div className="mt-4 space-y-3">
          <StatusRow
            label="API Server"
            value={health ? 'Online' : 'Offline'}
            ok={!!health}
          />
          <StatusRow
            label="Ollama"
            value={health?.ollama || 'Unknown'}
            ok={health?.ollama === 'connected'}
          />
          <StatusRow
            label="Embedding Model"
            value={settings?.embedding_model || 'Unknown'}
            ok={!!settings}
          />
          <StatusRow
            label="Default LLM"
            value={health?.default_llm || settings?.default_model || 'Unknown'}
            ok={!!settings}
          />
        </div>
      </div>

      {/* Available Models */}
      {models.length > 0 && (
        <div className="rounded-xl border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800 p-6">
          <h2 className="text-lg font-semibold text-gray-900 dark:text-white">Available Models</h2>
          <p className="mt-1 text-sm text-gray-500 dark:text-gray-400">Models available via Ollama</p>
          <div className="mt-4 grid gap-2 sm:grid-cols-2 lg:grid-cols-3">
            {models.map((model) => (
              <div key={model.name} className="flex items-center justify-between rounded-lg bg-gray-50 dark:bg-gray-700/50 px-4 py-2.5">
                <span className="text-sm font-medium text-gray-700 dark:text-gray-300">{model.name}</span>
                {model.size && (
                  <span className="text-xs text-gray-400">{(model.size / 1024 / 1024 / 1024).toFixed(1)}GB</span>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Configuration */}
      {settings && (
        <>
          <div className="rounded-xl border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800 p-6">
            <h2 className="text-lg font-semibold text-gray-900 dark:text-white">Retrieval Configuration</h2>
            <div className="mt-4 grid gap-6 sm:grid-cols-2">
              <SettingDisplay label="Chunking Strategy" value={settings.chunking_strategy} />
              <SettingDisplay label="Chunk Size" value={`${settings.chunk_size} tokens`} />
              <SettingDisplay label="Chunk Overlap" value={`${settings.chunk_overlap} tokens`} />
              <SettingDisplay label="Hybrid Alpha" value={`${settings.hybrid_alpha} (${settings.hybrid_alpha > 0.5 ? 'vector-heavy' : 'keyword-heavy'})`} />
              <SettingDisplay label="Default Top-K" value={String(settings.default_top_k)} />
              <SettingDisplay label="Reranker" value={settings.use_reranker ? 'Enabled' : 'Disabled'} />
            </div>
          </div>

          <div className="rounded-xl border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800 p-6">
            <h2 className="text-lg font-semibold text-gray-900 dark:text-white">Evaluation Configuration</h2>
            <div className="mt-4 grid gap-6 sm:grid-cols-2">
              <SettingDisplay label="Eval on Query" value={settings.eval_on_query ? 'Enabled' : 'Disabled'} />
              <SettingDisplay label="Lightweight Mode" value={settings.eval_lightweight ? 'Yes (faithfulness + relevance only)' : 'No (full suite)'} />
            </div>
          </div>

          <div className="rounded-xl border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800 p-6">
            <h2 className="text-lg font-semibold text-gray-900 dark:text-white">Environment Variables</h2>
            <p className="mt-1 text-sm text-gray-500 dark:text-gray-400">
              Configure these via environment variables or .env file (prefix: RAG_)
            </p>
            <div className="mt-4">
              <pre className="rounded-lg bg-gray-50 dark:bg-gray-900 p-4 text-xs text-gray-700 dark:text-gray-300 overflow-x-auto">
{`RAG_QDRANT_URL=http://localhost:6333
RAG_EMBEDDING_MODEL=${settings.embedding_model}
RAG_CHUNKING_STRATEGY=${settings.chunking_strategy}
RAG_CHUNK_SIZE=${settings.chunk_size}
RAG_CHUNK_OVERLAP=${settings.chunk_overlap}
RAG_OLLAMA_URL=http://localhost:11434
RAG_DEFAULT_MODEL=${settings.default_model}
RAG_HYBRID_ALPHA=${settings.hybrid_alpha}
RAG_DEFAULT_TOP_K=${settings.default_top_k}
RAG_EVAL_ON_QUERY=${settings.eval_on_query}
RAG_EVAL_LIGHTWEIGHT=${settings.eval_lightweight}`}
              </pre>
            </div>
          </div>
        </>
      )}
    </div>
  );
}

function StatusRow({ label, value, ok }: { label: string; value: string; ok: boolean }) {
  return (
    <div className="flex items-center justify-between rounded-lg bg-gray-50 dark:bg-gray-700/50 px-4 py-3">
      <span className="text-sm font-medium text-gray-700 dark:text-gray-300">{label}</span>
      <div className="flex items-center gap-2">
        <div className={`h-2 w-2 rounded-full ${ok ? 'bg-green-500' : 'bg-red-500'}`} />
        <span className="text-sm text-gray-600 dark:text-gray-400">{value}</span>
      </div>
    </div>
  );
}

function SettingDisplay({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <p className="text-sm font-medium text-gray-500 dark:text-gray-400">{label}</p>
      <p className="mt-1 text-sm text-gray-900 dark:text-white">{value}</p>
    </div>
  );
}

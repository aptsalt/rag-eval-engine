const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

async function fetchAPI<T>(path: string, options?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    ...options,
    headers: {
      'Content-Type': 'application/json',
      ...options?.headers,
    },
  });

  if (!response.ok) {
    let detail = '';
    try {
      const body = await response.json();
      detail = body.detail || body.message || JSON.stringify(body);
    } catch {
      detail = await response.text().catch(() => '');
    }

    if (response.status === 502 || response.status === 503 || detail.toLowerCase().includes('qdrant')) {
      throw new Error('Qdrant vector database is unavailable. Check that Qdrant is running on port 6333.');
    }
    if (detail.toLowerCase().includes('ollama') || detail.toLowerCase().includes('connection refused')) {
      throw new Error('Ollama LLM server is unreachable. Check that Ollama is running on port 11434.');
    }
    throw new Error(detail || `Request failed (${response.status})`);
  }

  return response.json() as Promise<T>;
}

export interface HealthStatus {
  status: string;
  ollama: string;
  embedding_model: string;
  default_llm: string;
  eval_enabled: boolean;
}

export interface Collection {
  name: string;
  doc_count: number;
  total_chunks: number;
  total_tokens: number;
  vectors_count: number;
}

export interface OllamaModel {
  name: string;
  size: number | null;
  modified_at: string | null;
}

export interface IngestResponse {
  job_id: string;
  status: string;
  message: string;
}

export interface JobStatus {
  job_id: string;
  status: string;
  total_files: number;
  processed_files: number;
  total_chunks: number;
  error: string | null;
}

export interface Source {
  index: number;
  text: string;
  source: string;
  score: number;
  chunk_index: number;
}

export interface EvalScores {
  faithfulness: number;
  relevance: number;
  hallucination_rate: number;
  context_precision: number;
  context_recall: number | null;
  latency_retrieval_ms: number;
  latency_generation_ms: number;
}

export interface QueryResponse {
  query_id: string;
  answer: string;
  sources: Source[];
  eval_scores: EvalScores | null;
  tokens_used: number;
  latency_ms: number;
  model: string;
}

export interface MetricsResponse {
  total_queries: number;
  avg_faithfulness: number | null;
  avg_relevance: number | null;
  avg_hallucination_rate: number | null;
  avg_latency_ms: number | null;
  p50_latency_ms: number | null;
  p95_latency_ms: number | null;
  time_series: TimeSeriesPoint[];
}

export interface TimeSeriesPoint {
  query_id: string;
  timestamp: number;
  faithfulness: number | null;
  relevance: number | null;
  hallucination_rate: number | null;
  latency_ms: number | null;
  tokens_used: number | null;
}

export interface AppSettings {
  embedding_model: string;
  chunking_strategy: string;
  chunk_size: number;
  chunk_overlap: number;
  default_model: string;
  hybrid_alpha: number;
  default_top_k: number;
  eval_on_query: boolean;
  eval_lightweight: boolean;
  use_reranker: boolean;
}

export interface TestSet {
  id: string;
  name: string;
  collection: string;
  question_count: number;
  created_at: string;
}

export interface EvalRun {
  id: string;
  test_set_id: string;
  status: string;
  total: number;
  completed: number;
  created_at: string;
}

export interface RetrievalResult {
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

export const api = {
  health: () => fetchAPI<HealthStatus>('/health'),

  getCollections: () => fetchAPI<Collection[]>('/api/collections'),

  deleteCollection: (name: string) =>
    fetchAPI<{ status: string }>(`/api/collections/${name}`, { method: 'DELETE' }),

  getModels: () => fetchAPI<OllamaModel[]>('/api/models'),

  ingestFiles: async (
    files: File[],
    collection: string,
    options?: {
      chunking_strategy?: string;
      chunk_size?: number;
      chunk_overlap?: number;
    },
  ): Promise<IngestResponse> => {
    const formData = new FormData();
    files.forEach((file) => formData.append('files', file));

    const params = new URLSearchParams();
    params.set('collection', collection);
    if (options?.chunking_strategy) params.set('chunking_strategy', options.chunking_strategy);
    if (options?.chunk_size) params.set('chunk_size', String(options.chunk_size));
    if (options?.chunk_overlap) params.set('chunk_overlap', String(options.chunk_overlap));

    const response = await fetch(
      `${API_BASE}/api/ingest?${params.toString()}`,
      { method: 'POST', body: formData },
    );

    if (!response.ok) {
      let detail = '';
      try {
        const body = await response.json();
        detail = body.detail || body.message || '';
      } catch {
        detail = await response.text().catch(() => '');
      }
      throw new Error(detail || `Upload failed (${response.status})`);
    }
    return response.json() as Promise<IngestResponse>;
  },

  getJobStatus: (jobId: string) => fetchAPI<JobStatus>(`/api/ingest/${jobId}`),

  query: (params: {
    query: string;
    collection: string;
    top_k?: number;
    model?: string;
    evaluate?: boolean;
  }) =>
    fetchAPI<QueryResponse>('/api/query', {
      method: 'POST',
      body: JSON.stringify({ ...params, stream: false }),
    }),

  queryStream: (params: {
    query: string;
    collection: string;
    top_k?: number;
    model?: string;
  }) => {
    return fetch(`${API_BASE}/api/query`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ ...params, stream: true }),
    });
  },

  retrieve: async (params: {
    query: string;
    collection: string;
    top_k?: number;
    alpha?: number;
    source_filter?: string;
  }): Promise<RetrievalResult[]> => {
    const data = await fetchAPI<{ query: string; chunks: RetrievalResult[] }>('/api/retrieve', {
      method: 'POST',
      body: JSON.stringify(params),
    });
    return data.chunks;
  },

  getMetrics: (collection?: string, limit?: number) => {
    const params = new URLSearchParams();
    if (collection) params.set('collection', collection);
    if (limit) params.set('limit', String(limit));
    const query = params.toString();
    return fetchAPI<MetricsResponse>(`/api/metrics${query ? `?${query}` : ''}`);
  },

  getSettings: () => fetchAPI<AppSettings>('/api/settings'),

  getTestSets: () => fetchAPI<TestSet[]>('/api/test-sets'),

  getTestSet: (id: string) => fetchAPI<TestSet>(`/api/test-sets/${id}`),

  deleteTestSet: (id: string) =>
    fetchAPI<{ status: string }>(`/api/test-sets/${id}`, { method: 'DELETE' }),

  autoGenerateTestSet: (
    collection: string,
    numQuestions: number,
    testSetName: string,
    model?: string,
  ) =>
    fetchAPI<{ questions: Array<{ question: string }>; count: number }>('/api/test-sets/auto-generate', {
      method: 'POST',
      body: JSON.stringify({
        collection,
        num_questions: numQuestions,
        test_set_name: testSetName,
        model,
      }),
    }),

  runBatchEval: (testSetId: string, model?: string) =>
    fetchAPI<{ status: string; test_set_id: string }>('/api/evaluate/batch', {
      method: 'POST',
      body: JSON.stringify({ test_set_id: testSetId, model }),
    }),

  getEvalRuns: (testSetId?: string) => {
    const params = testSetId ? `?test_set_id=${testSetId}` : '';
    return fetchAPI<EvalRun[]>(`/api/evaluate/runs${params}`);
  },
};

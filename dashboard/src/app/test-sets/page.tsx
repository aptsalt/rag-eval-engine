'use client';

import { useState, useEffect, useCallback } from 'react';
import { api, type TestSet, type Collection, type EvalRun } from '@/lib/api';
import { cn } from '@/lib/utils';

export default function TestSetsPage() {
  const [testSets, setTestSets] = useState<TestSet[]>([]);
  const [evalRuns, setEvalRuns] = useState<EvalRun[]>([]);
  const [collections, setCollections] = useState<Collection[]>([]);
  const [loading, setLoading] = useState(true);
  const [showCreate, setShowCreate] = useState(false);
  const [generating, setGenerating] = useState(false);
  const [runningEval, setRunningEval] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Create form state
  const [newName, setNewName] = useState('');
  const [newCollection, setNewCollection] = useState('documents');
  const [numQuestions, setNumQuestions] = useState(10);

  const loadData = useCallback(async () => {
    try {
      const [ts, cols, runs] = await Promise.all([
        api.getTestSets(),
        api.getCollections(),
        api.getEvalRuns(),
      ]);
      setTestSets(ts);
      setCollections(cols);
      setEvalRuns(runs);
    } catch {
      // API unavailable
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadData();
  }, [loadData]);

  const handleAutoGenerate = async () => {
    if (!newName.trim()) {
      setError('Test set name is required');
      return;
    }
    setGenerating(true);
    setError(null);

    try {
      await api.autoGenerateTestSet(newCollection, numQuestions, newName);
      setNewName('');
      setShowCreate(false);
      await loadData();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Generation failed');
    } finally {
      setGenerating(false);
    }
  };

  const handleRunEval = async (testSetId: string) => {
    setRunningEval(testSetId);
    try {
      await api.runBatchEval(testSetId);
      await loadData();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Evaluation failed');
    } finally {
      setRunningEval(null);
    }
  };

  const handleDeleteTestSet = async (id: string) => {
    if (!confirm('Delete this test set?')) return;
    try {
      await api.deleteTestSet(id);
      await loadData();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Delete failed');
    }
  };

  const getRunsForTestSet = (testSetId: string) =>
    evalRuns.filter((r) => r.test_set_id === testSetId);

  if (loading) {
    return (
      <div className="flex h-full items-center justify-center">
        <div className="h-8 w-8 animate-spin rounded-full border-4 border-brand-200 border-t-brand-600" />
      </div>
    );
  }

  return (
    <div className="space-y-8">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900 dark:text-white">Test Sets</h1>
          <p className="mt-1 text-sm text-gray-500 dark:text-gray-400">
            Manage evaluation test sets and run batch evaluations
          </p>
        </div>
        <button
          onClick={() => setShowCreate(!showCreate)}
          className="rounded-lg bg-brand-600 px-4 py-2 text-sm font-medium text-white hover:bg-brand-700 transition-colors"
        >
          {showCreate ? 'Cancel' : 'New Test Set'}
        </button>
      </div>

      {error && (
        <div className="rounded-lg bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 p-3 text-sm text-red-700 dark:text-red-400">
          {error}
          <button onClick={() => setError(null)} className="ml-2 text-red-500 hover:text-red-700">dismiss</button>
        </div>
      )}

      {/* Create Test Set */}
      {showCreate && (
        <div className="rounded-xl border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800 p-6">
          <h2 className="text-lg font-semibold text-gray-900 dark:text-white">Auto-Generate Test Set</h2>
          <p className="mt-1 text-sm text-gray-500 dark:text-gray-400">
            Automatically generate evaluation questions from your documents using the LLM
          </p>
          <div className="mt-4 grid gap-4 sm:grid-cols-3">
            <div>
              <label className="block text-xs font-medium text-gray-500 dark:text-gray-400 mb-1">Name</label>
              <input
                type="text"
                value={newName}
                onChange={(e) => setNewName(e.target.value)}
                placeholder="e.g., baseline-v1"
                className="w-full rounded-lg border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-700 px-3 py-2 text-sm dark:text-white focus:border-brand-500 focus:outline-none focus:ring-1 focus:ring-brand-500"
              />
            </div>
            <div>
              <label className="block text-xs font-medium text-gray-500 dark:text-gray-400 mb-1">Collection</label>
              <select
                value={newCollection}
                onChange={(e) => setNewCollection(e.target.value)}
                className="w-full rounded-lg border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-700 px-3 py-2 text-sm dark:text-white focus:border-brand-500 focus:outline-none focus:ring-1 focus:ring-brand-500"
              >
                {collections.length === 0 && <option value="documents">documents</option>}
                {collections.map((col) => (
                  <option key={col.name} value={col.name}>{col.name}</option>
                ))}
              </select>
            </div>
            <div>
              <label className="block text-xs font-medium text-gray-500 dark:text-gray-400 mb-1">
                Questions: {numQuestions}
              </label>
              <input
                type="range"
                min={3}
                max={50}
                value={numQuestions}
                onChange={(e) => setNumQuestions(Number(e.target.value))}
                className="w-full accent-brand-600 mt-2"
              />
            </div>
          </div>
          <button
            onClick={handleAutoGenerate}
            disabled={generating || !newName.trim()}
            className={cn(
              'mt-4 rounded-lg px-4 py-2 text-sm font-medium text-white transition-colors',
              generating || !newName.trim()
                ? 'cursor-not-allowed bg-gray-400'
                : 'bg-brand-600 hover:bg-brand-700'
            )}
          >
            {generating ? 'Generating...' : `Generate ${numQuestions} Questions`}
          </button>
        </div>
      )}

      {/* Test Sets List */}
      {testSets.length === 0 ? (
        <div className="rounded-xl border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800 p-8 text-center">
          <svg className="mx-auto h-12 w-12 text-gray-400" fill="none" viewBox="0 0 24 24" strokeWidth={1} stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" d="M9 12h3.75M9 15h3.75M9 18h3.75m3 .75H18a2.25 2.25 0 0 0 2.25-2.25V6.108c0-1.135-.845-2.098-1.976-2.192a48.424 48.424 0 0 0-1.123-.08m-5.801 0c-.065.21-.1.433-.1.664 0 .414.336.75.75.75h4.5a.75.75 0 0 0 .75-.75 2.25 2.25 0 0 0-.1-.664m-5.8 0A2.251 2.251 0 0 1 13.5 2.25H15c1.012 0 1.867.668 2.15 1.586m-5.8 0c-.376.023-.75.05-1.124.08C9.095 4.01 8.25 4.973 8.25 6.108V8.25m0 0H4.875c-.621 0-1.125.504-1.125 1.125v11.25c0 .621.504 1.125 1.125 1.125h9.75c.621 0 1.125-.504 1.125-1.125V9.375c0-.621-.504-1.125-1.125-1.125H8.25ZM6.75 12h.008v.008H6.75V12Zm0 3h.008v.008H6.75V15Zm0 3h.008v.008H6.75V18Z" />
          </svg>
          <h3 className="mt-4 text-lg font-medium text-gray-900 dark:text-white">No test sets yet</h3>
          <p className="mt-2 text-sm text-gray-500 dark:text-gray-400">
            Create a test set to start running batch evaluations on your RAG pipeline.
          </p>
        </div>
      ) : (
        <div className="space-y-4">
          {testSets.map((ts) => {
            const runs = getRunsForTestSet(ts.id);
            const latestRun = runs[0];
            return (
              <div key={ts.id} className="rounded-xl border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800 p-5">
                <div className="flex items-start justify-between">
                  <div>
                    <h3 className="font-semibold text-gray-900 dark:text-white">{ts.name}</h3>
                    <div className="mt-1 flex items-center gap-3 text-xs text-gray-500 dark:text-gray-400">
                      <span>{ts.question_count} questions</span>
                      <span>Collection: {ts.collection}</span>
                      <span>Created: {new Date(ts.created_at).toLocaleDateString()}</span>
                    </div>
                  </div>
                  <div className="flex items-center gap-2">
                    <button
                      onClick={() => handleRunEval(ts.id)}
                      disabled={runningEval === ts.id}
                      className={cn(
                        'rounded-lg px-3 py-1.5 text-xs font-medium text-white transition-colors',
                        runningEval === ts.id
                          ? 'cursor-not-allowed bg-gray-400'
                          : 'bg-green-600 hover:bg-green-700'
                      )}
                    >
                      {runningEval === ts.id ? 'Running...' : 'Run Eval'}
                    </button>
                    <button
                      onClick={() => handleDeleteTestSet(ts.id)}
                      className="rounded-lg p-1.5 text-gray-400 hover:bg-red-50 dark:hover:bg-red-900/20 hover:text-red-600 transition-colors"
                    >
                      <svg className="h-4 w-4" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" d="m14.74 9-.346 9m-4.788 0L9.26 9m9.968-3.21c.342.052.682.107 1.022.166m-1.022-.165L18.16 19.673a2.25 2.25 0 0 1-2.244 2.077H8.084a2.25 2.25 0 0 1-2.244-2.077L4.772 5.79m14.456 0a48.108 48.108 0 0 0-3.478-.397m-12 .562c.34-.059.68-.114 1.022-.165m0 0a48.11 48.11 0 0 1 3.478-.397m7.5 0v-.916c0-1.18-.91-2.164-2.09-2.201a51.964 51.964 0 0 0-3.32 0c-1.18.037-2.09 1.022-2.09 2.201v.916m7.5 0a48.667 48.667 0 0 0-7.5 0" />
                      </svg>
                    </button>
                  </div>
                </div>

                {/* Eval Runs */}
                {runs.length > 0 && (
                  <div className="mt-4 border-t border-gray-100 dark:border-gray-700 pt-3">
                    <p className="text-xs font-medium text-gray-500 dark:text-gray-400 mb-2">
                      Evaluation Runs ({runs.length})
                    </p>
                    <div className="space-y-2">
                      {runs.slice(0, 3).map((run) => (
                        <div key={run.id} className="flex items-center justify-between rounded-lg bg-gray-50 dark:bg-gray-700/50 px-3 py-2 text-xs">
                          <div className="flex items-center gap-2">
                            <div className={cn(
                              'h-2 w-2 rounded-full',
                              run.status === 'completed' ? 'bg-green-500' :
                              run.status === 'running' ? 'bg-blue-500 animate-pulse' : 'bg-yellow-500'
                            )} />
                            <span className="text-gray-600 dark:text-gray-300">{run.status}</span>
                          </div>
                          <div className="flex items-center gap-3 text-gray-500 dark:text-gray-400">
                            <span>{run.completed}/{run.total} queries</span>
                            <span>{new Date(run.created_at).toLocaleDateString()}</span>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {latestRun === undefined && (
                  <p className="mt-3 text-xs text-gray-400 dark:text-gray-500 italic">
                    No evaluations run yet. Click &ldquo;Run Eval&rdquo; to start.
                  </p>
                )}
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}

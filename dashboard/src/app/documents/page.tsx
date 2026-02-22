'use client';

import { useState, useEffect, useCallback, useRef } from 'react';
import { api, type Collection, type JobStatus } from '@/lib/api';
import { cn, formatNumber } from '@/lib/utils';
import { useToast } from '@/components/toast';
import { SkeletonCard, SkeletonRow } from '@/components/skeleton';

const MAX_FILE_SIZE_MB = 50;
const MAX_FILES_PER_UPLOAD = 20;
const MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024;
const ACCEPTED_EXTENSIONS = '.pdf,.docx,.txt,.md,.py,.js,.ts,.tsx,.json,.yaml,.yml,.toml,.csv,.html,.css';

const FILE_TYPE_ICONS: Record<string, { icon: string; color: string }> = {
  pdf: { icon: 'PDF', color: 'bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-400' },
  docx: { icon: 'DOC', color: 'bg-blue-100 text-blue-700 dark:bg-blue-900/30 dark:text-blue-400' },
  txt: { icon: 'TXT', color: 'bg-gray-100 text-gray-700 dark:bg-gray-700 dark:text-gray-300' },
  md: { icon: 'MD', color: 'bg-purple-100 text-purple-700 dark:bg-purple-900/30 dark:text-purple-400' },
  py: { icon: 'PY', color: 'bg-yellow-100 text-yellow-700 dark:bg-yellow-900/30 dark:text-yellow-400' },
  js: { icon: 'JS', color: 'bg-yellow-100 text-yellow-700 dark:bg-yellow-900/30 dark:text-yellow-400' },
  ts: { icon: 'TS', color: 'bg-blue-100 text-blue-700 dark:bg-blue-900/30 dark:text-blue-400' },
  tsx: { icon: 'TSX', color: 'bg-blue-100 text-blue-700 dark:bg-blue-900/30 dark:text-blue-400' },
  json: { icon: 'JSON', color: 'bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400' },
  yaml: { icon: 'YAML', color: 'bg-orange-100 text-orange-700 dark:bg-orange-900/30 dark:text-orange-400' },
  yml: { icon: 'YML', color: 'bg-orange-100 text-orange-700 dark:bg-orange-900/30 dark:text-orange-400' },
  csv: { icon: 'CSV', color: 'bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400' },
  html: { icon: 'HTML', color: 'bg-orange-100 text-orange-700 dark:bg-orange-900/30 dark:text-orange-400' },
};

function getFileTypeInfo(filename: string) {
  const ext = filename.split('.').pop()?.toLowerCase() || '';
  return FILE_TYPE_ICONS[ext] || { icon: 'FILE', color: 'bg-gray-100 text-gray-700 dark:bg-gray-700 dark:text-gray-300' };
}

export default function DocumentsPage() {
  const { addToast } = useToast();
  const [collections, setCollections] = useState<Collection[]>([]);
  const [loading, setLoading] = useState(true);
  const [uploading, setUploading] = useState(false);
  const [dragOver, setDragOver] = useState(false);
  const [collectionName, setCollectionName] = useState('documents');
  const [chunkingStrategy, setChunkingStrategy] = useState('recursive');
  const [chunkSize, setChunkSize] = useState(512);
  const [activeJob, setActiveJob] = useState<JobStatus | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [selectedFiles, setSelectedFiles] = useState<File[]>([]);
  const [showAdvanced, setShowAdvanced] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const loadCollections = useCallback(async () => {
    try {
      const cols = await api.getCollections();
      setCollections(cols);
    } catch {
      // API not available
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadCollections();
  }, [loadCollections]);

  useEffect(() => {
    if (!activeJob || activeJob.status === 'completed' || activeJob.status === 'failed') return;

    const interval = setInterval(async () => {
      try {
        const status = await api.getJobStatus(activeJob.job_id);
        setActiveJob(status);
        if (status.status === 'completed' || status.status === 'failed') {
          loadCollections();
        }
      } catch {
        // ignore
      }
    }, 2000);

    return () => clearInterval(interval);
  }, [activeJob, loadCollections]);

  const validateFiles = (files: File[]): string | null => {
    if (files.length === 0) return 'No files selected';
    if (files.length > MAX_FILES_PER_UPLOAD) {
      return `Too many files. Maximum ${MAX_FILES_PER_UPLOAD} files per upload, got ${files.length}.`;
    }
    for (const file of files) {
      if (file.size > MAX_FILE_SIZE_BYTES) {
        return `File "${file.name}" exceeds ${MAX_FILE_SIZE_MB}MB limit (${(file.size / 1024 / 1024).toFixed(1)}MB).`;
      }
    }
    return null;
  };

  const handleFilesSelected = (files: FileList | File[]) => {
    const fileArray = Array.from(files);
    const validationError = validateFiles(fileArray);
    if (validationError) {
      setError(validationError);
      return;
    }
    setError(null);
    setSelectedFiles(fileArray);
  };

  const handleUpload = async () => {
    if (selectedFiles.length === 0) return;

    const validationError = validateFiles(selectedFiles);
    if (validationError) {
      setError(validationError);
      return;
    }

    setUploading(true);
    setError(null);

    try {
      const response = await api.ingestFiles(selectedFiles, collectionName, {
        chunking_strategy: chunkingStrategy,
        chunk_size: chunkSize,
      });
      addToast(`Upload started: ${selectedFiles.length} file${selectedFiles.length > 1 ? 's' : ''}`, 'success');
      setActiveJob({
        job_id: response.job_id,
        status: response.status,
        total_files: selectedFiles.length,
        processed_files: 0,
        total_chunks: 0,
        error: null,
      });
      setSelectedFiles([]);
      if (fileInputRef.current) {
        fileInputRef.current.value = '';
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Upload failed');
    } finally {
      setUploading(false);
    }
  };

  const handleDelete = async (name: string) => {
    if (!confirm(`Delete collection "${name}" and all its documents?`)) return;
    try {
      await api.deleteCollection(name);
      addToast(`Collection "${name}" deleted`, 'success');
      loadCollections();
    } catch (err) {
      addToast(err instanceof Error ? err.message : 'Delete failed', 'error');
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(false);
    if (e.dataTransfer.files.length > 0) {
      handleFilesSelected(e.dataTransfer.files);
    }
  };

  const removeFile = (index: number) => {
    setSelectedFiles((prev) => prev.filter((_, i) => i !== index));
  };

  const totalSize = selectedFiles.reduce((sum, f) => sum + f.size, 0);
  const totalDocs = collections.reduce((sum, c) => sum + c.doc_count, 0);
  const totalChunks = collections.reduce((sum, c) => sum + c.total_chunks, 0);
  const totalVectors = collections.reduce((sum, c) => sum + c.vectors_count, 0);

  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-2xl font-bold text-gray-900 dark:text-white">Document Management</h1>
        <p className="mt-1 text-sm text-gray-500 dark:text-gray-400">Upload and manage your document collections</p>
      </div>

      {/* Stats Overview */}
      {collections.length > 0 && (
        <div className="grid gap-4 sm:grid-cols-4">
          <StatCard label="Collections" value={String(collections.length)} icon="folder" />
          <StatCard label="Documents" value={formatNumber(totalDocs, 0)} icon="file" />
          <StatCard label="Chunks" value={formatNumber(totalChunks, 0)} icon="puzzle" />
          <StatCard label="Vectors" value={formatNumber(totalVectors, 0)} icon="cube" />
        </div>
      )}

      {/* Upload Section */}
      <div className="rounded-xl border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800 p-6">
        <h2 className="text-lg font-semibold text-gray-900 dark:text-white">Upload Documents</h2>
        <div className="mt-4 flex flex-wrap items-center gap-4">
          <div>
            <label className="block text-xs font-medium text-gray-500 dark:text-gray-400 mb-1">Collection</label>
            <input
              type="text"
              value={collectionName}
              onChange={(e) => setCollectionName(e.target.value)}
              className="rounded-lg border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-700 px-3 py-1.5 text-sm dark:text-white focus:border-brand-500 focus:outline-none focus:ring-1 focus:ring-brand-500"
              placeholder="Collection name"
            />
          </div>
          <div>
            <label className="block text-xs font-medium text-gray-500 dark:text-gray-400 mb-1">Chunking</label>
            <select
              value={chunkingStrategy}
              onChange={(e) => setChunkingStrategy(e.target.value)}
              className="rounded-lg border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-700 px-3 py-1.5 text-sm dark:text-white focus:border-brand-500 focus:outline-none focus:ring-1 focus:ring-brand-500"
            >
              <option value="recursive">Recursive (recommended)</option>
              <option value="fixed">Fixed-size</option>
              <option value="semantic">Semantic</option>
            </select>
          </div>
          <button
            type="button"
            onClick={() => setShowAdvanced(!showAdvanced)}
            className="mt-4 text-xs text-brand-600 dark:text-brand-400 hover:text-brand-700 dark:hover:text-brand-300"
          >
            {showAdvanced ? 'Hide' : 'Show'} advanced
          </button>
        </div>

        {showAdvanced && (
          <div className="mt-3 flex items-center gap-4 rounded-lg bg-gray-50 dark:bg-gray-700/50 p-3">
            <div>
              <label className="block text-xs font-medium text-gray-500 dark:text-gray-400 mb-1">
                Chunk size: {chunkSize} tokens
              </label>
              <input
                type="range"
                min={128}
                max={2048}
                step={64}
                value={chunkSize}
                onChange={(e) => setChunkSize(Number(e.target.value))}
                className="w-40 accent-brand-600"
              />
            </div>
          </div>
        )}

        <div
          onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
          onDragLeave={() => setDragOver(false)}
          onDrop={handleDrop}
          className={cn(
            'mt-4 flex flex-col items-center justify-center rounded-xl border-2 border-dashed p-12 transition-all',
            dragOver
              ? 'border-brand-500 bg-brand-50 dark:bg-brand-900/20 scale-[1.01]'
              : 'border-gray-300 dark:border-gray-600 hover:border-gray-400 dark:hover:border-gray-500'
          )}
        >
          <svg className="h-12 w-12 text-gray-400" fill="none" viewBox="0 0 24 24" strokeWidth={1} stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" d="M12 16.5V9.75m0 0 3 3m-3-3-3 3M6.75 19.5a4.5 4.5 0 0 1-1.41-8.775 5.25 5.25 0 0 1 10.233-2.33 3 3 0 0 1 3.758 3.848A3.752 3.752 0 0 1 18 19.5H6.75Z" />
          </svg>
          <p className="mt-2 text-sm text-gray-600 dark:text-gray-400">
            <label className="cursor-pointer font-medium text-brand-600 dark:text-brand-400 hover:text-brand-500">
              Select files
              <input
                ref={fileInputRef}
                type="file"
                multiple
                onChange={(e) => {
                  if (e.target.files) handleFilesSelected(e.target.files);
                }}
                className="sr-only"
                accept={ACCEPTED_EXTENSIONS}
              />
            </label>
            {' or drag and drop'}
          </p>
          <p className="mt-1 text-xs text-gray-400">
            PDF, DOCX, TXT, MD, and code files — max {MAX_FILE_SIZE_MB}MB each, up to {MAX_FILES_PER_UPLOAD} files
          </p>
        </div>

        {/* File preview */}
        {selectedFiles.length > 0 && (
          <div className="mt-4 rounded-lg border border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-700/50 p-4">
            <div className="flex items-center justify-between">
              <span className="text-sm font-medium text-gray-700 dark:text-gray-300">
                {selectedFiles.length} file{selectedFiles.length > 1 ? 's' : ''} selected ({(totalSize / 1024 / 1024).toFixed(1)}MB total)
              </span>
              <button
                type="button"
                onClick={() => {
                  setSelectedFiles([]);
                  if (fileInputRef.current) fileInputRef.current.value = '';
                }}
                className="text-xs text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200"
              >
                Clear all
              </button>
            </div>
            <ul className="mt-2 space-y-1">
              {selectedFiles.map((file, index) => {
                const typeInfo = getFileTypeInfo(file.name);
                return (
                  <li key={index} className="flex items-center justify-between text-xs text-gray-600 dark:text-gray-400 py-1">
                    <div className="flex items-center gap-2">
                      <span className={cn('rounded px-1.5 py-0.5 text-[10px] font-bold', typeInfo.color)}>
                        {typeInfo.icon}
                      </span>
                      <span className="truncate">{file.name}</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <span className="shrink-0 text-gray-400">{(file.size / 1024).toFixed(0)}KB</span>
                      <button
                        onClick={() => removeFile(index)}
                        className="text-gray-400 hover:text-red-500 transition-colors"
                      >
                        <svg className="h-3.5 w-3.5" fill="none" viewBox="0 0 24 24" strokeWidth={2} stroke="currentColor">
                          <path strokeLinecap="round" strokeLinejoin="round" d="M6 18 18 6M6 6l12 12" />
                        </svg>
                      </button>
                    </div>
                  </li>
                );
              })}
            </ul>
            <button
              type="button"
              onClick={handleUpload}
              disabled={uploading}
              className={cn(
                'mt-3 w-full rounded-lg px-4 py-2.5 text-sm font-medium text-white transition-colors',
                uploading
                  ? 'cursor-not-allowed bg-gray-400'
                  : 'bg-brand-600 hover:bg-brand-700'
              )}
            >
              {uploading ? 'Uploading...' : `Upload ${selectedFiles.length} file${selectedFiles.length > 1 ? 's' : ''}`}
            </button>
          </div>
        )}

        {error && (
          <div className="mt-4 rounded-lg bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 p-3 text-sm text-red-700 dark:text-red-400">
            {error}
          </div>
        )}

        {activeJob && (
          <div className={cn(
            'mt-4 rounded-lg border p-4',
            activeJob.status === 'completed'
              ? 'bg-green-50 dark:bg-green-900/20 border-green-200 dark:border-green-800'
              : activeJob.status === 'failed'
                ? 'bg-red-50 dark:bg-red-900/20 border-red-200 dark:border-red-800'
                : 'bg-blue-50 dark:bg-blue-900/20 border-blue-200 dark:border-blue-800'
          )}>
            <div className="flex items-center justify-between">
              <span className={cn(
                'text-sm font-medium',
                activeJob.status === 'completed' ? 'text-green-800 dark:text-green-400' :
                activeJob.status === 'failed' ? 'text-red-800 dark:text-red-400' :
                'text-blue-800 dark:text-blue-400'
              )}>
                {activeJob.status === 'completed' ? 'Ingestion Complete' :
                 activeJob.status === 'failed' ? 'Ingestion Failed' : 'Processing...'}
              </span>
              <span className="text-xs text-blue-600 dark:text-blue-400">
                {activeJob.processed_files}/{activeJob.total_files} files
              </span>
            </div>
            <div className="mt-2 h-2 w-full rounded-full bg-gray-200 dark:bg-gray-600 overflow-hidden">
              <div
                className={cn(
                  'h-2 rounded-full transition-all duration-500',
                  activeJob.status === 'failed' ? 'bg-red-500' :
                  activeJob.status === 'completed' ? 'bg-green-500' : 'bg-blue-600'
                )}
                style={{
                  width: `${activeJob.total_files > 0 ? (activeJob.processed_files / activeJob.total_files) * 100 : 0}%`,
                }}
              />
            </div>
            {activeJob.total_chunks > 0 && (
              <p className="mt-1 text-xs text-blue-600 dark:text-blue-400">{activeJob.total_chunks} chunks created</p>
            )}
            {activeJob.error && (
              <p className="mt-1 text-xs text-red-600 dark:text-red-400">{activeJob.error}</p>
            )}
          </div>
        )}
      </div>

      {/* Collections List */}
      <div>
        <h2 className="text-lg font-semibold text-gray-900 dark:text-white">Collections</h2>
        {collections.length === 0 ? (
          <div className="mt-4 rounded-xl border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800 p-8 text-center">
            <svg className="mx-auto h-12 w-12 text-gray-400" fill="none" viewBox="0 0 24 24" strokeWidth={1} stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" d="M2.25 12.75V12A2.25 2.25 0 0 1 4.5 9.75h15A2.25 2.25 0 0 1 21.75 12v.75m-8.69-6.44-2.12-2.12a1.5 1.5 0 0 0-1.061-.44H4.5A2.25 2.25 0 0 0 2.25 6v12a2.25 2.25 0 0 0 2.25 2.25h15A2.25 2.25 0 0 0 21.75 18V9a2.25 2.25 0 0 0-2.25-2.25h-5.379a1.5 1.5 0 0 1-1.06-.44Z" />
            </svg>
            <p className="mt-3 text-sm text-gray-500 dark:text-gray-400">No collections yet. Upload documents to get started.</p>
          </div>
        ) : (
          <div className="mt-4 grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
            {collections.map((col) => (
              <div key={col.name} className="rounded-xl border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800 p-5 hover:shadow-md transition-shadow">
                <div className="flex items-start justify-between">
                  <div>
                    <h3 className="font-semibold text-gray-900 dark:text-white">{col.name}</h3>
                    <p className="mt-0.5 text-xs text-gray-500 dark:text-gray-400">{col.doc_count} documents</p>
                  </div>
                  <button
                    onClick={() => handleDelete(col.name)}
                    className="rounded-lg p-1 text-gray-400 hover:bg-red-50 dark:hover:bg-red-900/20 hover:text-red-600 transition-colors"
                    title="Delete collection"
                  >
                    <svg className="h-4 w-4" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" d="m14.74 9-.346 9m-4.788 0L9.26 9m9.968-3.21c.342.052.682.107 1.022.166m-1.022-.165L18.16 19.673a2.25 2.25 0 0 1-2.244 2.077H8.084a2.25 2.25 0 0 1-2.244-2.077L4.772 5.79m14.456 0a48.108 48.108 0 0 0-3.478-.397m-12 .562c.34-.059.68-.114 1.022-.165m0 0a48.11 48.11 0 0 1 3.478-.397m7.5 0v-.916c0-1.18-.91-2.164-2.09-2.201a51.964 51.964 0 0 0-3.32 0c-1.18.037-2.09 1.022-2.09 2.201v.916m7.5 0a48.667 48.667 0 0 0-7.5 0" />
                    </svg>
                  </button>
                </div>
                <div className="mt-4 grid grid-cols-3 gap-2">
                  <Stat label="Chunks" value={formatNumber(col.total_chunks, 0)} />
                  <Stat label="Tokens" value={formatNumber(col.total_tokens, 0)} />
                  <Stat label="Vectors" value={formatNumber(col.vectors_count, 0)} />
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="text-center">
      <p className="text-lg font-semibold text-gray-900 dark:text-white">{value}</p>
      <p className="text-xs text-gray-500 dark:text-gray-400">{label}</p>
    </div>
  );
}

function StatCard({ label, value, icon }: { label: string; value: string; icon: string }) {
  const iconMap: Record<string, JSX.Element> = {
    folder: (
      <svg className="h-5 w-5" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" d="M2.25 12.75V12A2.25 2.25 0 0 1 4.5 9.75h15A2.25 2.25 0 0 1 21.75 12v.75m-8.69-6.44-2.12-2.12a1.5 1.5 0 0 0-1.061-.44H4.5A2.25 2.25 0 0 0 2.25 6v12a2.25 2.25 0 0 0 2.25 2.25h15A2.25 2.25 0 0 0 21.75 18V9a2.25 2.25 0 0 0-2.25-2.25h-5.379a1.5 1.5 0 0 1-1.06-.44Z" />
      </svg>
    ),
    file: (
      <svg className="h-5 w-5" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" d="M19.5 14.25v-2.625a3.375 3.375 0 0 0-3.375-3.375h-1.5A1.125 1.125 0 0 1 13.5 7.125v-1.5a3.375 3.375 0 0 0-3.375-3.375H8.25m2.25 0H5.625c-.621 0-1.125.504-1.125 1.125v17.25c0 .621.504 1.125 1.125 1.125h12.75c.621 0 1.125-.504 1.125-1.125V11.25a9 9 0 0 0-9-9Z" />
      </svg>
    ),
    puzzle: (
      <svg className="h-5 w-5" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" d="M14.25 6.087c0-.355.186-.676.401-.959.221-.29.349-.634.349-1.003 0-1.036-1.007-1.875-2.25-1.875s-2.25.84-2.25 1.875c0 .369.128.713.349 1.003.215.283.401.604.401.959v0a.64.64 0 0 1-.657.643 48.39 48.39 0 0 1-4.163-.3c.186 1.613.293 3.25.315 4.907a.656.656 0 0 1-.658.663v0c-.355 0-.676-.186-.959-.401a1.647 1.647 0 0 0-1.003-.349c-1.036 0-1.875 1.007-1.875 2.25s.84 2.25 1.875 2.25c.369 0 .713-.128 1.003-.349.283-.215.604-.401.959-.401v0c.31 0 .555.26.532.57a48.039 48.039 0 0 1-.642 5.056c1.518.19 3.058.309 4.616.354a.64.64 0 0 0 .657-.643v0c0-.355-.186-.676-.401-.959a1.647 1.647 0 0 1-.349-1.003c0-1.035 1.008-1.875 2.25-1.875 1.243 0 2.25.84 2.25 1.875 0 .369-.128.713-.349 1.003-.215.283-.4.604-.4.959v0c0 .333.277.599.61.58a48.1 48.1 0 0 0 5.427-.63 48.05 48.05 0 0 0 .582-4.717.532.532 0 0 0-.533-.57v0c-.355 0-.676.186-.959.401-.29.221-.634.349-1.003.349-1.035 0-1.875-1.007-1.875-2.25s.84-2.25 1.875-2.25c.37 0 .713.128 1.003.349.283.215.604.401.96.401v0a.656.656 0 0 0 .658-.663 48.422 48.422 0 0 0-.37-5.36c-1.886.342-3.81.574-5.766.689a.578.578 0 0 1-.61-.58v0Z" />
      </svg>
    ),
    cube: (
      <svg className="h-5 w-5" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" d="m21 7.5-9-5.25L3 7.5m18 0-9 5.25m9-5.25v9l-9 5.25M3 7.5l9 5.25M3 7.5v9l9 5.25m0-9v9" />
      </svg>
    ),
  };

  return (
    <div className="rounded-xl border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800 p-4">
      <div className="flex items-center gap-3">
        <div className="rounded-lg bg-brand-50 dark:bg-brand-900/20 p-2 text-brand-600 dark:text-brand-400">
          {iconMap[icon]}
        </div>
        <div>
          <p className="text-2xl font-bold text-gray-900 dark:text-white">{value}</p>
          <p className="text-xs text-gray-500 dark:text-gray-400">{label}</p>
        </div>
      </div>
    </div>
  );
}

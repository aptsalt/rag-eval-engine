import { cn } from '@/lib/utils';

export function SkeletonCard({ className }: { className?: string }) {
  return (
    <div className={cn('rounded-xl border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800 p-5', className)}>
      <div className="h-3 w-24 rounded bg-gray-200 dark:bg-gray-700 animate-pulse" />
      <div className="mt-3 h-8 w-20 rounded bg-gray-200 dark:bg-gray-700 animate-pulse" />
      <div className="mt-2 h-2.5 w-40 rounded bg-gray-100 dark:bg-gray-700/50 animate-pulse" />
    </div>
  );
}

export function SkeletonChart({ className }: { className?: string }) {
  return (
    <div className={cn('rounded-xl border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800 p-6', className)}>
      <div className="h-4 w-40 rounded bg-gray-200 dark:bg-gray-700 animate-pulse" />
      <div className="mt-2 h-3 w-64 rounded bg-gray-100 dark:bg-gray-700/50 animate-pulse" />
      <div className="mt-6 flex items-end gap-2 h-48">
        {Array.from({ length: 12 }).map((_, i) => (
          <div
            key={i}
            className="flex-1 rounded-t bg-gray-200 dark:bg-gray-700 animate-pulse"
            style={{ height: `${20 + Math.random() * 80}%`, animationDelay: `${i * 100}ms` }}
          />
        ))}
      </div>
    </div>
  );
}

export function SkeletonText({ lines = 3, className }: { lines?: number; className?: string }) {
  return (
    <div className={cn('space-y-2', className)}>
      {Array.from({ length: lines }).map((_, i) => (
        <div
          key={i}
          className="h-3 rounded bg-gray-200 dark:bg-gray-700 animate-pulse"
          style={{ width: `${60 + Math.random() * 40}%`, animationDelay: `${i * 75}ms` }}
        />
      ))}
    </div>
  );
}

export function SkeletonRow({ className }: { className?: string }) {
  return (
    <div className={cn('flex items-center gap-4 rounded-xl border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800 p-4', className)}>
      <div className="h-8 w-8 rounded-full bg-gray-200 dark:bg-gray-700 animate-pulse shrink-0" />
      <div className="flex-1 space-y-2">
        <div className="h-3 w-3/4 rounded bg-gray-200 dark:bg-gray-700 animate-pulse" />
        <div className="h-2.5 w-1/2 rounded bg-gray-100 dark:bg-gray-700/50 animate-pulse" />
      </div>
      <div className="h-6 w-16 rounded bg-gray-200 dark:bg-gray-700 animate-pulse shrink-0" />
    </div>
  );
}

import type { Metadata } from 'next';
import { Inter } from 'next/font/google';
import './globals.css';
import { Sidebar } from '@/components/sidebar';
import { ToastProvider } from '@/components/toast';

const inter = Inter({ subsets: ['latin'] });

export const metadata: Metadata = {
  title: 'RAG Eval Engine — Production RAG with Built-in Evaluation',
  description: 'Production-grade Retrieval-Augmented Generation system with hybrid search, multi-model support, and continuous quality evaluation.',
  keywords: ['RAG', 'retrieval augmented generation', 'evaluation', 'LLM', 'vector search', 'NLP', 'semantic cache', 'MCP'],
  openGraph: {
    title: 'RAG Eval Engine',
    description: 'Production RAG with hybrid retrieval, multi-provider LLM routing, semantic caching, and built-in evaluation',
    type: 'website',
  },
  robots: 'index, follow',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" suppressHydrationWarning>
      <head>
        <meta name="theme-color" content="#4f46e5" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
      </head>
      <body className={inter.className}>
        <ToastProvider>
          <div className="flex h-screen bg-gray-50 dark:bg-gray-900">
            <Sidebar />
            <main className="flex-1 overflow-y-auto p-6 md:p-8">{children}</main>
          </div>
        </ToastProvider>
      </body>
    </html>
  );
}

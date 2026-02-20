import type { Metadata } from 'next';
import { Inter } from 'next/font/google';
import './globals.css';
import { Sidebar } from '@/components/sidebar';

const inter = Inter({ subsets: ['latin'] });

export const metadata: Metadata = {
  title: 'RAG Eval Engine — Production RAG with Built-in Evaluation',
  description: 'Production-grade Retrieval-Augmented Generation system with hybrid search, multi-model support, and continuous quality evaluation.',
  keywords: ['RAG', 'retrieval augmented generation', 'evaluation', 'LLM', 'vector search', 'NLP'],
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body className={inter.className}>
        <div className="flex h-screen bg-gray-50 dark:bg-gray-900">
          <Sidebar />
          <main className="flex-1 overflow-y-auto p-6 md:p-8">{children}</main>
        </div>
      </body>
    </html>
  );
}

'use client';

import JobsTable from '@/components/JobsTable';
import { TopBar, MainContent } from '@/components/layout';
import Link from 'next/link';
import { useRef, useState } from 'react';

export default function Dashboard() {
  const [search, setSearch] = useState('');
  const scrollRoot = useRef<HTMLDivElement>(null);
  return (
    <>
      <TopBar>
        <div>
          <h1 className="text-base sm:text-lg">Queue</h1>
        </div>
        <div className="flex-1"></div>
        <div>
          <Link
            href="/jobs/new"
            className="text-white bg-slate-600 px-2 sm:px-3 py-1 rounded-md text-sm sm:text-base whitespace-nowrap"
          >
            <span className="sm:hidden">+ New Job</span>
            <span className="hidden sm:inline">New Training Job</span>
          </Link>
        </div>
      </TopBar>
      <div className="absolute top-12 left-0 w-full h-16 z-10 bg-gray-950 border-b border-gray-800 px-2 sm:px-4 flex items-center gap-3">
        <input
          type="search"
          aria-label="Search training jobs"
          placeholder="Search training jobs…"
          title="Match all space-separated substrings in job names (case-insensitive)"
          value={search}
          onChange={event => setSearch(event.target.value)}
          className="min-w-0 flex-1 rounded-md bg-gray-800 border border-gray-600 text-sm text-gray-200 placeholder-gray-500 px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
        />
      </div>
      <MainContent ref={scrollRoot} belowTopBar style={{ top: '7rem' }}>
        <JobsTable search={search} pageSize={100} scrollRoot={scrollRoot} />
      </MainContent>
    </>
  );
}

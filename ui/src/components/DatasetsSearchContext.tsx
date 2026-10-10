'use client';

import { createContext, useContext, useState, type ReactNode } from 'react';

type DatasetsSearchState = {
  search: string;
  setSearch: (search: string) => void;
};

const DatasetsSearchContext = createContext<DatasetsSearchState | null>(null);

export function DatasetsSearchProvider({ children }: { children: ReactNode }) {
  const [search, setSearch] = useState('');

  return <DatasetsSearchContext.Provider value={{ search, setSearch }}>{children}</DatasetsSearchContext.Provider>;
}

export function useDatasetsSearch() {
  const context = useContext(DatasetsSearchContext);
  if (!context) throw new Error('Dataset search requires DatasetsSearchProvider.');
  return context;
}

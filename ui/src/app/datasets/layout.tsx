import { DatasetsSearchProvider } from '@/components/DatasetsSearchContext';

export default function DatasetsLayout({ children }: { children: React.ReactNode }) {
  return <DatasetsSearchProvider>{children}</DatasetsSearchProvider>;
}

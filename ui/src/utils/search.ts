/** Case-insensitive AND matching, with terms allowed in any order. */
export function searchTerms(query: string): string[] {
  return query.toLowerCase().trim().split(/\s+/).filter(Boolean);
}

export function matchesSearch(text: string, query: string): boolean {
  const normalized = text.toLowerCase();
  return searchTerms(query).every(term => normalized.includes(term));
}

export function filterSelectOption(option: { label: string; value: string }, query: string): boolean {
  return matchesSearch(`${option.label} ${option.value}`, query);
}

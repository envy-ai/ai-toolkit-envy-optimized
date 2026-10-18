/** Case-insensitive AND matching, with terms allowed in any order. */
export function matchesSearch(text: string, query: string): boolean {
  const normalized = text.toLowerCase();
  return query.toLowerCase().trim().split(/\s+/).filter(Boolean).every(term => normalized.includes(term));
}

export function filterSelectOption(option: { label: string; value: string }, query: string): boolean {
  return matchesSearch(`${option.label} ${option.value}`, query);
}

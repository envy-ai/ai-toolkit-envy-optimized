/** Keep fast-server generation and the Next.js cache reader on the same variant. */
export type ThumbnailVariant = 'square' | 'fit';

export function getThumbnailVariant(value: string | null): ThumbnailVariant {
  return value === 'fit' ? 'fit' : 'square';
}

export function getThumbnailCacheBasename(sourceBasename: string, variant: ThumbnailVariant): string {
  return variant === 'fit' ? `${sourceBasename}.fit` : sourceBasename;
}

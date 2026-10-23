import sharp from 'sharp';

/** Resize only the in-memory observation; uploaded and dataset source files stay unchanged. */
export async function normalizeAssistantImage(input: string | Buffer, megapixels: number) {
  if (!Number.isFinite(megapixels) || megapixels < 0.000001 || megapixels > 16)
    throw new Error('Assistant image size must be between 0.000001 and 16 megapixels.');
  const pipeline = sharp(input, { limitInputPixels: 64 * 1024 * 1024 });
  const metadata = await pipeline.metadata();
  if (!metadata.width || !metadata.height) throw new Error('Assistant image dimensions are unavailable.');
  const swapAxes = metadata.orientation !== undefined && metadata.orientation >= 5 && metadata.orientation <= 8;
  const width = swapAxes ? metadata.height : metadata.width;
  const height = swapAxes ? metadata.width : metadata.height;
  const pixels = Math.floor(megapixels * 1_000_000);
  // Also bound each axis by the area budget for extremely thin, one-pixel images.
  const scale = Math.min(
    1,
    Math.sqrt(pixels / (width * height)),
    4096 / width,
    4096 / height,
    pixels / width,
    pixels / height,
  );
  const resizedWidth = Math.max(1, Math.floor(width * scale));
  const resizedHeight = Math.max(1, Math.floor(height * scale));
  const { data, info } = await pipeline
    .rotate()
    .resize({ width: resizedWidth, height: resizedHeight, fit: 'fill', withoutEnlargement: true })
    .png()
    .toBuffer({ resolveWithObject: true });
  return { data, info, metadata };
}

/** The toolkit normalizes visual observations to PNG before provider transport. */
export function inspectImageBytes(bytes: Uint8Array, mime: string, _name?: string) {
  const png = Buffer.from(bytes);
  if (
    mime !== 'image/png' ||
    png.length < 33 ||
    !png.subarray(0, 8).equals(Buffer.from([137, 80, 78, 71, 13, 10, 26, 10])) ||
    png.readUInt32BE(8) !== 13 ||
    png.toString('ascii', 12, 16) !== 'IHDR'
  ) {
    throw new Error('Assistant observations must be normalized PNG images.');
  }
  return { width: png.readUInt32BE(16), height: png.readUInt32BE(20) };
}

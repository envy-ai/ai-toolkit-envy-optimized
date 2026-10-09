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

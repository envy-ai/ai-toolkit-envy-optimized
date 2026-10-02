import test from "node:test";
import assert from "node:assert/strict";

import {
  adjustUnitValue,
  buildColorNoiseMask,
  buildMixedMask,
  calculateMegapixelSize,
  compositeContrastCopy,
  compositeMatchedAdditiveNoise,
  createPerlinNoise,
  createSobelMask,
  gaussianBlurMask,
  mixColorNoiseMasks,
} from "../public/synth.js";

test("linear mask adjustment clips both ends", () => {
  assert.equal(adjustUnitValue(0.25, 3, 0), 0);
  assert.equal(adjustUnitValue(0.75, 3, 0), 1);
  assert.equal(adjustUnitValue(0.5, 1, 0.2), 0.7);
});

test("one-megapixel sizing preserves aspect ratio", () => {
  assert.deepEqual(calculateMegapixelSize(1024, 1024), { width: 1024, height: 1024 });
  const landscape = calculateMegapixelSize(1600, 900);
  const portrait = calculateMegapixelSize(900, 1600);

  assert.ok(Math.abs(landscape.width * landscape.height - 1024 * 1024) < 1024);
  assert.ok(Math.abs(landscape.width / landscape.height - 16 / 9) < 0.002);
  assert.equal(portrait.width, landscape.height);
  assert.equal(portrait.height, landscape.width);
});

test("Perlin noise is deterministic, bounded, and seed-dependent", () => {
  const first = createPerlinNoise(32, 24, 16, 42);
  const same = createPerlinNoise(32, 24, 16, 42);
  const other = createPerlinNoise(32, 24, 16, 43);
  assert.deepEqual(first, same);
  assert.notDeepEqual(first, other);
  assert.ok(first.every((value) => value >= 0 && value <= 1));
});

test("Sobel mask is dark on a flat image and bright on an edge", () => {
  const width = 5;
  const height = 5;
  const flat = new Uint8ClampedArray(width * height * 4).fill(127);
  for (let index = 3; index < flat.length; index += 4) flat[index] = 255;
  assert.ok(createSobelMask(flat, width, height).every((value) => value === 0));

  const edge = flat.slice();
  for (let y = 0; y < height; y += 1) {
    for (let x = 3; x < width; x += 1) {
      const offset = (y * width + x) * 4;
      edge[offset] = 255;
      edge[offset + 1] = 255;
      edge[offset + 2] = 255;
    }
  }
  assert.ok(createSobelMask(edge, width, height)[2 * width + 2] > 0.4);
});

test("Gaussian blur spreads an edge before mask tone adjustment", () => {
  const width = 9;
  const height = width;
  const impulse = new Float32Array(width * height);
  const center = 4 * width + 4;
  impulse[center] = 1;

  const blurred = gaussianBlurMask(impulse, width, height, 1);

  assert.ok(blurred[center] < 1);
  assert.ok(blurred[center] > blurred[center + 1]);
  assert.ok(blurred[center + 1] > 0);
  assert.ok(Math.abs(blurred[center - 1] - blurred[center + 1]) < 1e-7);
  assert.deepEqual(gaussianBlurMask(impulse, width, height, 0), impulse);
});

test("edge selector chooses large in black and small in white", () => {
  const result = buildMixedMask(
    Float32Array.of(0.2, 0.2),
    Float32Array.of(0.8, 0.8),
    Float32Array.of(0, 1),
  );
  assert.ok(Math.abs(result[0] - 0.2) < 1e-6);
  assert.ok(Math.abs(result[1] - 0.8) < 1e-6);
});

test("color variation ranges from grayscale to independent RGB masks", () => {
  const channels = [
    Float32Array.of(0.2),
    Float32Array.of(0.5),
    Float32Array.of(0.8),
  ];
  const grayscale = buildColorNoiseMask(channels, 0);
  const fullColor = buildColorNoiseMask(channels, 1);

  for (const value of grayscale) assert.ok(Math.abs(value - 0.2) < 1e-6);
  for (const [index, expected] of [0.2, 0.5, 0.8].entries()) {
    assert.ok(Math.abs(fullColor[index] - expected) < 1e-6);
  }
});

test("edge selection mixes each color channel", () => {
  const mixed = mixColorNoiseMasks(
    Float32Array.of(0.1, 0.2, 0.3),
    Float32Array.of(0.7, 0.8, 0.9),
    Float32Array.of(0.5),
  );

  assert.ok(Math.abs(mixed[0] - 0.4) < 1e-6);
  assert.ok(Math.abs(mixed[1] - 0.5) < 1e-6);
  assert.ok(Math.abs(mixed[2] - 0.6) < 1e-6);
});

test("composite preserves alpha and moves pixels toward the contrast copy", () => {
  const source = Uint8ClampedArray.of(64, 128, 192, 91);
  const unchanged = compositeContrastCopy(source, Float32Array.of(0), { copyContrast: 2 });
  const changed = compositeContrastCopy(source, Float32Array.of(1), { copyContrast: 2 });
  assert.deepEqual(unchanged, source);
  assert.ok(changed[0] < source[0]);
  assert.ok(changed[2] > source[2]);
  assert.equal(changed[3], 91);
});

test("composite accepts per-channel color masks", () => {
  const source = Uint8ClampedArray.of(64, 128, 192, 255);
  const changed = compositeContrastCopy(
    source,
    Float32Array.of(1, 0, 1),
    { copyContrast: 2 },
  );

  assert.ok(changed[0] < source[0]);
  assert.equal(changed[1], source[1]);
  assert.ok(changed[2] > source[2]);
});

test("matched additive synthesis preserves per-channel means and contrast", () => {
  const width = 64;
  const height = 64;
  const pixelCount = width * height;
  const source = new Uint8ClampedArray(pixelCount * 4);
  const mask = new Float32Array(pixelCount * 3);
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const index = y * width + x;
      source[index * 4] = 24 + Math.round(180 * x / (width - 1));
      source[index * 4 + 1] = 18 + Math.round(150 * y / (height - 1));
      source[index * 4 + 2] = 12 + Math.round(110 * (x + y) / (width + height - 2));
      source[index * 4 + 3] = 77 + (index % 179);
      mask[index * 3] = 0.5 + 0.35 * Math.sin(x / 3);
      mask[index * 3 + 1] = 0.5 + 0.35 * Math.cos(y / 4);
      mask[index * 3 + 2] = 0.5 + 0.35 * Math.sin((x + y) / 5);
    }
  }

  const result = compositeMatchedAdditiveNoise(source, mask, {
    copyContrast: 1.6,
    copyBrightness: -0.15,
    effectStrength: 1.2,
  });
  const moments = (pixels, channel) => {
    const values = Array.from(
      { length: pixelCount },
      (_, index) => pixels[index * 4 + channel] / 255,
    );
    const mean = values.reduce((sum, value) => sum + value, 0) / values.length;
    const variance = values.reduce((sum, value) => sum + (value - mean) ** 2, 0) / values.length;
    return [mean, Math.sqrt(variance)];
  };

  for (let channel = 0; channel < 3; channel += 1) {
    const sourceMoments = moments(source, channel);
    const resultMoments = moments(result, channel);
    assert.ok(Math.abs(sourceMoments[0] - resultMoments[0]) < 2 / 255);
    assert.ok(Math.abs(sourceMoments[1] - resultMoments[1]) < 2 / 255);
  }
  assert.ok(result.some((value, index) => index % 4 !== 3 && value !== source[index]));
  for (let index = 3; index < result.length; index += 4) {
    assert.equal(result[index], source[index]);
  }
});

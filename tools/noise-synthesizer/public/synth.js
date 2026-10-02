export function clamp01(value) {
  return Math.max(0, Math.min(1, value));
}

export function adjustUnitValue(value, contrast = 1, brightness = 0) {
  return clamp01((value - 0.5) * contrast + 0.5 + brightness);
}

export const ONE_MEGAPIXEL = 1024 * 1024;

export function calculateMegapixelSize(width, height, targetPixels = ONE_MEGAPIXEL) {
  if (width <= 0 || height <= 0 || targetPixels <= 0) {
    throw new RangeError("Image dimensions and target pixel count must be positive");
  }
  const aspectRatio = width / height;
  const resizedWidth = Math.max(1, Math.round(Math.sqrt(targetPixels * aspectRatio)));
  const resizedHeight = Math.max(1, Math.round(targetPixels / resizedWidth));
  return { width: resizedWidth, height: resizedHeight };
}

function mulberry32(seed) {
  let state = seed >>> 0;
  return () => {
    state += 0x6d2b79f5;
    let value = state;
    value = Math.imul(value ^ (value >>> 15), value | 1);
    value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
    return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
  };
}

function makePermutation(seed) {
  const random = mulberry32(seed);
  const values = Uint16Array.from({ length: 256 }, (_, index) => index);
  for (let index = values.length - 1; index > 0; index -= 1) {
    const other = Math.floor(random() * (index + 1));
    [values[index], values[other]] = [values[other], values[index]];
  }
  const permutation = new Uint16Array(512);
  for (let index = 0; index < permutation.length; index += 1) {
    permutation[index] = values[index & 255];
  }
  return permutation;
}

function fade(value) {
  return value * value * value * (value * (value * 6 - 15) + 10);
}

function lerp(left, right, amount) {
  return left + (right - left) * amount;
}

function gradient(hash, x, y) {
  switch (hash & 7) {
    case 0: return x + y;
    case 1: return -x + y;
    case 2: return x - y;
    case 3: return -x - y;
    case 4: return x;
    case 5: return -x;
    case 6: return y;
    default: return -y;
  }
}

export function createPerlinNoise(width, height, period, seed = 1) {
  if (width <= 0 || height <= 0) {
    throw new RangeError("Noise dimensions must be positive");
  }
  if (!Number.isFinite(period) || period <= 0) {
    throw new RangeError("Noise period must be greater than zero");
  }

  const permutation = makePermutation(seed);
  const output = new Float32Array(width * height);

  for (let y = 0; y < height; y += 1) {
    const sampleY = y / period;
    const cellY = Math.floor(sampleY);
    const localY = sampleY - cellY;
    const wrappedY = cellY & 255;
    const blendY = fade(localY);

    for (let x = 0; x < width; x += 1) {
      const sampleX = x / period;
      const cellX = Math.floor(sampleX);
      const localX = sampleX - cellX;
      const wrappedX = cellX & 255;
      const blendX = fade(localX);

      const topLeft = permutation[permutation[wrappedX] + wrappedY];
      const topRight = permutation[permutation[wrappedX + 1] + wrappedY];
      const bottomLeft = permutation[permutation[wrappedX] + wrappedY + 1];
      const bottomRight = permutation[permutation[wrappedX + 1] + wrappedY + 1];

      const top = lerp(
        gradient(topLeft, localX, localY),
        gradient(topRight, localX - 1, localY),
        blendX,
      );
      const bottom = lerp(
        gradient(bottomLeft, localX, localY - 1),
        gradient(bottomRight, localX - 1, localY - 1),
        blendX,
      );
      output[y * width + x] = clamp01(lerp(top, bottom, blendY) * 0.5 + 0.5);
    }
  }

  return output;
}

export function createSobelMask(rgba, width, height) {
  if (rgba.length !== width * height * 4) {
    throw new RangeError("RGBA data does not match the supplied dimensions");
  }

  const luminance = new Float32Array(width * height);
  for (let index = 0; index < luminance.length; index += 1) {
    const offset = index * 4;
    luminance[index] = (
      rgba[offset] * 0.2126 + rgba[offset + 1] * 0.7152 + rgba[offset + 2] * 0.0722
    );
  }

  const output = new Float32Array(width * height);
  const at = (x, y) => luminance[
    Math.max(0, Math.min(height - 1, y)) * width + Math.max(0, Math.min(width - 1, x))
  ];

  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const gx = (
        -at(x - 1, y - 1) + at(x + 1, y - 1)
        - 2 * at(x - 1, y) + 2 * at(x + 1, y)
        - at(x - 1, y + 1) + at(x + 1, y + 1)
      );
      const gy = (
        -at(x - 1, y - 1) - 2 * at(x, y - 1) - at(x + 1, y - 1)
        + at(x - 1, y + 1) + 2 * at(x, y + 1) + at(x + 1, y + 1)
      );
      output[y * width + x] = clamp01(Math.hypot(gx, gy) / 1020);
    }
  }
  return output;
}

export function gaussianBlurMask(mask, width, height, sigma) {
  if (width <= 0 || height <= 0 || mask.length !== width * height) {
    throw new RangeError("Mask data does not match the supplied dimensions");
  }
  if (!Number.isFinite(sigma) || sigma < 0) {
    throw new RangeError("Gaussian blur must be zero or greater");
  }
  if (sigma === 0) return Float32Array.from(mask);

  const radius = Math.max(1, Math.ceil(sigma * 3));
  const kernel = new Float32Array(radius * 2 + 1);
  const denominator = 2 * sigma * sigma;
  let kernelSum = 0;
  for (let offset = -radius; offset <= radius; offset += 1) {
    const weight = Math.exp(-(offset * offset) / denominator);
    kernel[offset + radius] = weight;
    kernelSum += weight;
  }
  for (let index = 0; index < kernel.length; index += 1) {
    kernel[index] /= kernelSum;
  }

  // A separable convolution produces the same 2D Gaussian with far less work.
  // Clamp coordinates at image boundaries so the mask does not darken there.
  const horizontal = new Float32Array(mask.length);
  const output = new Float32Array(mask.length);
  for (let y = 0; y < height; y += 1) {
    const row = y * width;
    for (let x = 0; x < width; x += 1) {
      let value = 0;
      for (let offset = -radius; offset <= radius; offset += 1) {
        const sampleX = Math.max(0, Math.min(width - 1, x + offset));
        value += mask[row + sampleX] * kernel[offset + radius];
      }
      horizontal[row + x] = value;
    }
  }
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      let value = 0;
      for (let offset = -radius; offset <= radius; offset += 1) {
        const sampleY = Math.max(0, Math.min(height - 1, y + offset));
        value += horizontal[sampleY * width + x] * kernel[offset + radius];
      }
      output[y * width + x] = value;
    }
  }
  return output;
}

export function buildColorNoiseMask(
  noiseChannels,
  colorVariation = 0,
  contrast = 1,
  brightness = 0,
) {
  if (!Array.isArray(noiseChannels) || noiseChannels.length !== 3) {
    throw new RangeError("Color noise requires three channel masks");
  }
  const pixelCount = noiseChannels[0].length;
  if (noiseChannels.some((channel) => channel.length !== pixelCount)) {
    throw new RangeError("Color-noise channel sizes differ");
  }

  const variation = clamp01(colorVariation);
  const grayscale = noiseChannels[0];
  const output = new Float32Array(pixelCount * 3);
  for (let index = 0; index < pixelCount; index += 1) {
    for (let channel = 0; channel < 3; channel += 1) {
      const colored = lerp(grayscale[index], noiseChannels[channel][index], variation);
      output[index * 3 + channel] = adjustUnitValue(colored, contrast, brightness);
    }
  }
  return output;
}

export function mixColorNoiseMasks(
  largeColorMask,
  smallColorMask,
  edgeMask,
  edgeContrast = 1,
  edgeBrightness = 0,
) {
  if (
    largeColorMask.length !== smallColorMask.length
    || largeColorMask.length !== edgeMask.length * 3
  ) {
    throw new RangeError("Color masks and edge mask dimensions differ");
  }

  const output = new Float32Array(largeColorMask.length);
  for (let index = 0; index < edgeMask.length; index += 1) {
    const edge = adjustUnitValue(edgeMask[index], edgeContrast, edgeBrightness);
    for (let channel = 0; channel < 3; channel += 1) {
      const colorIndex = index * 3 + channel;
      output[colorIndex] = lerp(
        largeColorMask[colorIndex],
        smallColorMask[colorIndex],
        edge,
      );
    }
  }
  return output;
}

export function buildMixedMask(largeNoise, smallNoise, edgeMask, controls = {}) {
  if (largeNoise.length !== smallNoise.length || largeNoise.length !== edgeMask.length) {
    throw new RangeError("All masks must have equal lengths");
  }

  const {
    largeContrast = 1,
    largeBrightness = 0,
    smallContrast = 1,
    smallBrightness = 0,
    edgeContrast = 1,
    edgeBrightness = 0,
  } = controls;
  const output = new Float32Array(largeNoise.length);

  for (let index = 0; index < output.length; index += 1) {
    const large = adjustUnitValue(largeNoise[index], largeContrast, largeBrightness);
    const small = adjustUnitValue(smallNoise[index], smallContrast, smallBrightness);
    const edge = adjustUnitValue(edgeMask[index], edgeContrast, edgeBrightness);
    output[index] = lerp(large, small, edge);
  }
  return output;
}

export function compositeContrastCopy(sourceRgba, mixedMask, options = {}) {
  const pixelCount = sourceRgba.length / 4;
  const hasColorMask = mixedMask.length === pixelCount * 3;
  if (!hasColorMask && mixedMask.length !== pixelCount) {
    throw new RangeError("Source pixels and mask dimensions do not match");
  }

  const {
    copyContrast = 1.25,
    copyBrightness = 0,
    effectStrength = 1,
  } = options;
  const output = new Uint8ClampedArray(sourceRgba.length);

  for (let index = 0; index < pixelCount; index += 1) {
    const offset = index * 4;
    for (let channel = 0; channel < 3; channel += 1) {
      const maskIndex = hasColorMask ? index * 3 + channel : index;
      const amount = clamp01(mixedMask[maskIndex] * effectStrength);
      const original = sourceRgba[offset + channel] / 255;
      const contrastCopy = adjustUnitValue(original, copyContrast, copyBrightness);
      output[offset + channel] = Math.round(lerp(original, contrastCopy, amount) * 255);
    }
    output[offset + 3] = sourceRgba[offset + 3];
  }
  return output;
}

const LUMA_WEIGHTS = [0.2126, 0.7152, 0.0722];

function imageChannelMoments(sourceRgba) {
  const pixelCount = sourceRgba.length / 4;
  const means = [0, 0, 0];
  const secondMoments = [0, 0, 0];
  for (let index = 0; index < pixelCount; index += 1) {
    const offset = index * 4;
    for (let channel = 0; channel < 3; channel += 1) {
      const value = sourceRgba[offset + channel] / 255;
      means[channel] += value;
      secondMoments[channel] += value * value;
    }
  }
  for (let channel = 0; channel < 3; channel += 1) {
    means[channel] /= pixelCount;
    secondMoments[channel] /= pixelCount;
  }
  return {
    means,
    standardDeviations: secondMoments.map((value, channel) => (
      Math.sqrt(Math.max(0, value - means[channel] * means[channel]))
    )),
  };
}

function residualLuminanceRms(candidateRgba, targetRgba) {
  if (candidateRgba.length !== targetRgba.length) {
    throw new RangeError("Candidate and target dimensions do not match");
  }
  const pixelCount = targetRgba.length / 4;
  let targetMean = 0;
  let deltaMean = 0;
  for (let index = 0; index < pixelCount; index += 1) {
    const offset = index * 4;
    let targetLuma = 0;
    let candidateLuma = 0;
    for (let channel = 0; channel < 3; channel += 1) {
      targetLuma += targetRgba[offset + channel] / 255 * LUMA_WEIGHTS[channel];
      candidateLuma += candidateRgba[offset + channel] / 255 * LUMA_WEIGHTS[channel];
    }
    targetMean += targetLuma;
    deltaMean += candidateLuma - targetLuma;
  }
  targetMean /= pixelCount;
  deltaMean /= pixelCount;

  let covariance = 0;
  let targetVariance = 0;
  for (let index = 0; index < pixelCount; index += 1) {
    const offset = index * 4;
    let targetLuma = 0;
    let candidateLuma = 0;
    for (let channel = 0; channel < 3; channel += 1) {
      targetLuma += targetRgba[offset + channel] / 255 * LUMA_WEIGHTS[channel];
      candidateLuma += candidateRgba[offset + channel] / 255 * LUMA_WEIGHTS[channel];
    }
    const centeredTarget = targetLuma - targetMean;
    covariance += centeredTarget * (candidateLuma - targetLuma - deltaMean);
    targetVariance += centeredTarget * centeredTarget;
  }
  const slope = targetVariance > 0 ? covariance / targetVariance : 0;

  let residualSquared = 0;
  for (let index = 0; index < pixelCount; index += 1) {
    const offset = index * 4;
    let targetLuma = 0;
    let candidateLuma = 0;
    for (let channel = 0; channel < 3; channel += 1) {
      targetLuma += targetRgba[offset + channel] / 255 * LUMA_WEIGHTS[channel];
      candidateLuma += candidateRgba[offset + channel] / 255 * LUMA_WEIGHTS[channel];
    }
    const residual = (
      candidateLuma - targetLuma
      - slope * (targetLuma - targetMean)
      - deltaMean
    );
    residualSquared += residual * residual;
  }
  return Math.sqrt(residualSquared / pixelCount);
}

function decorrelateColorNoise(mixedMask, targetRgba) {
  const pixelCount = targetRgba.length / 4;
  if (mixedMask.length !== pixelCount * 3) {
    throw new RangeError("Additive synthesis requires an RGB noise mask");
  }

  const targetMeans = [0, 0, 0];
  const noiseMeans = [0, 0, 0];
  for (let index = 0; index < pixelCount; index += 1) {
    for (let channel = 0; channel < 3; channel += 1) {
      targetMeans[channel] += targetRgba[index * 4 + channel] / 255;
      noiseMeans[channel] += mixedMask[index * 3 + channel];
    }
  }
  for (let channel = 0; channel < 3; channel += 1) {
    targetMeans[channel] /= pixelCount;
    noiseMeans[channel] /= pixelCount;
  }

  const covariances = [0, 0, 0];
  const variances = [0, 0, 0];
  for (let index = 0; index < pixelCount; index += 1) {
    for (let channel = 0; channel < 3; channel += 1) {
      const centeredTarget = targetRgba[index * 4 + channel] / 255 - targetMeans[channel];
      const centeredNoise = mixedMask[index * 3 + channel] - noiseMeans[channel];
      covariances[channel] += centeredNoise * centeredTarget;
      variances[channel] += centeredTarget * centeredTarget;
    }
  }
  const slopes = covariances.map((value, channel) => (
    variances[channel] > 0 ? value / variances[channel] : 0
  ));

  const output = new Float64Array(mixedMask.length);
  for (let index = 0; index < pixelCount; index += 1) {
    for (let channel = 0; channel < 3; channel += 1) {
      const centeredTarget = targetRgba[index * 4 + channel] / 255 - targetMeans[channel];
      output[index * 3 + channel] = (
        mixedMask[index * 3 + channel]
        - noiseMeans[channel]
        - slopes[channel] * centeredTarget
      );
    }
  }
  return output;
}

function scaleNoiseToLuminanceRms(noise, desiredRms) {
  const pixelCount = noise.length / 3;
  let mean = 0;
  for (let index = 0; index < pixelCount; index += 1) {
    mean += (
      noise[index * 3] * LUMA_WEIGHTS[0]
      + noise[index * 3 + 1] * LUMA_WEIGHTS[1]
      + noise[index * 3 + 2] * LUMA_WEIGHTS[2]
    );
  }
  mean /= pixelCount;
  let variance = 0;
  for (let index = 0; index < pixelCount; index += 1) {
    const value = (
      noise[index * 3] * LUMA_WEIGHTS[0]
      + noise[index * 3 + 1] * LUMA_WEIGHTS[1]
      + noise[index * 3 + 2] * LUMA_WEIGHTS[2]
    );
    variance += (value - mean) ** 2;
  }
  const currentRms = Math.sqrt(variance / pixelCount);
  const scale = currentRms > 0 ? desiredRms / currentRms : 0;
  for (let index = 0; index < noise.length; index += 1) noise[index] *= scale;
}

function evaluateClippedChannel(targetRgba, noise, channel, targetMean, scale, offset) {
  const pixelCount = targetRgba.length / 4;
  let sum = 0;
  let squareSum = 0;
  let activeTargetSum = 0;
  let activeCount = 0;
  let activeValueTargetSum = 0;
  let activeValueSum = 0;

  for (let index = 0; index < pixelCount; index += 1) {
    const centeredTarget = targetRgba[index * 4 + channel] / 255 - targetMean;
    const raw = targetMean + scale * centeredTarget + offset + noise[index * 3 + channel];
    const value = clamp01(raw);
    sum += value;
    squareSum += value * value;
    if (raw > 0 && raw < 1) {
      activeTargetSum += centeredTarget;
      activeCount += 1;
      activeValueTargetSum += value * centeredTarget;
      activeValueSum += value;
    }
  }

  const mean = sum / pixelCount;
  const standardDeviation = Math.sqrt(Math.max(0, squareSum / pixelCount - mean * mean));
  const meanScaleDerivative = activeTargetSum / pixelCount;
  const meanOffsetDerivative = activeCount / pixelCount;
  const safeDenominator = Math.max(standardDeviation * 2, 1e-12);
  const stdScaleDerivative = (
    2 * activeValueTargetSum / pixelCount - 2 * mean * meanScaleDerivative
  ) / safeDenominator;
  const stdOffsetDerivative = (
    2 * activeValueSum / pixelCount - 2 * mean * meanOffsetDerivative
  ) / safeDenominator;
  return {
    mean,
    standardDeviation,
    meanScaleDerivative,
    meanOffsetDerivative,
    stdScaleDerivative,
    stdOffsetDerivative,
  };
}

function solveClippedChannelMoments(targetRgba, noise, channel, targetMean, targetStd) {
  let scale = 1;
  let offset = 0;
  for (let iteration = 0; iteration < 20; iteration += 1) {
    const result = evaluateClippedChannel(
      targetRgba, noise, channel, targetMean, scale, offset,
    );
    const meanError = result.mean - targetMean;
    const stdError = result.standardDeviation - targetStd;
    if (Math.max(Math.abs(meanError), Math.abs(stdError)) < 1e-9) break;

    const determinant = (
      result.meanScaleDerivative * result.stdOffsetDerivative
      - result.meanOffsetDerivative * result.stdScaleDerivative
    );
    if (Math.abs(determinant) < 1e-12) break;
    let scaleStep = (
      -result.stdOffsetDerivative * meanError
      + result.meanOffsetDerivative * stdError
    ) / determinant;
    let offsetStep = (
      result.stdScaleDerivative * meanError
      - result.meanScaleDerivative * stdError
    ) / determinant;
    scaleStep = Math.max(-0.5, Math.min(0.5, scaleStep));
    offsetStep = Math.max(-0.1, Math.min(0.1, offsetStep));
    scale = Math.max(0, Math.min(4, scale + scaleStep));
    offset = Math.max(-1, Math.min(1, offset + offsetStep));
  }
  return { scale, offset };
}

function matchChannelMoments(targetRgba, noise) {
  const pixelCount = targetRgba.length / 4;
  const targetMoments = imageChannelMoments(targetRgba);
  const parameters = targetMoments.means.map((targetMean, channel) => (
    solveClippedChannelMoments(
      targetRgba,
      noise,
      channel,
      targetMean,
      targetMoments.standardDeviations[channel],
    )
  ));
  const output = new Uint8ClampedArray(targetRgba.length);
  for (let index = 0; index < pixelCount; index += 1) {
    for (let channel = 0; channel < 3; channel += 1) {
      const centeredTarget = (
        targetRgba[index * 4 + channel] / 255 - targetMoments.means[channel]
      );
      const value = (
        targetMoments.means[channel]
        + parameters[channel].scale * centeredTarget
        + parameters[channel].offset
        + noise[index * 3 + channel]
      );
      output[index * 4 + channel] = Math.round(clamp01(value) * 255);
    }
    output[index * 4 + 3] = targetRgba[index * 4 + 3];
  }
  return output;
}

export function compositeMatchedAdditiveNoise(sourceRgba, mixedMask, options = {}) {
  const pixelCount = sourceRgba.length / 4;
  if (mixedMask.length !== pixelCount * 3) {
    throw new RangeError("Matched additive synthesis requires a per-channel color mask");
  }

  // Preserve the user's existing effect-strength controls, but use the old
  // contrast blend only to measure the intended noise amplitude. The rendered
  // output is rebuilt as additive, zero-mean noise instead of retaining the
  // old signal-dependent brightness and contrast drift.
  const legacyComposite = compositeContrastCopy(sourceRgba, mixedMask, options);
  const desiredRms = residualLuminanceRms(legacyComposite, sourceRgba);
  const noise = decorrelateColorNoise(mixedMask, sourceRgba);
  scaleNoiseToLuminanceRms(noise, desiredRms);
  return matchChannelMoments(sourceRgba, noise);
}

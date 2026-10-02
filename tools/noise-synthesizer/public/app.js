import {
  adjustUnitValue,
  buildColorNoiseMask,
  calculateMegapixelSize,
  compositeMatchedAdditiveNoise,
  createPerlinNoise,
  createSobelMask,
  gaussianBlurMask,
  mixColorNoiseMasks,
} from "./synth.js";

const ids = [
  "resizeToMegapixel",
  "copyContrast", "copyBrightness", "effectStrength",
  "largePeriod", "largeContrast", "largeBrightness", "largeColorVariation",
  "smallPeriod", "smallContrast", "smallBrightness", "smallColorVariation",
  "edgeBlur", "edgeContrast", "edgeBrightness", "seed",
];
const controls = Object.fromEntries(ids.map((id) => [id, document.getElementById(id)]));
const canvases = Object.fromEntries(
  ["output", "source", "large", "small", "edge", "mixed"].map((name) => [
    name,
    document.getElementById(`${name}Canvas`),
  ]),
);
const state = {
  originalSource: null,
  source: null,
  edge: null,
  blurredEdge: null,
  edgeBlurKey: "",
  largeNoise: null,
  smallNoise: null,
  noiseKey: "",
  renderQueued: false,
  filename: "demo-source",
};

function number(id) {
  return Number.parseFloat(controls[id].value);
}

function settingValue(id) {
  const input = controls[id];
  return input.type === "checkbox" ? input.checked : number(id);
}

function setCanvasSize(canvas, width, height) {
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }
}

function paintMask(canvas, mask, width, height, adjustment = null) {
  setCanvasSize(canvas, width, height);
  const pixelCount = width * height;
  const hasColor = mask.length === pixelCount * 3;
  const rgba = new Uint8ClampedArray(pixelCount * 4);
  for (let index = 0; index < pixelCount; index += 1) {
    for (let channel = 0; channel < 3; channel += 1) {
      const maskIndex = hasColor ? index * 3 + channel : index;
      const value = adjustment ? adjustment(mask[maskIndex]) : mask[maskIndex];
      rgba[index * 4 + channel] = Math.round(value * 255);
    }
    rgba[index * 4 + 3] = 255;
  }
  canvas.getContext("2d").putImageData(new ImageData(rgba, width, height), 0, 0);
}

function render() {
  state.renderQueued = false;
  if (!state.source) return;

  const status = document.getElementById("status");
  status.textContent = "Rendering";
  const { width, height, data } = state.source;
  const seed = Math.max(0, Math.round(number("seed"))) >>> 0;
  const largeColorEnabled = number("largeColorVariation") > 0;
  const smallColorEnabled = number("smallColorVariation") > 0;
  const noiseKey = [
    width,
    height,
    number("largePeriod"),
    number("smallPeriod"),
    seed,
    Number(largeColorEnabled),
    Number(smallColorEnabled),
  ].join(":");
  if (noiseKey !== state.noiseKey) {
    const largeBase = createPerlinNoise(width, height, number("largePeriod"), seed);
    const smallSeed = seed ^ 0x9e3779b9;
    const smallBase = createPerlinNoise(width, height, number("smallPeriod"), smallSeed);
    state.largeNoise = largeColorEnabled ? [
      largeBase,
      createPerlinNoise(width, height, number("largePeriod"), seed ^ 0x85ebca6b),
      createPerlinNoise(width, height, number("largePeriod"), seed ^ 0xc2b2ae35),
    ] : [largeBase, largeBase, largeBase];
    state.smallNoise = smallColorEnabled ? [
      smallBase,
      createPerlinNoise(width, height, number("smallPeriod"), smallSeed ^ 0x27d4eb2f),
      createPerlinNoise(width, height, number("smallPeriod"), smallSeed ^ 0x165667b1),
    ] : [smallBase, smallBase, smallBase];
    state.noiseKey = noiseKey;
  }

  const edgeBlurKey = `${width}:${height}:${number("edgeBlur")}`;
  if (edgeBlurKey !== state.edgeBlurKey) {
    state.blurredEdge = gaussianBlurMask(
      state.edge,
      width,
      height,
      number("edgeBlur"),
    );
    state.edgeBlurKey = edgeBlurKey;
  }

  const maskControls = {
    largeContrast: number("largeContrast"),
    largeBrightness: number("largeBrightness"),
    smallContrast: number("smallContrast"),
    smallBrightness: number("smallBrightness"),
    edgeContrast: number("edgeContrast"),
    edgeBrightness: number("edgeBrightness"),
  };
  const largeColorMask = buildColorNoiseMask(
    state.largeNoise,
    number("largeColorVariation"),
    maskControls.largeContrast,
    maskControls.largeBrightness,
  );
  const smallColorMask = buildColorNoiseMask(
    state.smallNoise,
    number("smallColorVariation"),
    maskControls.smallContrast,
    maskControls.smallBrightness,
  );
  const mixedMask = mixColorNoiseMasks(
    largeColorMask,
    smallColorMask,
    state.blurredEdge,
    maskControls.edgeContrast,
    maskControls.edgeBrightness,
  );
  const output = compositeMatchedAdditiveNoise(data, mixedMask, {
    copyContrast: number("copyContrast"),
    copyBrightness: number("copyBrightness"),
    effectStrength: number("effectStrength"),
  });

  setCanvasSize(canvases.output, width, height);
  canvases.output.getContext("2d").putImageData(new ImageData(output, width, height), 0, 0);
  paintMask(canvases.large, largeColorMask, width, height);
  paintMask(canvases.small, smallColorMask, width, height);
  paintMask(canvases.edge, state.blurredEdge, width, height, (value) => (
    adjustUnitValue(value, maskControls.edgeContrast, maskControls.edgeBrightness)
  ));
  paintMask(canvases.mixed, mixedMask, width, height);
  status.textContent = "Ready";
}

function scheduleRender(invalidateNoise = false) {
  if (invalidateNoise) state.noiseKey = "";
  if (!state.renderQueued) {
    state.renderQueued = true;
    requestAnimationFrame(render);
  }
}

function updateReadouts() {
  for (const input of document.querySelectorAll("input[type='range']")) {
    const output = document.querySelector(`output[for='${input.id}']`);
    const value = Number.parseFloat(input.value);
    if (input.id.includes("Brightness")) output.value = `${value >= 0 ? "+" : ""}${value.toFixed(2)}`;
    else if (input.id.includes("Period")) output.value = `${Math.round(value)} px`;
    else if (input.id.includes("Blur")) output.value = `${value.toFixed(2)} px`;
    else if (input.id.includes("ColorVariation")) output.value = value.toFixed(2);
    else output.value = `${value.toFixed(2)}×`;
  }
}

function resizeInputToMegapixel(imageData) {
  const size = calculateMegapixelSize(imageData.width, imageData.height);
  if (size.width === imageData.width && size.height === imageData.height) {
    return imageData;
  }

  const sourceCanvas = document.createElement("canvas");
  sourceCanvas.width = imageData.width;
  sourceCanvas.height = imageData.height;
  sourceCanvas.getContext("2d").putImageData(imageData, 0, 0);

  const resizedCanvas = document.createElement("canvas");
  resizedCanvas.width = size.width;
  resizedCanvas.height = size.height;
  const context = resizedCanvas.getContext("2d", { willReadFrequently: true });
  context.imageSmoothingEnabled = true;
  context.imageSmoothingQuality = "high";
  context.drawImage(sourceCanvas, 0, 0, size.width, size.height);
  return context.getImageData(0, 0, size.width, size.height);
}

function applyInputTransform() {
  if (!state.originalSource) return;
  const imageData = controls.resizeToMegapixel.checked
    ? resizeInputToMegapixel(state.originalSource)
    : state.originalSource;

  state.source = imageData;
  state.edge = createSobelMask(imageData.data, imageData.width, imageData.height);
  state.edgeBlurKey = "";
  state.noiseKey = "";
  const original = state.originalSource;
  const resized = imageData.width !== original.width || imageData.height !== original.height;
  document.getElementById("dimensions").textContent = resized
    ? `${imageData.width} × ${imageData.height} · from ${original.width} × ${original.height}`
    : `${imageData.width} × ${imageData.height}`;
  setCanvasSize(canvases.source, imageData.width, imageData.height);
  canvases.source.getContext("2d").putImageData(imageData, 0, 0);
  scheduleRender(true);
}

function exportSettings() {
  const preset = {
    format: "ai-toolkit-noise-synthesizer",
    version: 1,
    values: Object.fromEntries(ids.map((id) => [id, settingValue(id)])),
  };
  const blob = new Blob([`${JSON.stringify(preset, null, 2)}\n`], {
    type: "application/json",
  });
  const anchor = document.createElement("a");
  anchor.href = URL.createObjectURL(blob);
  anchor.download = "noise-synthesizer-settings.json";
  anchor.click();
  setTimeout(() => URL.revokeObjectURL(anchor.href), 1000);
}

async function importSettings(file) {
  if (!file) return;
  const status = document.getElementById("status");
  try {
    const preset = JSON.parse(await file.text());
    const values = preset?.values ?? preset;
    for (const id of ids) {
      if (!(id in values)) continue;
      const input = controls[id];
      if (input.type === "checkbox") {
        const value = values[id];
        if (value === true || value === 1 || value === "true") input.checked = true;
        else if (value === false || value === 0 || value === "false") input.checked = false;
        else throw new TypeError(`${id} is not boolean`);
        continue;
      }
      const value = Number(values[id]);
      if (!Number.isFinite(value)) throw new TypeError(`${id} is not numeric`);
      const minimum = input.min === "" ? -Infinity : Number(input.min);
      const maximum = input.max === "" ? Infinity : Number(input.max);
      input.value = String(Math.max(minimum, Math.min(maximum, value)));
    }
    state.noiseKey = "";
    state.edgeBlurKey = "";
    updateReadouts();
    applyInputTransform();
  } catch (error) {
    console.error("Could not import noise-synthesizer settings", error);
    status.textContent = "Invalid settings JSON";
  }
}

function setSource(imageData, filename) {
  state.originalSource = imageData;
  state.filename = filename.replace(/\.[^.]+$/, "") || "synthesized-noise";
  document.getElementById("imageName").textContent = filename;
  applyInputTransform();
}

async function loadFile(file) {
  if (!file?.type.startsWith("image/")) return;
  const bitmap = await createImageBitmap(file);
  const scratch = document.createElement("canvas");
  scratch.width = bitmap.width;
  scratch.height = bitmap.height;
  const context = scratch.getContext("2d", { willReadFrequently: true });
  context.drawImage(bitmap, 0, 0);
  const imageData = context.getImageData(0, 0, bitmap.width, bitmap.height);
  bitmap.close();
  setSource(imageData, file.name);
}

function createDemo() {
  const width = 960;
  const height = 640;
  const scratch = document.createElement("canvas");
  scratch.width = width;
  scratch.height = height;
  const context = scratch.getContext("2d");
  const gradient = context.createLinearGradient(0, 0, width, height);
  gradient.addColorStop(0, "#f5d3a4");
  gradient.addColorStop(0.48, "#d96b54");
  gradient.addColorStop(1, "#26384d");
  context.fillStyle = gradient;
  context.fillRect(0, 0, width, height);
  context.fillStyle = "#102333";
  context.beginPath();
  context.arc(510, 318, 172, 0, Math.PI * 2);
  context.fill();
  context.strokeStyle = "#f7e5c4";
  context.lineWidth = 18;
  context.beginPath();
  context.moveTo(120, 520);
  context.bezierCurveTo(260, 90, 680, 560, 870, 120);
  context.stroke();
  context.fillStyle = "#f2ae52";
  context.fillRect(424, 232, 172, 172);
  context.fillStyle = "#f7e5c4";
  context.font = "700 78px system-ui";
  context.fillText("CLEAN", 55, 110);
  setSource(context.getImageData(0, 0, width, height), "Demo source");
}

for (const input of Object.values(controls)) {
  input.addEventListener("input", () => {
    updateReadouts();
    if (input.id === "resizeToMegapixel") applyInputTransform();
    else scheduleRender(input.matches("[data-noise-control]"));
  });
}

function downloadCanvas(canvas, filename) {
  canvas.toBlob((blob) => {
    if (!blob) return;
    const anchor = document.createElement("a");
    anchor.href = URL.createObjectURL(blob);
    anchor.download = filename;
    anchor.click();
    setTimeout(() => URL.revokeObjectURL(anchor.href), 1000);
  }, "image/png");
}

document.getElementById("fileInput").addEventListener("change", (event) => loadFile(event.target.files[0]));
document.getElementById("demoButton").addEventListener("click", createDemo);
document.getElementById("exportSettingsButton").addEventListener("click", exportSettings);
document.getElementById("importSettingsButton").addEventListener("click", () => {
  document.getElementById("settingsInput").click();
});
document.getElementById("settingsInput").addEventListener("change", async (event) => {
  await importSettings(event.target.files[0]);
  event.target.value = "";
});
document.getElementById("randomizeButton").addEventListener("click", () => {
  controls.seed.value = String(crypto.getRandomValues(new Uint32Array(1))[0]);
  scheduleRender(true);
});
document.getElementById("downloadButton").addEventListener("click", () => {
  downloadCanvas(canvases.output, `${state.filename}-noised.png`);
});
document.getElementById("downloadInputButton").addEventListener("click", () => {
  const suffix = controls.resizeToMegapixel.checked ? "1mp-input" : "input";
  downloadCanvas(canvases.source, `${state.filename}-${suffix}.png`);
});

const dropZone = document.getElementById("dropZone");
for (const eventName of ["dragenter", "dragover"]) {
  dropZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropZone.classList.add("dragging");
  });
}
for (const eventName of ["dragleave", "drop"]) {
  dropZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropZone.classList.remove("dragging");
  });
}
dropZone.addEventListener("drop", (event) => loadFile(event.dataTransfer.files[0]));

updateReadouts();
createDemo();

const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const Module = require('node:module');
const net = require('node:net');
const os = require('node:os');
const test = require('node:test');
const ts = require('typescript');
const React = require('react');
const { renderToStaticMarkup } = require('react-dom/server');
const { NextRequest } = require('next/server');
const sharp = require(require.resolve('sharp', { paths: [path.dirname(require.resolve('next/package.json'))] }));

const projectRoot = path.resolve(__dirname, '../..');
const tempRoot = path.join(projectRoot, 'tmp');
fs.mkdirSync(tempRoot, { recursive: true });
const root = fs.mkdtempSync(path.join(tempRoot, 'thumbnail-fit-test-'));
const variantPath = path.resolve(__dirname, '../cron/thumbnailVariant.ts');

function compileModule(filename, source, customRequire) {
  const compiled = ts.transpileModule(source, {
    fileName: filename,
    compilerOptions: {
      module: ts.ModuleKind.CommonJS,
      esModuleInterop: true,
      target: ts.ScriptTarget.ES2020,
      jsx: ts.JsxEmit.ReactJSX,
    },
  }).outputText;
  const instance = new Module(filename, module);
  instance.filename = filename;
  instance.paths = Module._nodeModulePaths(path.dirname(filename));
  const originalRequire = instance.require.bind(instance);
  instance.require = id => customRequire(id, originalRequire);
  instance._compile(compiled, filename);
  return instance.exports;
}

const thumbnailVariant = compileModule(variantPath, fs.readFileSync(variantPath, 'utf8'), (_id, originalRequire) => originalRequire(_id));

function loadFileServer() {
  const filename = path.resolve(__dirname, '../cron/fileServer.ts');
  const source = fs.readFileSync(filename, 'utf8');
  const bootstrap = source.lastIndexOf('\nif (cluster.isPrimary) {');
  assert.notEqual(bootstrap, -1, 'file server bootstrap marker is present');
  const isolated = `${source.slice(0, bootstrap)}\nexport { generateThumb, startServer };\n`.replace(
    'server.listen(publicPort);',
    'globalThis.__thumbnailTestServer = server.listen(publicPort);',
  );
  return compileModule(filename, isolated, (id, originalRequire) => {
    if (id === './prisma') return {
      __esModule: true,
      default: {
        settings: {
          findMany: async () => [
            { key: 'DATASETS_FOLDER', value: root },
            { key: 'TRAINING_FOLDER', value: root },
            { key: 'DATA_ROOT', value: root },
          ],
        },
      },
    };
    if (id === './paths') return {
      defaultDatasetsFolder: root,
      defaultTrainFolder: root,
      defaultDataRoot: root,
      TOOLKIT_ROOT: root,
    };
    if (id === './thumbnailVariant') return thumbnailVariant;
    return originalRequire(id);
  });
}

function loadImageRoute() {
  const filename = path.resolve(__dirname, '../src/app/api/img/[...imagePath]/route.ts');
  const source = fs.readFileSync(filename, 'utf8');
  return compileModule(filename, source, (id, originalRequire) => {
    if (id === '@/server/settings') return {
      getDatasetsRoot: async () => root,
      getTrainingFolder: async () => root,
      getDataRoot: async () => root,
    };
    if (id === '@/server/catchAllPath') return {
      catchAllToFilePath: value => (Array.isArray(value) ? path.join(...value) : value),
    };
    if (id === '../../../../../cron/thumbnailVariant') return thumbnailVariant;
    return originalRequire(id);
  });
}

function loadDatasetGrid() {
  const filename = path.resolve(__dirname, '../src/components/DatasetGrid.tsx');
  const source = fs.readFileSync(filename, 'utf8');
  return compileModule(filename, source, (id, originalRequire) => {
    if (id === 'next/link') return { __esModule: true, default: ({ children, ...props }) => React.createElement('a', props, children) };
    if (id === 'react-icons/fa') return { FaRegTrashAlt: () => React.createElement('span', null, 'delete') };
    if (id === 'react-icons/md') return { MdImageNotSupported: () => React.createElement('span', null, 'missing') };
    if (id === '@/utils/basic') return { encodeFilePathForUrl: value => encodeURIComponent(value) };
    return originalRequire(id);
  }).default;
}

function pixel(data, info, x, y) {
  const offset = (y * info.width + x) * info.channels;
  return [...data.subarray(offset, offset + info.channels)];
}

test('DatasetGrid requests the fit thumbnail while containing it in the square card', () => {
  const DatasetGrid = loadDatasetGrid();
  const markup = renderToStaticMarkup(
    React.createElement(DatasetGrid, {
      datasets: ['fixture'],
      firstImages: { fixture: '/datasets/fixture/wide.png' },
      size: 'medium',
      onDelete: () => {},
    }),
  );
  assert.match(markup, /\/api\/img\/[^\"]+\?thumb=fit/);
  assert.match(markup, /object-contain/);
});

async function freePort() {
  const server = net.createServer();
  await new Promise((resolve, reject) => server.once('error', reject).listen(0, '127.0.0.1', resolve));
  const port = server.address().port;
  await new Promise((resolve, reject) => server.close(error => (error ? reject(error) : resolve())));
  return port;
}

test('fit thumbnails preserve whole non-square images and alpha in a distinct cache from square crops', async () => {
  const fileServer = loadFileServer();
  const route = loadImageRoute();
  const port = await freePort();
  fileServer.startServer(port, port + 1);
  const server = globalThis.__thumbnailTestServer;
  await new Promise(resolve => server.once('listening', resolve));
  const width = 600;
  const height = 300;
  const raw = Buffer.alloc(width * height * 4);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const offset = (y * width + x) * 4;
      if (x < 90) raw.set([255, 0, 0, 255], offset);
      else if (x >= 510) raw.set([0, 0, 255, 255], offset);
      else raw.set([0, 255, 0, 255], offset);
      if (x >= 280 && x < 320 && y >= 120 && y < 180) raw[offset + 3] = 0;
    }
  }
  const sourcePath = path.join(root, 'wide-alpha.png');
  await sharp(raw, { raw: { width, height, channels: 4 } }).png().toFile(sourcePath);
  const thumbDirectory = path.join(root, '.thumbs');
  const squareBase = path.join(thumbDirectory, thumbnailVariant.getThumbnailCacheBasename(path.basename(sourcePath), 'square'));
  const fitBase = path.join(thumbDirectory, thumbnailVariant.getThumbnailCacheBasename(path.basename(sourcePath), 'fit'));

  const fetchFastThumb = async thumb => {
    const response = await fetch(
      `http://127.0.0.1:${port}/api/img/${encodeURIComponent(sourcePath)}?thumb=${thumb}`,
    );
    assert.equal(response.status, 200);
    assert.equal(response.headers.get('content-type'), 'image/png');
    return Buffer.from(await response.arrayBuffer());
  };
  const fitFastBytes = await fetchFastThumb('fit');
  const squareFastBytes = await fetchFastThumb('1');
  const squarePath = `${squareBase}.png`;
  const fitPath = `${fitBase}.png`;
  assert.equal(squarePath, `${squareBase}.png`);
  assert.equal(fitPath, `${fitBase}.png`);
  assert.notEqual(squarePath, fitPath);
  assert.equal(path.basename(squareBase), 'wide-alpha.png');
  assert.equal(path.basename(fitBase), 'wide-alpha.png.fit');
  assert.deepEqual(fitFastBytes, await fs.promises.readFile(fitPath));
  assert.deepEqual(squareFastBytes, await fs.promises.readFile(squarePath));

  const square = await sharp(squarePath).raw().toBuffer({ resolveWithObject: true });
  const fit = await sharp(fitPath).raw().toBuffer({ resolveWithObject: true });
  assert.deepEqual({ width: square.info.width, height: square.info.height, channels: square.info.channels }, { width: 300, height: 300, channels: 4 });
  assert.deepEqual({ width: fit.info.width, height: fit.info.height, channels: fit.info.channels }, { width: 300, height: 150, channels: 4 });
  assert.deepEqual(pixel(fit.data, fit.info, 10, 75), [255, 0, 0, 255], 'fit keeps the left edge');
  assert.deepEqual(pixel(fit.data, fit.info, 290, 75), [0, 0, 255, 255], 'fit keeps the right edge');
  assert.equal(pixel(fit.data, fit.info, 150, 75)[3], 0, 'fit keeps transparent pixels');
  assert.deepEqual(pixel(square.data, square.info, 0, 150), [0, 255, 0, 255], 'default square crop excludes the left edge');
  assert.deepEqual(pixel(square.data, square.info, 299, 150), [0, 255, 0, 255], 'default square crop excludes the right edge');
  assert.equal(pixel(square.data, square.info, 150, 150)[3], 0, 'default square crop still preserves alpha');

  const getThumbnail = async thumb => {
    const request = new NextRequest(`http://localhost/api/img/wide-alpha.png?thumb=${thumb}`);
    return route.GET(request, { params: Promise.resolve({ imagePath: sourcePath }) });
  };
  const fitResponse = await getThumbnail('fit');
  assert.equal(fitResponse.status, 200);
  assert.equal(fitResponse.headers.get('content-type'), 'image/png');
  const fitBytes = Buffer.from(await fitResponse.arrayBuffer());
  assert.deepEqual(fitBytes, await fs.promises.readFile(fitPath), 'Next fallback serves the fit cache');
  const legacyResponse = await getThumbnail('1');
  assert.equal(legacyResponse.status, 200);
  assert.deepEqual(Buffer.from(await legacyResponse.arrayBuffer()), await fs.promises.readFile(squarePath), 'legacy thumb=1 still serves square crop');
});

test.after(async () => {
  const server = globalThis.__thumbnailTestServer;
  if (server?.listening) await new Promise((resolve, reject) => server.close(error => (error ? reject(error) : resolve())));
  fs.rmSync(root, { recursive: true, force: true });
});

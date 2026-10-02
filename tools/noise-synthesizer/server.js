import { X509Certificate } from "node:crypto";
import { createReadStream, readFileSync } from "node:fs";
import { stat } from "node:fs/promises";
import { createServer as createHttpServer } from "node:http";
import { createServer } from "node:https";
import path from "node:path";
import { fileURLToPath } from "node:url";

const publicRoot = path.resolve(fileURLToPath(new URL("./public/", import.meta.url)));
const host = process.env.NOISE_SYNTH_HOST ?? "0.0.0.0";
const port = Number.parseInt(process.env.NOISE_SYNTH_PORT ?? "4178", 10);
const caPort = Number.parseInt(process.env.NOISE_SYNTH_CA_PORT ?? "4177", 10);
const certificatePath = process.env.NOISE_SYNTH_CERT ?? fileURLToPath(
  new URL("./certs/noise-synthesizer.crt", import.meta.url),
);
const keyPath = process.env.NOISE_SYNTH_KEY ?? fileURLToPath(
  new URL("./certs/noise-synthesizer.key", import.meta.url),
);
const caCertificatePath = process.env.NOISE_SYNTH_CA_CERT ?? fileURLToPath(
  new URL("./certs/shodan-local-ca.crt", import.meta.url),
);
const caCertificate = readFileSync(caCertificatePath);
const caFingerprint = new X509Certificate(caCertificate).fingerprint256;

const mimeTypes = {
  ".css": "text/css; charset=utf-8",
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".png": "image/png",
  ".svg": "image/svg+xml",
};

function responseHeaders(headers = {}) {
  return {
    // Explicitly clear any previously cached HSTS policy for this origin.
    "Strict-Transport-Security": "max-age=0",
    ...headers,
  };
}

const server = createServer({
  cert: readFileSync(certificatePath),
  key: readFileSync(keyPath),
}, async (request, response) => {
  try {
    const requestUrl = new URL(request.url ?? "/", "https://localhost");
    const pathname = decodeURIComponent(requestUrl.pathname);
    const relativePath = pathname === "/" ? "index.html" : pathname.slice(1);
    const filePath = path.resolve(publicRoot, relativePath);

    if (!filePath.startsWith(publicRoot + path.sep)) {
      response.writeHead(403, responseHeaders()).end("Forbidden");
      return;
    }

    const fileStats = await stat(filePath);
    if (!fileStats.isFile()) {
      throw new Error("Not a file");
    }

    response.writeHead(200, responseHeaders({
      "Cache-Control": "no-store",
      "Content-Length": fileStats.size,
      "Content-Type": mimeTypes[path.extname(filePath)] ?? "application/octet-stream",
    }));
    createReadStream(filePath).pipe(response);
  } catch {
    response.writeHead(404, responseHeaders({
      "Content-Type": "text/plain; charset=utf-8",
    }));
    response.end("Not found");
  }
});

server.listen(port, host, () => {
  console.log(`Noise synthesizer listening on https://${host}:${port}`);
});

if (caPort > 0) {
  const caServer = createHttpServer((request, response) => {
    const requestUrl = new URL(request.url ?? "/", "http://localhost");
    if (requestUrl.pathname === "/shodan-local-ca.crt") {
      response.writeHead(200, {
        "Cache-Control": "no-store",
        "Content-Disposition": "attachment; filename=shodan-local-ca.crt",
        "Content-Length": caCertificate.length,
        "Content-Type": "application/x-x509-ca-cert",
        "X-Content-Type-Options": "nosniff",
      });
      response.end(caCertificate);
      return;
    }

    const message = [
      "Shodan Local Development CA bootstrap",
      "",
      `Download: http://${request.headers.host}/shodan-local-ca.crt`,
      `SHA-256 fingerprint: ${caFingerprint}`,
      "",
      "Only the public CA certificate is served here. The private key is not exposed.",
    ].join("\n");
    response.writeHead(200, {
      "Cache-Control": "no-store",
      "Content-Type": "text/plain; charset=utf-8",
      "X-Content-Type-Options": "nosniff",
    });
    response.end(message);
  });
  caServer.listen(caPort, host, () => {
    console.log(`CA certificate: http://${host}:${caPort}/shodan-local-ca.crt`);
    console.log(`CA SHA-256 fingerprint: ${caFingerprint}`);
  });
}

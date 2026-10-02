import {
  chmodSync,
  existsSync,
  mkdirSync,
  rmSync,
  writeFileSync,
} from "node:fs";
import { hostname, networkInterfaces } from "node:os";
import path from "node:path";
import { spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";

const appRoot = path.dirname(fileURLToPath(import.meta.url));
const certificateDirectory = path.join(appRoot, "certs");
const caCertificatePath = path.join(certificateDirectory, "shodan-local-ca.crt");
const caKeyPath = path.join(certificateDirectory, "shodan-local-ca.key");
const caSerialPath = path.join(certificateDirectory, "shodan-local-ca.srl");
const certificatePath = path.join(certificateDirectory, "noise-synthesizer.crt");
const keyPath = path.join(certificateDirectory, "noise-synthesizer.key");
const requestPath = path.join(certificateDirectory, "noise-synthesizer.csr");
const extensionsPath = path.join(certificateDirectory, "noise-synthesizer.ext");
const forceLeaf = process.argv.includes("--force");
const rotateCa = process.argv.includes("--rotate-ca");

function runOpenSsl(args, options = {}) {
  const result = spawnSync("openssl", args, {
    stdio: options.quiet ? "ignore" : "inherit",
  });
  if (result.error) throw result.error;
  if (result.status !== 0 && !options.allowFailure) {
    process.exit(result.status ?? 1);
  }
  return result.status === 0;
}

mkdirSync(certificateDirectory, { recursive: true });

if (rotateCa) {
  for (const filePath of [caCertificatePath, caKeyPath, caSerialPath]) {
    rmSync(filePath, { force: true });
  }
}

let createdCa = false;
if (!existsSync(caCertificatePath) || !existsSync(caKeyPath)) {
  // Never keep half of a CA pair: it would be impossible to issue renewals.
  rmSync(caCertificatePath, { force: true });
  rmSync(caKeyPath, { force: true });
  rmSync(caSerialPath, { force: true });
  runOpenSsl([
    "req",
    "-x509",
    "-newkey", "rsa:3072",
    "-sha256",
    "-nodes",
    "-days", "3650",
    "-keyout", caKeyPath,
    "-out", caCertificatePath,
    "-subj", "/CN=Shodan Local Development CA",
    "-addext", "basicConstraints=critical,CA:TRUE,pathlen:0",
    "-addext", "keyUsage=critical,keyCertSign,cRLSign",
    "-addext", "subjectKeyIdentifier=hash",
  ]);
  chmodSync(caKeyPath, 0o600);
  createdCa = true;
  console.log(`Created local CA: ${caCertificatePath}`);
}

const dnsNames = new Set(["localhost", hostname()]);
const ipAddresses = new Set(["127.0.0.1"]);
for (const addresses of Object.values(networkInterfaces())) {
  for (const address of addresses ?? []) {
    if (!address.internal) ipAddresses.add(address.address.split("%")[0]);
  }
}
const subjectAltNames = [
  ...[...dnsNames].map((name) => `DNS:${name}`),
  ...[...ipAddresses].map((address) => `IP:${address}`),
].join(",");

const leafVerifies = (
  existsSync(certificatePath)
  && existsSync(keyPath)
  && runOpenSsl(
    ["verify", "-CAfile", caCertificatePath, certificatePath],
    { allowFailure: true, quiet: true },
  )
);

if (createdCa || rotateCa || forceLeaf || !leafVerifies) {
  for (const filePath of [certificatePath, keyPath, requestPath, extensionsPath]) {
    rmSync(filePath, { force: true });
  }

  writeFileSync(extensionsPath, [
    `subjectAltName=${subjectAltNames}`,
    "basicConstraints=critical,CA:FALSE",
    "keyUsage=critical,digitalSignature,keyEncipherment",
    "extendedKeyUsage=serverAuth",
    "subjectKeyIdentifier=hash",
    "authorityKeyIdentifier=keyid,issuer",
    "",
  ].join("\n"), { mode: 0o600 });

  try {
    runOpenSsl([
      "req",
      "-new",
      "-newkey", "rsa:2048",
      "-sha256",
      "-nodes",
      "-keyout", keyPath,
      "-out", requestPath,
      "-subj", `/CN=${hostname()}`,
    ]);
    runOpenSsl([
      "x509",
      "-req",
      "-in", requestPath,
      "-CA", caCertificatePath,
      "-CAkey", caKeyPath,
      "-CAcreateserial",
      "-out", certificatePath,
      "-days", "397",
      "-sha256",
      "-extfile", extensionsPath,
    ]);
  } finally {
    rmSync(requestPath, { force: true });
    rmSync(extensionsPath, { force: true });
  }

  chmodSync(keyPath, 0o600);
  if (!runOpenSsl(["verify", "-CAfile", caCertificatePath, certificatePath])) {
    process.exit(1);
  }
  console.log(`Issued HTTPS certificate for ${subjectAltNames}`);
}

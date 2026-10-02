# Noise Pair Synthesizer

A dependency-free local web tool for producing synthetic noisy counterparts to
clean training images.

```bash
cd tools/noise-synthesizer
npm start
```

The HTTPS server binds to `0.0.0.0` by default. Open port `4178` using this
machine's local-network address, load or drop an image, adjust the masks, and
use **Download PNG**. On the current machine that is:

<https://192.168.50.8:4178>

The first start creates a private local certificate authority and uses it to
sign a server certificate for `shodan`, `localhost`, and the machine's current
local addresses. The CA persists across normal server-certificate renewals.

Download the public CA certificate over the LAN at:

<http://192.168.50.8:4177/shodan-local-ca.crt>

Verify its SHA-256 fingerprint against the value printed by `npm start`, then
import it into Firefox under **Settings → Privacy & Security → Certificates →
View Certificates → Authorities → Import** and allow it to identify websites.
Only the public certificate is downloadable; the CA private key remains in the
ignored `certs/` directory with mode `0600`.

The HTTPS server does not enforce HSTS and sends `max-age=0` to clear a
previously cached policy. To reissue the server certificate after an address
change without changing the trusted CA:

```bash
npm run cert
```

Rotating the CA invalidates the trust installed on every client and therefore
is a separate explicit operation:

```bash
npm run cert:rotate-ca
```

To choose another bind address or port:

```bash
NOISE_SYNTH_HOST=0.0.0.0 NOISE_SYNTH_PORT=4180 NOISE_SYNTH_CA_PORT=4179 npm start
```

Set `NOISE_SYNTH_CA_PORT=0` to disable the public-CA download endpoint.

You can supply an existing trusted certificate instead:

```bash
NOISE_SYNTH_CERT=/path/to/cert.pem NOISE_SYNTH_KEY=/path/to/key.pem npm start
```

The pipeline is:

1. Optionally resize the source to approximately `1024²` pixels while
   preserving its aspect ratio. This processed input can be downloaded as PNG.
2. Make a linearly contrast/brightness-adjusted copy of the processed source.
3. Generate independent large- and small-period Perlin masks.
   Each mask can interpolate from grayscale to independent RGB noise using its
   own color-variation control.
4. Generate a Sobel edge mask and Gaussian-blur it (4 px by default).
5. Apply linear brightness/contrast to the blurred edge mask, then select large
   noise in dark regions and small noise in bright ones.
6. Use the legacy contrast blend to measure the requested effect strength, then
   rebuild the result as zero-mean additive RGB noise. The noise is decorrelated
   from the clean image, and a clipping-aware solve matches every clean RGB
   channel's mean and standard deviation in the final output.

All control values can be exported to or restored from a versioned JSON preset.

Every contrast/brightness operation is linear followed by a hard clamp to
`[0, 1]`; it does not use a sigmoid or another range-compressing curve. Matching
RGB channel means and standard deviations prevents the generated side of a
denoising pair from acquiring a global color, brightness, or contrast target.

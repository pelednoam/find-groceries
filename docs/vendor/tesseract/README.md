# Tesseract, vendored

The optical character reader that turns a photograph of a receipt into text.
It is here, rather than on a CDN, for the same reason the fonts and Leaflet
are: the page must work under `default-src 'none'`, and a receipt is private
enough that no third party should learn one is being read.

Nothing in this directory loads until someone actually scans a photo — about
7 MB of it, fetched once and cached. A reader who never scans never pays for
it.

| File | From | Version |
| --- | --- | --- |
| `tesseract.min.js`, `worker.min.js` | [`tesseract.js`](https://github.com/naptha/tesseract.js) | 5.1.1 |
| `core/tesseract-core-simd-lstm.wasm.js` | [`tesseract.js-core`](https://github.com/naptha/tesseract.js-core) | 5.1.1 |
| `core/tesseract-core-lstm.wasm.js` | same, for browsers without WebAssembly SIMD | 5.1.1 |
| `lang/eng.traineddata.gz` | [`@tesseract.js-data/eng`](https://github.com/naptha/tessdata) `4.0.0_best_int` | 1.0.0 |

`tesseract.js` and `tesseract.js-core` are Apache-2.0; the licence texts are
`LICENSE.txt` and `core/LICENSE`. The English model is the integerised
`tessdata_best` build — a third the size of the float one at close to the
same accuracy, which matters when the reader is on a phone.

To update: `npm pack tesseract.js@5 tesseract.js-core@5 @tesseract.js-data/eng`,
then copy the files above out of the tarballs. `worker.min.js` and the core
build must come from matching versions.

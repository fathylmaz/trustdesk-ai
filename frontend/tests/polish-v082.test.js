import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { parseComparison } from '../src/model.js';
import { friendlyApiError, apiResponseError } from '../src/errors.js';

const app = readFileSync(new URL('../src/App.jsx', import.meta.url), 'utf8');
const css = readFileSync(new URL('../src/styles.css', import.meta.url), 'utf8');
const html = readFileSync(new URL('../index.html', import.meta.url), 'utf8');

test('visible workspace chrome and model headings use consistent Turkish language', () => {
  for (const phrase of ['ÇALIŞMA ALANI', 'Çalışma alanı', 'Model laboratuvarı', 'YEREL GELİŞTİRME', 'DENEYSEL MODEL LABORATUVARI', 'BAŞLANGIÇ MODELİ', 'TRANSFORMER MODELİ']) {
    assert.ok(app.includes(phrase), `Missing localized text: ${phrase}`);
  }
  assert.match(html, /<html lang="tr">/);
  assert.match(html, /Model Laboratuvarı/);
});

test('labels and timings are described without unjustified calibration claims', () => {
  assert.match(app, /Model işlem süresi/);
  assert.match(app, /Tek bir isteğin model üzerinde işlenme süresi/);
  assert.match(app, /Skorlar doğrulanmış güven olasılığı değildir/);
  assert.match(app, /Otomatik bankacılık yönlendirmesi ve gerçek finansal işlemler kapalıdır/);
});

test('responsive and contrast improvements retain known breakpoints', () => {
  assert.match(css, /v0\.8\.2 — okunabilirlik/);
  assert.match(css, /\.result-metrics span \{ color:#516482; font-size:12px/);
  assert.match(css, /@media\s*\(max-width:680px\)/);
  assert.match(css, /@media \(max-width:900px\)/);
  assert.match(css, /\.sample-pill:focus-visible/);
});

test('textarea communicates input format and live character count', () => {
  assert.match(app, /id="input-guidance"/);
  assert.match(app, /aria-describedby="input-guidance message-count"/);
  assert.match(app, /id="message-count"/);
  assert.match(app, /aria-label="Karşılaştırma sonucunu JSON olarak kopyala"/);
});

test('successful model comparison restores online UI state and clears stale copy message', () => {
  assert.match(app, /setConnection\(old => \(\{ status: 'online'/);
  assert.match(app, /setCopyStatus\(''\);\s*const validationError/);
});

test('friendly errors distinguish unavailability, timeout and server errors', () => {
  assert.match(friendlyApiError(new TypeError('Failed to fetch')), /servisine ulaşılamıyor/);
  assert.match(friendlyApiError({name:'AbortError'}), /uzun sürdü/);
  assert.match(friendlyApiError(apiResponseError(500)), /Sunucuda geçici/);
  assert.doesNotMatch(friendlyApiError(new TypeError('Failed to fetch')), /CORS|8002|HTTP/);
});

test('API still rejects any automatic routing and missing model entries', () => {
  const sample = {model:'tfidf',label:'cancel_transfer',score:0.2,inference_ms:2.3,
    top_predictions:[{label:'cancel_transfer',score:0.2}],automatic_routing_enabled:false};
  const response = {text:'test',predictions:[sample,{...sample,model:'distilbert'}]};
  assert.equal(parseComparison(response).agreement,true);
  assert.throws(() => parseComparison({...response, predictions:[{...sample, automatic_routing_enabled:true},{...sample,model:'distilbert'}]}));
});

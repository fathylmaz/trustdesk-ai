import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { readableLabel } from '../src/model.js';
import { friendlyApiError, apiResponseError } from '../src/errors.js';

const app = readFileSync(new URL('../src/App.jsx', import.meta.url), 'utf8');
const css = readFileSync(new URL('../src/styles.css', import.meta.url), 'utf8');

test('Turkish labels preserve actual model intent and unknown codes degrade gracefully', () => {
  assert.equal(readableLabel('terminate_account'), 'Hesap kapatma');
  assert.equal(readableLabel('cancel_transfer'), 'Transfer iptali');
  assert.equal(readableLabel('lost_or_stolen_card'), 'Kayıp veya çalıntı kart');
  assert.equal(readableLabel('some_new_intent'), 'some new intent');
});

test('network failures have friendly, nontechnical user copy', () => {
  const msg = friendlyApiError(new TypeError('Failed to fetch'));
  assert.match(msg, /servisine ulaşılamıyor/);
  assert.doesNotMatch(msg, /CORS|8002|TypeError/);
});

test('timeout and HTTP errors produce contextual messages', () => {
  assert.match(friendlyApiError({ name:'AbortError'}), /uzun sürdü/);
  assert.match(friendlyApiError(apiResponseError(503)), /Sunucuda geçici/);
  assert.match(friendlyApiError(apiResponseError(422)), /Mesaj işlenemedi/);
});

test('model result presents translated label before original technical code', () => {
  assert.ok(app.indexOf('<div className="intent-raw">{readableLabel(prediction.label)}</div>') < app.indexOf('<div className="intent-label" title="Modelin teknik kategori kodu">{prediction.label}</div>'));
  assert.match(app, /title=\{`Teknik kod: \$\{p.label\}`\}>\{readableLabel\(p.label\)\}/);
});

test('retry control and live model status are provided', () => {
  assert.match(app, /Yeniden dene<\/button>/);
  assert.match(app, /onClick=\{\(\) => compare\(\)\}/);
  assert.match(app, /'2 model hazır'/);
  assert.match(app, /connection.status === 'online'/);
});

test('accessibility focus styles and responsive breakpoints remain', () => {
  assert.match(css, /\.status-button:focus-visible/);
  assert.match(css, /@media\s*\(max-width:680px\)/);
  assert.match(css, /@media \(max-width:900px\)/);
  assert.match(app, /ref=\{messageInputRef\}/);
});

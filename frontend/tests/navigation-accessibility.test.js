import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

const source = readFileSync(new URL('../src/App.jsx', import.meta.url), 'utf8');
const css = readFileSync(new URL('../src/styles.css', import.meta.url), 'utf8');

// Static regression guards; keyboard behavior is also verified manually in the browser.
test('sidebar menu item is a native, labelled button', () => {
  assert.match(source, /<button[\s\S]*?className="nav-item active"[\s\S]*?aria-label="Model karşılaştırma/);
  assert.doesNotMatch(source, /<div className="nav-item active">/);
});
test('sidebar navigation button focuses the actual textarea', () => {
  assert.match(source, /const messageInputRef = useRef\(null\)/);
  assert.match(source, /onClick=\{\(\) => messageInputRef\.current\?\.focus\(\)\}/);
  assert.match(source, /<textarea ref=\{messageInputRef\} id="customer-text"/);
});
test('nav button has a visible keyboard focus style', () => {
  assert.match(css, /\.sidebar \.nav-item:focus-visible\{outline:/);
});

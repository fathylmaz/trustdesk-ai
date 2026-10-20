import { intentLabel } from './labels-tr.js';
export const API_DEFAULT = 'http://127.0.0.1:8002';
export const MAX_TEXT_LENGTH = 2000;
export const MODEL_NAMES = {
  tfidf: 'TF-IDF + Logistic Regression',
  distilbert: 'DistilBERT Transformer',
};

export const SAMPLES = [
  { name: 'Hesap kapatma', tag: 'Benzer niyetler', text: 'I want to cancel my bank account.' },
  { name: 'Çalıntı kart', tag: 'Hassas talep', text: 'My card was stolen yesterday.' },
  { name: 'Reddedilen ödeme', tag: 'Kart işlemi', text: 'My card payment was declined at the supermarket.' },
  { name: 'Hava durumu', tag: 'Konu dışı', text: 'How humid is the air in the city today?' },
];

export function readableLabel(label) {
  return intentLabel(label);
}

export function scoreText(value) {
  return typeof value === 'number' && Number.isFinite(value) && value >= 0 && value <= 1
    ? value.toFixed(5) : '—';
}

export function latencyText(value) {
  return typeof value === 'number' && Number.isFinite(value) && value >= 0
    ? `${value.toFixed(2)} ms` : 'Ölçülemedi';
}

export function validateText(text) {
  const n = text.trim().length;
  if (n < 3) return 'Lütfen en az 3 karakterlik İngilizce bir mesaj yazın.';
  if (text.length > MAX_TEXT_LENGTH) return 'Mesaj 2.000 karakterden uzun olamaz.';
  return '';
}

export function parseComparison(json) {
  if (!json || typeof json !== 'object' || !Array.isArray(json.predictions)) {
    throw new Error('API beklenen karşılaştırma yanıtını göndermedi.');
  }
  const names = ['tfidf', 'distilbert'];
  const predictions = names.map((modelName) => {
    const p = json.predictions.find((item) => item?.model === modelName);
    if (!p || typeof p.label !== 'string' || !Number.isFinite(p.score) ||
      !Array.isArray(p.top_predictions) || !Number.isFinite(p.inference_ms) ||
      p.automatic_routing_enabled !== false) {
      throw new Error(`API yanıtındaki ${modelName} model verisi eksik veya geçersiz.`);
    }
    return p;
  });
  return {
    text: typeof json.text === 'string' ? json.text : '',
    predictions,
    agreement: predictions[0].label === predictions[1].label,
    warning: typeof json.warning === 'string' ? json.warning : '',
  };
}

import React, { useEffect, useRef, useState } from 'react';
import Icon from './Icons.jsx';
import { friendlyApiError, apiResponseError } from './errors.js';
import { API_DEFAULT, MAX_TEXT_LENGTH, MODEL_NAMES, SAMPLES, latencyText, parseComparison, readableLabel, scoreText, validateText } from './model.js';

const API_BASE = (import.meta.env.VITE_API_BASE_URL || API_DEFAULT).replace(/\/$/, '');


async function getHealth(signal) {
  const response = await fetch(`${API_BASE}/health`, { signal });
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  const data = await response.json();
  if (data.status !== 'ok' || !data.models_loaded) throw new Error('API modelleri henüz yüklememiş.');
  return data;
}

function StatusDot({ status }) {
  return <span className={`status-dot ${status}`} aria-hidden="true" />;
}

function HeroMetric({ icon, value, label }) {
  return <div className="hero-metric"><div className="metric-icon"><Icon name={icon} size={16}/></div><div><strong>{value}</strong><span>{label}</span></div></div>;
}

function PredictionCard({ prediction, index }) {
  const name = MODEL_NAMES[prediction.model] || prediction.model;
  return <article className={`model-card model-card-${index}`}>
    <div className="model-card-top">
      <div className={`model-symbol symbol-${index}`}><Icon name={index === 0 ? 'layers' : 'spark'} size={21} /></div>
      <div className="model-heading"><span className="small-heading">{index === 0 ? 'BAŞLANGIÇ MODELİ' : 'TRANSFORMER MODELİ'}</span><h3>{name}</h3></div>
      <span className="model-version">{index === 0 ? 'ML' : 'DL'}</span>
    </div>
    <div className="prediction-zone"><span className="field-heading">TAHMİN EDİLEN NİYET</span><div className="intent-raw">{readableLabel(prediction.label)}</div><div className="intent-label" title="Modelin teknik kategori kodu">{prediction.label}</div></div>
    <div className="result-metrics"><div><span>Ham model skoru</span><strong>{scoreText(prediction.score)}</strong></div><div><span title="Tek bir isteğin model üzerinde işlenme süresi; kapsamlı hız testi değildir.">Model işlem süresi</span><strong>{latencyText(prediction.inference_ms)}</strong></div></div>
    <div className="top-title"><span>İlk 3 tahmin</span><span>Ham skor</span></div>
    <ol className="top-predictions">
      {prediction.top_predictions.slice(0,3).map((p, i) => <li key={`${p.label}-${i}`}><span className="rank">{i+1}</span><span className="top-label" title={`Teknik kod: ${p.label}`}>{readableLabel(p.label)}</span><span className="top-score">{scoreText(p.score)}</span></li>)}
    </ol>
    <div className="model-note"><Icon name="shield" size={15}/><span>Otomatik işlem yok • Model skorları doğrulanmış güven olasılıkları değildir.</span></div>
  </article>;
}

export default function App() {
  const [text, setText] = useState('I want to cancel my bank account.');
  const [connection, setConnection] = useState({ status: 'checking', device: '' });
  const [result, setResult] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');
  const [lastCompared, setLastCompared] = useState('');
  const [copyStatus, setCopyStatus] = useState('');
  const requestRef = useRef(null);
  const messageInputRef = useRef(null);

  const refreshHealth = async () => {
    setConnection({ status: 'checking', device: '' });
    const ctl = new AbortController();
    const handle = setTimeout(() => ctl.abort(), 8000);
    try {
      const h = await getHealth(ctl.signal);
      setConnection({ status: 'online', device: String(h.device || 'Hazır') });
    } catch {
      setConnection({ status: 'offline', device: '' });
    } finally { clearTimeout(handle); }
  };

  useEffect(() => {
    refreshHealth();
    return () => { if (requestRef.current) requestRef.current.abort(); };
  }, []);

  const compare = async (event) => {
    event?.preventDefault();
    if (isLoading) return;
    setError('');
    setCopyStatus('');
    const validationError = validateText(text);
    if (validationError) { setError(validationError); return; }
    setResult(null);
    const ctl = new AbortController();
    requestRef.current = ctl;
    const handle = setTimeout(() => ctl.abort(), 60000);
    setIsLoading(true);
    try {
      const response = await fetch(`${API_BASE}/v1/compare`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: text.trim() }), signal: ctl.signal,
      });
      if (!response.ok) {
        throw apiResponseError(response.status);
      }
      const body = parseComparison(await response.json());
      setResult(body);
      setLastCompared(text.trim());
      // A successful real comparison also proves the local API is reachable.
      setConnection(old => ({ status: 'online', device: old.device || 'Hazır' }));
    } catch (e) {
      if (e instanceof TypeError) setConnection({ status: 'offline', device: '' });
      setError(friendlyApiError(e));
    }
    finally { clearTimeout(handle); requestRef.current = null; setIsLoading(false); }
  };

  const selectSample = (sample) => { setText(sample.text); setResult(null); setError(''); setCopyStatus(''); };
  const clear = () => { setText(''); setResult(null); setError(''); setCopyStatus(''); };
  const copyJson = async () => {
    if (!result || !navigator.clipboard?.writeText) return;
    try {
      await navigator.clipboard.writeText(JSON.stringify({text:result.text, predictions:result.predictions, warning:result.warning}, null, 2));
      setCopyStatus('JSON kopyalandı');
    } catch { setCopyStatus('Kopyalama başarısız'); }
  };

  return <div className="app-shell">
    <aside className="sidebar">
      <div className="brand"><span className="brand-icon"><Icon name="spark" size={22}/></span><div><strong>TrustDesk<span>AI</span></strong><small>YAPAY ZEKÂ ÇALIŞMA ALANI</small></div></div>
      <div className="sidebar-section-label">ÇALIŞMA ALANI</div>
      <button
        type="button"
        className="nav-item active"
        aria-label="Model karşılaştırma – müşteri mesajı alanına git"
        aria-controls="customer-text"
        title="Model karşılaştırma alanına git"
        onClick={() => messageInputRef.current?.focus()}
      ><Icon name="layers" size={19}/> Model karşılaştırma <span>01</span></button>
      <div className="sidebar-section-label sidebar-second">PROJE BİLGİLERİ</div>
      <div className="sidebar-info"><div><span className="mini-circle"/> {connection.status === 'online' ? '2 model hazır' : connection.status === 'checking' ? 'Modeller kontrol ediliyor' : 'Model bağlantısı kesildi'}</div><div><span className="mini-circle"/> 77 niyet sınıfı</div><div><span className="mini-circle"/> BANKING77 deney ortamı</div></div>
      <div className="sidebar-bottom"><div className="lock-box"><Icon name="shield" size={19}/><div><strong>Yalnızca yerel deney</strong><span>Gerçek banka verisi kullanmayın. Bu demo finansal işlem yapmaz.</span></div></div><div className="sidebar-footer">TrustDesk AI <span>v0.8.2 • Arayüz</span></div></div>
    </aside>

    <main className="main-panel">
      <div className="topbar"><div className="crumb">Çalışma alanı <span>/</span> Model laboratuvarı</div><div className="topbar-right"><span className="topbar-local"><Icon name="shield" size={14}/> YEREL GELİŞTİRME</span><button className="status-button" type="button" aria-live="polite" onClick={refreshHealth} title="API bağlantısını tekrar kontrol et" aria-label="API bağlantısını yeniden kontrol et"><StatusDot status={connection.status}/>{connection.status === 'online' ? 'API çevrimiçi' : connection.status === 'checking' ? 'Kontrol ediliyor' : 'API çevrimdışı'}<Icon name="refresh" size={14}/></button></div></div>
      <div className="content">
        <header className="page-header"><div className="eyebrow"><span className="tiny-line"/> DENEYSEL MODEL LABORATUVARI</div><h1>AI modellerini <em>yan yana</em> karşılaştırın.</h1><p>Aynı müşteri mesajını iki farklı sınıflandırma modeline gönderin. Tahminleri, ilk üç kategoriyi ve işlem sürelerini tek ekranda inceleyin.</p></header>
        <div className="hero-strip"><HeroMetric icon="layers" value="02" label="Karşılaştırılan model"/><HeroMetric icon="globe" value="77" label="Bankacılık niyeti"/><HeroMetric icon="bolt" value={connection.status === 'online' ? 'Hazır' : connection.status === 'checking' ? 'Kontrol' : 'Bağlantı yok'} label="Yerel API bağlantısı"/></div>
        <section className="compose-card" aria-labelledby="compose-title">
          <div className="section-top"><div><div className="section-overline">01 / MESAJ</div><h2 id="compose-title">Müşteri mesajı</h2><p>İngilizce bir örnek mesaj girin veya aşağıdaki senaryolardan birini seçin.</p><p className="input-guidance" id="input-guidance">Model yalnızca İngilizce bankacılık talepleri üzerinde eğitildi.</p></div><div className="english-tag">İNGİLİZCE MESAJLAR <span>•</span> 3–2000 KARAKTER</div></div>
          <form onSubmit={compare}><label className="sr-only" htmlFor="customer-text">Müşteri mesajı</label><textarea ref={messageInputRef} id="customer-text" aria-describedby="input-guidance message-count" value={text} maxLength={MAX_TEXT_LENGTH} onChange={e=>{setText(e.target.value);setError('')}} spellCheck="false" rows={4} placeholder="Örneğin: I want to cancel my bank account."/><div className="compose-controls"><span className="length-count" id="message-count">{text.length} / {MAX_TEXT_LENGTH} karakter</span><div className="action-buttons"><button className="btn btn-subtle" onClick={clear} type="button" disabled={isLoading}>Temizle</button><button className="btn btn-primary" disabled={isLoading || Boolean(validateText(text))} type="submit">{isLoading ? 'Modeller çalışıyor…' : 'Modelleri karşılaştır'}<Icon name={isLoading?'refresh':'arrow'} size={17} className={isLoading?'spin':''}/></button></div></div></form>
          <div className="sample-area"><span className="sample-label">HIZLI SENARYOLAR</span><div className="sample-pills">{SAMPLES.map((s,i)=><button key={s.name} type="button" className="sample-pill" disabled={isLoading} onClick={()=>selectSample(s)}><span>{String(i+1).padStart(2,'0')}</span>{s.name}<small>{s.tag}</small></button>)}</div></div>
          {error && <div className="error-message" role="alert"><Icon name="alert" size={19}/><div className="error-content"><strong>Karşılaştırma yapılamadı</strong><p>{error}</p><button type="button" className="error-retry" onClick={() => compare()} disabled={isLoading}><Icon name="refresh" size={14}/>Yeniden dene</button></div></div>}
        </section>
        <section className="results-section" aria-labelledby="results-title" aria-busy={isLoading}><div className="result-heading"><div><div className="section-overline">02 / SONUÇLAR</div><h2 id="results-title">Model karşılaştırması</h2><p>Son çalıştırılan isteğin sonuçları</p></div>{result && <button className="copy-button" onClick={copyJson} type="button" aria-label="Karşılaştırma sonucunu JSON olarak kopyala"><Icon name="copy" size={15}/>{copyStatus || 'JSON kopyala'}</button>}</div>
          {!result ? <div className="empty-results"><div className="empty-icon"><Icon name="terminal" size={28}/></div><h3>{isLoading ? 'Modeller tahmin üretiyor…' : 'İlk karşılaştırmanı başlat'}</h3><p>{isLoading ? 'GPU yükü ve ilk çalıştırma süresi nedeniyle bu işlem biraz zaman alabilir.' : 'Yukarıya bir mesaj girip “Modelleri karşılaştır” düğmesine bas. Tahminler burada görünecek.'}</p></div> : <div className="results-content" aria-live="polite"><div className={`comparison-callout ${result.agreement ? 'callout-match' : 'callout-difference'}`}><Icon name={result.agreement?'check':'alert'} size={18}/><div><strong>{result.agreement?'İki model aynı kategoriyi tahmin etti':'Modeller farklı kategoriler tahmin etti'}</strong><span>{result.agreement ? 'Tek bir örnek üzerinde aynı sonuca ulaşmaları genel başarıyı kanıtlamaz.' : 'Farklı tahminleri inceleyin; gerçek etiket bilinmeden hangisinin doğru olduğu varsayılamaz.'}</span></div></div><div className="model-grid">{result.predictions.map((p,i)=><PredictionCard key={p.model} prediction={p} index={i}/>)}</div><div className="request-line"><span>Gönderilen metin</span><blockquote>{lastCompared}</blockquote></div></div>}
        </section>
        <footer className="notice-footer"><Icon name="shield" size={20}/><div><strong>Deneysel araştırma ortamı</strong><p>İki model de konu dışı metne bankacılık etiketi verebilir. Skorlar doğrulanmış güven olasılığı değildir. Otomatik bankacılık yönlendirmesi ve gerçek finansal işlemler kapalıdır. Tek seferlik gecikme ölçümü performans benchmark'ı sayılmaz.</p></div></footer>
      </div>
    </main>
  </div>;
}

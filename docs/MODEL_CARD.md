# TrustDesk AI — Model Card

**Sürüm:** v0.11 dokümantasyon güncellemesi · **Statü:** Araştırma/yerel portföy demosu, üretime hazır değil.

## Kullanım amacı

BANKING77 veri kümesindeki **77 İngilizce bankacılık destek niyetinin** metinden tahmini; farklı model ailelerinin deneysel karşılaştırması. Gerçek müşteri kaydı, hesap/kart/transfer yönetimi veya otomatik karar için uygun değildir.

## Modeller

| Bileşen | Teknik yaklaşım | Yerel kaynak |
|---|---|---|
| Başlangıç modeli | TF-IDF n-gram özellikleri + Logistic Regression | `artifacts/baseline.joblib` |
| Transformer | DistilBERT, görev için fine-tuning (4 epoch) | `artifacts/transformer_benchmark_4epochs/distilbert_model/` |

Her iki model de İngilizce BANKING77 niyetlerinden birini döndürür. **Konu dışı metinlerde bile kategori zorunlu olarak atanabilir.** `score` alanı iki model arasında doğrudan karşılaştırılabilir veya doğrulanmış güven olasılığı olarak kullanılabilir değildir.

## Veri ve değerlendirme protokolü

- Kaynak: [PolyAI BANKING77](https://github.com/PolyAI-LDN/task-specific-datasets/tree/master/banking_data), **CC BY 4.0**; 77 sınıf.
- Keşifsel benchmark: **6.401 eğitim**, **1.601 ayrılmış kalibrasyon (bu deneyde kullanılmayan)**, **2.001 validasyon** örneği. Resmî `test.csv` bu karşılaştırmada değerlendirme için kullanılmadı.
- DistilBERT, 4 epoch eğitildi; validasyon birkaç kez incelendiği için bu bölmenin sonucu bağımsız nihai test sonucu değildir.
- Karşılaştırma benchmark'ındaki TF-IDF aynı eğitim altkümesinde yeniden eğitildi; çalışan API'deki varsayılan `baseline.joblib` bundan farklı bir eğitim bölmesiyle üretilmiş olabilir. **Canlı tahminleri eşit eğitim koşullu model kıyaslaması saymayın.**

## Gözlenen sonuçlar

| Deney | Accuracy | Macro-F1 | Not |
|---|---:|---:|---|
| TF-IDF / Logistic Regression | **%83,36** | Raporda ayrıca incelenebilir | Eşit fit bölmeli karşılaştırmanın validasyonu |
| DistilBERT (4 epoch) | **%89,06** | **0,89034** | Aynı validasyon seti; 219 hatalı sınıflandırma |

Bunlar geçmiş yerel deney çıktılarından aktarılan, **validasyon** sonuçlarıdır. Bağımsız testte genelleme, üretim doğruluğu, güvenli OOD reddi veya farklı dillerde kalite kanıtı değildir. Raporların/model dosyalarının orijinal halleri yerel `artifacts/` dizininde saklanır; sürüm kontrolüne dahil edilmez.

## Gecikme (v0.9, yerel deney)

İki model için ayrı ayrı **3 ısınma + 20 sıralı HTTP ölçümü** (tek geliştirme bilgisayarı):

| Metrik | TF-IDF | DistilBERT |
|---|---:|---:|
| HTTP gidiş-dönüş p50 | 12,152 ms | 17,771 ms |
| HTTP gidiş-dönüş p95 | 20,129 ms | 31,778 ms |
| Model içi p50 | 0,473 ms | 10,313 ms |
| Model içi p95 | 0,763 ms | 14,478 ms |

Bu sonuçlar GPU etkin yerel API çalıştırmasından alınmıştır; Docker'daki CPU çalıştırma hızını temsil etmez. Yük testi, gerçek trafik veya SLA ölçümü değildir; küçük örneklemde p95 oynak olabilir.

## Riskler ve sınırlamalar

1. **OOD:** Alakasız/dağılım dışı metinler de 77 sınıftan biri olarak dönebilir. Ayrı deneysel OOD senaryoları ölçüldü, ancak güvenilir otomatik reddetme mekanizması kanıtlanmadı.
2. **Skor kalibrasyonu:** Ham `predict_proba` ve `softmax` skorları kalibre edilmiş güven olasılığı olarak doğrulanmadı; modellerin skorlarını birbirleriyle kıyaslamayın.
3. **Dil ve alan kayması:** İngilizce bankacılık örneklerinde çalışma; Türkçe niyet tanıma, gerçek kurum verisi veya farklı kültür/dil alanları üzerinde validasyon yok.
4. **Kritik niyetler:** Çalıntı kart, dolandırıcılık, hesap kapatma gibi sınıflarda yanlış sınıflandırma riski vardır; otomatik işlem veya müşteri yönlendirmesi yapılmaz.
5. **Veri gizliliği:** Gerçek müşteri metni girmeyin. Arayüz gösterimi için metin API yanıtında döndürülür; gerçek veriler için kayıt, erişim, anonimleştirme ve yasal süreç tasarlanmamıştır.

## Çalıştırma ve tekrar üretilebilirlik

- API: `trustdesk_unified_api_v06.py`, varsayılan yerel port 8002.
- API şeması: `/health`, `/v1/models`, `/v1/predict`, `/v1/compare`.
- Girdi: **3–2.000 karakter**; yanlış şemalar HTTP 422 döndürür.
- `automatic_routing_enabled` daima `false` olarak dönmelidir.
- Eğitim kaynak kodu depoda, ancak eğitilmiş ağırlıklar `.gitignore` ile hariç tutulmuştur. Tam yeniden üretim için orijinal veri, aynı bölme/seed, önceden eğitilmiş temel model sürümü ve eğitim ortamı gerekir.

**Atıf:** Casanueva ve diğerleri (2020), *Efficient Intent Detection with Dual Sentence Encoders*, NLP4ConvAI / ACL.

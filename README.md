# TrustDesk AI — NLP tabanlı destek talebi sınıflandırma (v0.1)

**Durum:** Başlangıç kodu. Resmî BANKING77 verisi üzerinde henüz eğitim sonucu raporlanmadı. Bu depo tam/üretim kullanıma hazır bir ürün veya dağıtılmış canlı demo değildir.

## Amaç
Gerçek müşteri taleplerinden, kullanıcının *niyetini* (77 sınıftan birini) belirleyen bir ML modeli eğitmek. Düşük skorlu tahminleri otomatik işlem yapmak yerine insan incelemesine bırakmak. Nihai hedef: karşılaştırmalı ML/Transformer değerlendirmesi + FastAPI + React + testler + Docker + model kartı.

**Mevcut v0.1:** veri indirme / doğrulama, TF-IDF + lojistik regresyon eğitimi, eğitim/validasyon bölmesi, başlangıç metrikleri, FastAPI `/predict`, testler, CI iskeleti ve dokümantasyon. **Henüz yok:** Transformer fine-tuning, React arayüzü, model skoru kalibrasyonu, canlı dağıtım ve kullanıcı kimlik doğrulaması.

## Ön koşullar
Python 3.11 veya 3.12 önerilir. Python 3.13 de desteklenen paket sürümleriyle çalışabilir. Komutlar proje klasörü kökünde çalıştırılır.

### Windows PowerShell
```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[dev]"
python -m trustdesk.download_data
python -m trustdesk.train
uvicorn trustdesk.api:app --reload
```
`py -3.11` bulunmazsa sisteminizde kurulu sürümü kontrol edin (`py --list`) ve uygun sürümü seçin. PowerShell aktivasyon engeli varsa `./.venv/Scripts/python.exe` ile komutları çalıştırabilirsiniz.

### Linux/macOS
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
python -m trustdesk.download_data
python -m trustdesk.train
uvicorn trustdesk.api:app --reload
```

API: http://127.0.0.1:8000/docs

### Tahmin örneği (PowerShell)
```powershell
$body = @{ text = "My card was stolen yesterday" } | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8000/predict" -ContentType "application/json" -Body $body
```

### Test
```bash
pytest -q
```

Testler küçük **yapay örnekler** kullanır; dış veri indirmeden çalışır. Gerçek model performansı olarak sunulamaz.

## Veri kaynağı ve etik
Kaynak: [PolyAI BANKING77](https://github.com/PolyAI-LDN/task-specific-datasets/tree/master/banking_data); İngilizce 10.003 train + 3.080 test örneği; 77 etiket. CSV sütunları `text`, `category`. Veri lisansı **CC BY 4.0**. Bu depo veriyi yeniden dağıtmaz; `download_data` dosyaları orijinal kaynaktan çeker ve SHA-256 değerlerini doğrular.

Alıntı: Casanueva, Temčinas, Gerz, Henderson, Vulić (2020), *Efficient Intent Detection with Dual Sentence Encoders*, NLP4ConvAI, ACL.

Kapsam: Model niyet sınıflandırır; **müşterinin sorununu çözmez, finansal karar vermez**. Bu İngilizce model için Türkçe dil desteği veya gerçek dünya üretim güvenilirliği iddia edilmez.

## Deney tasarımı
- Resmî train içinden tekrarlanabilir **stratified train/validation** bölmesi (seed 42); resmi test verisi ayrı tutulur.
- İlk baseline: TF-IDF (1–2 gram) + Logistic Regression.
- Metrikler: accuracy, macro-F1, weighted-F1 ve örnek/sınıf bazlı rapor. Metrikler `artifacts/metrics.json` ve `artifacts/classification_report.txt` konumuna yazılır.
- Önemli: Resmî test verisini tekrar tekrar hyperparameter seçmek için kullanmayacağız. İleri model karşılaştırması için validasyon ve ayrı dondurulmuş test protokolü oluşturacağız.
- `score` lojistik regresyonun ham `predict_proba` çıktısıdır; **kalibre edilmiş bir güven olasılığı olduğu gösterilmemiştir**. `needs_human_review` kuralları prototip amaçlıdır; eşikler kalibrasyon ve risk testleriyle seçilmelidir.

## Mimari
`data source -> hash verified CSV -> train/validation -> TF-IDF + LR -> report + trusted local model -> FastAPI -> review flag`

Kod dizini:
```
src/trustdesk/
  download_data.py  # doğrulanmış veri indirme
  data.py           # CSV yükleme/şema kontrolü
  train.py          # ML deney ve raporlama
  inference.py      # tahmin ve düşük skor politikası
  api.py            # FastAPI servisi
```

Ayrıntılar: `docs/ROADMAP.md`, `docs/MODEL_CARD.md`, `docs/SECURITY.md`, `docs/CV_GUIDANCE.md`.

## Güvenlik sınırları
- Test için üretilen `joblib` artefaktı **yalnızca kendi oluşturduğunuz ve güvendiğiniz yerel dosyadan** yükleyin. Joblib/pickle formatı güvenilmeyen kaynaklardan yüklenmemelidir.
- API şu anda sadece **yerel geliştirme içindir**: kimlik doğrulama, oran sınırlama, izleme ve kişisel veri koruma katmanı henüz yok. İnternete açık biçimde deploy etmeyin.
- Gerçek müşteri bilgilerini repoya veya hata kayıtlarına eklemeyin.

## Sürüm politikası
Çalışan, ölçülen ve belgelenen aşamalardan sonra GitHub'a yayınlayın. Başlangıç şablonunu bir başarı veya deneyim olarak CV'ye eklemeyin; kendi geliştirip test ettiğiniz işlevleri ve ölçümlerinizi ekleyin.

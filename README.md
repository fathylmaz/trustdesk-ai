# TrustDesk AI

**BANKING77 üzerinde iki modelin İngilizce müşteri destek niyetlerini karşılaştırdığı, yerel çalışan deneysel NLP uygulaması.**

TrustDesk AI; **TF-IDF + Logistic Regression** ve **fine-tune edilmiş DistilBERT** modellerini FastAPI üzerinden sunar. React arayüzü, aynı mesaj için iki modelin ilk üç tahminini, ham skorlarını ve işlem sürelerini yan yana gösterir. Docker Compose ile React + Nginx ve FastAPI bir arada, yalnızca yerel bilgisayarda çalıştırılabilir.

> **Araştırma / portföy projesi:** Üretim ortamına veya gerçek bankacılık işlemlerine hazır değildir. Gerçek müşteri verisi girmeyin. Her iki model de konu dışı metinlere 77 sınıftan birini atayabilir; ham skorlar kalibre edilmiş güven olasılığı değildir. Otomatik yönlendirme ve finansal işlemler kapalıdır.

## Özellikler

- **77 niyet sınıfı:** PolyAI BANKING77'nin İngilizce destek talepleri.
- **İki ayrı yaklaşım:** TF-IDF + Logistic Regression ve 4 epoch fine-tune edilmiş DistilBERT.
- **Karşılaştırmalı FastAPI:** `/health`, `/v1/models`, `/v1/predict` ve `/v1/compare`.
- **React arayüzü (v0.8.2):** Türkçe açıklamalar, teknik etiketler, örnek senaryolar, bağlantı/hata mesajları, klavye erişimi ve duyarlı tasarım.
- **Yerel Docker (v0.10):** Nginx üzerinden tek adres; FastAPI iç ağda; CPU çalıştırma seçeneği.
- **Kalite kontrolleri:** Python otomatik testleri, React testleri/derlemesi ve gerçek yerel API'de sözleşme/gecikme deneyi.
- **CI tanımı:** `.github/workflows/ci.yml` — Python testleri ve React test/derleme; *GitHub üzerindeki ilk çalıştırma henüz doğrulanmadı*.

## Mimari

```text
Tarayıcı (React)
      │
      ▼
Nginx :8080 ─── /api/* yönlendirme ───► FastAPI :8002 (Docker iç ağı)
                                             │
                               ┌─────────────┴──────────────┐
                               ▼                            ▼
                         TF-IDF + LR                   DistilBERT
                               └─────────────┬──────────────┘
                                             ▼
                                  Tahminler / ham skorlar
```

Docker haricindeki geliştirme modunda React `127.0.0.1:5173`, birleşik FastAPI ise `127.0.0.1:8002` üzerinden çalışır.

## Yerel Docker ile çalıştırma (Windows PowerShell)

**Gereksinimler:** Docker Desktop + çalışan Linux Engine/WSL2, model dosyaları bilgisayarınızda mevcut olmalı. Docker imajları **model ağırlıklarını içermez** ve otomatik indirmez. İlk CPU imajının kurulması uzun sürebilir.

Beklenen yerel dosyalar:

```text
artifacts/
  baseline.joblib
  transformer_benchmark_4epochs/
    distilbert_model/
      config.json
      model.safetensors
      tokenizer.json  (veya vocab.txt)
      ...
```

Model klasöründeki diğer tokenizer dosyalarını da koruyun. `.gitignore`, `artifacts` içeriğini Git dışında tutar. `.joblib` yalnızca **kendi eğittiğiniz ve güvendiğiniz** yerel kaynaktan yüklenmelidir.

Projenin ana klasöründe:

```powershell
.\.venv\Scripts\python.exe .\scripts\check_deploy_inputs.py
docker compose -f compose.yaml config --quiet
docker compose -f compose.yaml up --build
```

İlk komut için mevcut Python sanal ortamı gerekir; sanal ortam yoksa ilgili bağımlılıklarla oluşturulmalıdır. Kontrol, ağırlıkları yüklemez; dosya/etiket metadata'sını denetler.

Uygulama: **http://127.0.0.1:8080/**

Başka bir PowerShell penceresinde:

```powershell
docker compose -f compose.yaml ps
Invoke-RestMethod http://127.0.0.1:8080/api/health | Format-List *
```

Beklenen sağlık alanları: `status: ok`, `models_loaded: True`, `device: cpu`, `automatic_routing_enabled: False`. Durdurmak için `docker compose -f compose.yaml down` komutunu kullanın.

**Not:** İlk kurulum, ağırlıkları kendi bilgisayarında bulunan geliştirici için doğrulandı. Model dosyaları bulunmayan temiz bir clone, Docker'ı tek başına tam çalışan modele dönüştürmez; önce kendi yerel checkpoint'lerinizi hazırlamanız gerekir.

## Geliştirme ortamında çalıştırma

Aşağıdaki adımlar, modellerin yerel `artifacts` klasöründe bulunduğunu ve Python/Node bağımlılıklarının daha önce kurulduğunu varsayar.

```powershell
# Terminal 1 — proje ana klasörü
.\.venv\Scripts\python.exe -m uvicorn trustdesk_unified_api_v06:app --host 127.0.0.1 --port 8002
```

```powershell
# Terminal 2
cd frontend
npm.cmd ci
npm.cmd run dev
```

Tarayıcı: **http://127.0.0.1:5173/**. API belge ekranı: **http://127.0.0.1:8002/docs**.

Örnek PowerShell isteği:

```powershell
$body = @{ text = 'I want to cancel my bank account.' } | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri 'http://127.0.0.1:8002/v1/compare' -ContentType 'application/json' -Body $body
```

## Model sonuçları — keşifsel validasyon

| Model | Accuracy | Macro-F1 | Değerlendirme |
|---|---:|---:|---|
| TF-IDF + Logistic Regression | **%83,36** | — | DistilBERT karşılaştırması için aynı 2.001 örneklik validasyon bölmesinde yeniden eğitilmiş baseline |
| DistilBERT (4 epoch) | **%89,06** | **0,89034** | 6.401 eğitim örneği ve aynı 2.001 örneklik validasyon bölmesi |

Bu değerler **validasyon deneyi** sonuçlarıdır; bağımsız, dondurulmuş resmî test seti başarısı veya üretim performansı iddiası değildir. Modellerin önceki deneylerinde farklı train/validation bölmeleri de kullanıldığından, farklı koşullardaki sonuçlar doğrudan karşılaştırılmamalıdır. Ayrıca canlı API'de kullanılan mevcut TF-IDF checkpoint'i ile araştırma karşılaştırmasında aynı fit bölmesinde eğitilen TF-IDF modeli farklı olabilir. Detaylar: [Model kartı](docs/MODEL_CARD.md).

### Yerel API gecikme ölçümü (v0.9)

Tek bir bilgisayarda her model için **3 ısınma + 20 sıralı ölçüm isteği**; model içi süre ve HTTP gidiş-dönüş süreleri farklı metriklerdir:

| Ölçüm | TF-IDF | DistilBERT |
|---|---:|---:|
| API yanıt p50 | 12,152 ms | 17,771 ms |
| API yanıt p95 | 20,129 ms | 31,778 ms |
| Model işlem p50 | 0,473 ms | 10,313 ms |
| Model işlem p95 | 0,763 ms | 14,478 ms |

Bu ölçüm, yerel GPU kurulumunda gerçekleştirilmiş kısa, sıralı istek deneyidir; **Docker CPU ölçümü değildir**, eşzamanlı yük testi veya hizmet seviyesi garantisi değildir. Ham yerel rapor `artifacts/qa_v09/report.json` içinde tutulur ve Git'e eklenmez.

## Testler

Proje ana klasöründe (bağımlılıklar kuruluysa):

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe .\trustdesk_qa_v09.py --runs 20 --warmup 3
```

İkinci komut için yerel FastAPI'nin **8002 portunda çalışıyor olması** ve model dosyalarının yüklü olması gerekir.

```powershell
cd frontend
npm.cmd ci
npm.cmd run test
npm.cmd run build
```

Yerel kontrollerde **66 Python testi** ve **22 React testi** geçti; React üretim derlemesi de tamamlandı. Windows üzerinde bir Python test çalıştırması sırasında `scikit-learn` DLL'i uygulama denetimi tarafından geçici olarak engellendi; sonraki çalıştırmada testler geçti. GitHub Actions'ın Linux üzerinde ilk koşusu henüz doğrulanmadı.

## Veri ve sorumlu kullanım

- **Veri:** [PolyAI BANKING77](https://github.com/PolyAI-LDN/task-specific-datasets/tree/master/banking_data) (CC BY 4.0), 77 sınıf, resmî train/test bölmeleri. Veri otomatik indirme yolu mevcut olmakla birlikte ham veri Git'e dahil edilmez.
- **Atıf:** Casanueva ve diğerleri (2020), *Efficient Intent Detection with Dual Sentence Encoders*, NLP4ConvAI / ACL.
- **Dil:** Yalnızca İngilizce bankacılık metinleri üzerinde değerlendirilmiştir. Arayüz etiketlerinin Türkçeleştirilmesi, Türkçe niyet tanıma desteği anlamına gelmez.
- **Güvenlik:** Kimlik doğrulama, rate limiting, üretim düzeyinde gözlemleme ve gerçek müşteri verisi yönetimi bulunmaz. İnternete açık servis olarak yayımlamayın.
- **Model sınırlamaları:** Ham skorlar kalibre edilmiş güven anlamına gelmez; OOD algılama/güvenli yönlendirme kanıtlanmış değildir; otomatik bankacılık işlemi yoktur.

Ek belgeler: [Model kartı](docs/MODEL_CARD.md) · [Güvenlik](docs/SECURITY.md) · [Yol haritası](docs/ROADMAP.md) · [Katkı/CV ilkeleri](docs/CV_GUIDANCE.md).

**Yayın durumu:** Yerel demo ve Docker doğrulandı; herkese açık canlı servis kurulmadı. Kod GitHub'a aktarılırken CI sonuçları ayrıca kontrol edilecektir.

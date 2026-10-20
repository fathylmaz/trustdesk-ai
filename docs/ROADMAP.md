# TrustDesk AI — Geliştirme yol haritası

Bu dosya, **yerel olarak doğrulanan işler** ile **henüz doğrulanmamış gelecek işleri** ayırır. Sürüm numaraları proje içi kilometre taşlarını ifade eder; üretim sürümü oldukları iddia edilmez.

## Yerel olarak doğrulanan kilometre taşları

- [x] BANKING77 veri yükleme, doğrulama ve TF-IDF + Logistic Regression baseline.
- [x] Keşifsel hata, eşik, kalibrasyon, OOD ve risk değerlendirmeleri (üretim güvenliği iddiası yok).
- [x] 4 epoch DistilBERT fine-tuning ve aynı validasyon bölmesi üzerinde deneysel karşılaştırma.
- [x] Birleşik FastAPI (`/health`, `/v1/compare`, `/v1/predict`) ve iki model yanıtı.
- [x] React v0.8.2: Türkçe arayüz etiketleri, erişilebilirlik, duyarlı görünüm ve hata mesajları.
- [x] v0.9: yerel entegrasyon sözleşmesi ve sıralı gecikme ölçümleri.
- [x] v0.10: yerel Docker Compose (Nginx + FastAPI, CPU), checkpoint'lerin salt okunur bağlanması.
- [x] Yerelde 66 Python testi, 22 React testi ve React üretim derlemesinin başarılı son koşusu.
- [x] v0.11: yerel Git deposu ve ilk commit; kişisel e-posta yerine GitHub `noreply` adresi.

## Sıradaki aşamalar

- [x] Güncel README, model kartı, güvenlik ve yol haritası belgelerini commit etmek (v0.11).
- [ ] Public GitHub depo paylaşımından önce tüm CSV ve commit geçmişi için kişisel veri, secret ve lisans kontrolleri (ilk örnekler ve sınırlı desen taraması yapıldı; kapsamlı denetim yapılmadı).
- [x] GitHub'a ilk `push`; GitHub Actions Python/React işleri Linux üzerinde geçti (commit `414e5ae`).
- [x] v0.12 için gerçek yerel React v0.8.2 ekran görüntülerinin belgelere hazırlanması.
- [ ] Kod lisansı için sorumlu kişi/katkıcı onayı; `LICENSE` eklenmesi veya lisanssız politika kararı.
- [ ] GitHub üzerinde `docs/screenshots/` görüntülerini doğrulamak ve release belgelerini commit etmek.
- [ ] Görünürlüğü Public yapmadan önce [yayın kontrol listesi](RELEASE_CHECKLIST.md) üzerinden onay almak.
- [ ] İsteğe bağlı kısa demo videosu.
- [ ] Tam yeniden üretilebilir deney metadatası, bağımsız test protokolü ve sistematik hata analizi.
- [ ] Bağımlılık taramaları, güvenlik ve erişilebilirlik testleri, eşzamanlı yük testi.
- [ ] Üretim kimlik doğrulaması, oran sınırlama, KVKK süreçleri ve gerçek OOD/human-in-the-loop validasyonu **ayrı kapsam**.

## Katkıların belgelenmesi

Katkı sahipleri geçmiş Git commit'leri ve PR'ları ile kanıtlanmalı; bireysel katkı yapılmadan CV'de ortak başarı kişisel çalışma gibi gösterilmemelidir. Başlangıçtaki görev paylaşımı taslakları gerçekleşmiş katkı kaydı değildir.

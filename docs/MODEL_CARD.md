# TrustDesk AI — Model Card (DRAFT)

**Model:** TF-IDF + Logistic Regression baseline. **Durum:** eğitim ve benchmark sonuçları bekleniyor.

- Görev: İngilizce çevrimiçi bankacılık destek taleplerinin 77 niyet sınıfına ayrılması.
- Veri: BANKING77 (CC BY 4.0), resmi train/test bölmeleri.
- Dağılım dışı (OOD) destek: v0.1 için ölçülmedi; düşük skor = güvenilir OOD tespiti değildir.
- Türkçe: eğitilmedi ve değerlendirilmedi.
- Metrikler: accuracy [ölçülecek], macro-F1 [ölçülecek], latency [ölçülecek].
- İnsan incelemesi: basit skor eşiği prototip aşamasındadır; kalibrasyon gerekli.
- Sınırlar: İngilizce, simüle edilmiş çevrim içi bankacılık senaryosu; gerçek banka müşteri verisi/servisi kullanılmıyor.
- Sorumlu kullanım: Otomatik hesap, kart, transfer veya finansal karar verilmez.
- Veri lisansı/atıf: PolyAI BANKING77; Casanueva ve diğerleri (2020).

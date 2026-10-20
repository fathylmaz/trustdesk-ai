# TrustDesk AI — CV ve portföy anlatım kılavuzu

Proje artık **yerel demo, model deneyleri, otomatik testler ve Docker** içeren çalışan bir araştırma portföyüdür. Bununla birlikte **canlı bankacılık ürünü**, internete açık üretim servisi veya bağımsız resmî testte doğrulanmış sistem değildir.

## Doğrulanmış ve dikkatle kullanılabilecek ifadeler

- “BANKING77 veri kümesinde 77 sınıflı İngilizce niyet sınıflandırması için TF-IDF/Logistic Regression ve DistilBERT yaklaşımlarını karşılaştıran yerel deneysel uygulama geliştirdim.”
- “FastAPI, React ve Nginx/Docker Compose ile iki modelin tahminlerini karşılaştıran bir yerel demo oluşturdum.”
- “Yerel geliştirme ortamında 66 Python ve 22 React otomatik testini çalıştırdım; API için tekrarlı gecikme ölçümleri raporladım.”

Bu ifadeler yalnızca **kişinin gerçekten yaptığı iş** için kullanılmalı; kullanıcı katkıları GitHub commit/PR geçmişiyle ayrı ayrı kaydedilmelidir. Herhangi bir ikinci kişiye doğrulanmamış iş atfedilmemelidir.

## Metrikleri doğru sunma

- DistilBERT: yaklaşık **%89,06 validasyon accuracy**, **0,89034 macro-F1** (4 epoch, keşifsel).
- Eşit fit bölmeli TF-IDF: yaklaşık **%83,36 validasyon accuracy**.
- Bunlar **validasyon** rakamlarıdır, bağımsız resmî test veya gerçek bankacılık performansı değildir.
- Gecikme rakamları **yerel, sıralı 20 istek** üzerinden alınmıştır; ölçeklenebilirlik/SLA iddiasına dönüştürülmemelidir.
- GitHub Actions ancak gerçek GitHub koşuları geçtikten sonra “CI doğrulandı” şeklinde yazılmalıdır.

## Sunum kontrol listesi

- [ ] GitHub'a güncel README ve model kartını ekle.
- [ ] İlk push sonrası Actions sonucunu kontrol et.
- [ ] Gerçek proje ekran görüntüleri ve mimari anlatımı ekle.
- [ ] Katkıları kişiye göre ayrı ayrı belgele.
- [ ] Kullanılacak proje lisansını bilinçli olarak belirle.

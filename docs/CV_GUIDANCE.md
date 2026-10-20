# TrustDesk AI — CV ve portföy anlatım kılavuzu

Proje artık **yerel demo, model deneyleri, otomatik testler ve Docker** içeren çalışan bir araştırma portföyüdür. Bununla birlikte **canlı bankacılık ürünü**, internete açık üretim servisi veya bağımsız resmî testte doğrulanmış sistem değildir.

## Doğrulanmış ve dikkatle kullanılabilecek ifadeler

- “BANKING77 veri kümesinde 77 sınıflı İngilizce niyet sınıflandırması için TF-IDF/Logistic Regression ve DistilBERT yaklaşımlarını karşılaştıran yerel deneysel uygulama geliştirdim.”
- “FastAPI, React ve Nginx/Docker Compose ile iki modelin tahminlerini karşılaştıran bir yerel demo oluşturdum.”
- “Yerel geliştirme ortamında 66 Python ve 22 React otomatik testini çalıştırdım; API için tekrarlı gecikme ölçümleri raporladım.”
- “GitHub Actions üzerinde Python ve React test/derleme işlerinin başarılı çalışmasını doğruladım (ilk doğrulanan CI koşusu, `414e5ae`).”

Bu ifadeler yalnızca **kişinin gerçekten yaptığı iş** için kullanılmalı; kullanıcı katkıları GitHub commit/PR geçmişiyle ayrı ayrı kaydedilmelidir. Herhangi bir ikinci kişiye doğrulanmamış iş atfedilmemelidir.

## Metrikleri doğru sunma

- DistilBERT: yaklaşık **%89,06 validasyon accuracy**, **0,89034 macro-F1** (4 epoch, keşifsel).
- Eşit fit bölmeli TF-IDF: yaklaşık **%83,36 validasyon accuracy**.
- Bunlar **validasyon** rakamlarıdır, bağımsız resmî test veya gerçek bankacılık performansı değildir.
- Gecikme rakamları **yerel, sıralı 20 istek** üzerinden alınmıştır; ölçeklenebilirlik/SLA iddiasına dönüştürülmemelidir.
- GitHub Actions Python/React işleri ilk GitHub `push` sonrasında doğrulandı. Bu sonuç Docker, GPU veya güvenlik taramalarının CI'da yapıldığı anlamına gelmez.

## Sunum kontrol listesi

- [x] GitHub'a güncel README ve model kartını ekle.
- [x] İlk push sonrası Actions sonucunu kontrol et.
- [x] v0.12 belge paketine gerçek arayüz ekran görüntüleri ve mimari anlatımını ekle (GitHub push sonrası görünümü doğrula).
- [ ] Katkıları kişiye göre ayrı ayrı belgele.
- [ ] Kullanılacak proje lisansını bilinçli olarak belirle.

# Proje teslim planı (iki geliştirici)

Bu dosya hedefleri belirtir; tamamlanmış iş olarak gösterilmez.

## Hafta 1 — gerçek bir baseline
- [ ] İki kişi de Python sanal ortamı ve Git temellerini kurdu.
- [ ] Orijinal verinin lisansı, içeriği, sınıf sayısı ve bölmeleri kontrol edildi.
- [ ] Baseline eğitildi; validasyon metrikleri kaydedildi.
- [ ] Hata analizi: başarısız en az 20 örnek incelendi, nedenleri yazıldı.
- [ ] Unit/API testleri çalıştı; ortak GitHub deposunda iki kişinin anlamlı commit ve PR'ları var.

## Hafta 2 — mühendislik
- [ ] FastAPI giriş doğrulaması ve servis sağlık kontrolü.
- [ ] Confidence calibration / risk için ayrı değerlendirme protokolü.
- [ ] Docker ile yerel demo; CI testleri yeşil.
- [ ] İlk React arayüzü: talepler ve model tahminleri. (Henüz yapılmadı.)
- [ ] Model kartına ölçülmüş sonuç, hata örnekleri ve sınırlamalar yazıldı.

## Hafta 3–4 — derin öğrenme ve yayın
- [ ] DistilBERT veya uygun küçük Transformer fine-tune edildi (donanım uygunsa).
- [ ] Transformer ile baseline aynı protokolde macro-F1, latency, bellekte kullanım, CPU süresi açısından karşılaştırıldı.
- [ ] Belirsiz ve kapsam-dışı sorgu testleri, veri sızıntısı ve kullanıcı gizliliği değerlendirmeleri.
- [ ] Görsel demo/screenshot, kurulum videosu, temiz README, lisans seçimi.
- [ ] Her kişinin yaptığı katkı ve ölçülen teknik sonuçlar CV'ye yazıldı.

## Örnek görev paylaşımı
**Fatih:** veri hattı ve model baseline; daha sonra FastAPI / Docker.
**Gamze:** hata analizi ve test tasarımı; daha sonra React arayüzü.
**Ortak:** metrikler, güvenlik, PR incelemeleri, dokümantasyon; ikinci sprintte görev değiştirme.

Birinizin tüm işi yapıp diğerinin CV'sine aynı başarıyı yazmaması kritik. Her adımda kişisel katkıyı GitHub üzerinde doğrulanabilir biçimde tutun.

## Definition of Done
Kod kurulabilir; testler geçer; veri kaynağı/lisansı açık; tekrar edilebilir metrikler vardır; hata/sınırlamalar kayıtlıdır; çalışan demo vardır; gözlemleyebileceğiniz PR/commit geçmişi vardır.

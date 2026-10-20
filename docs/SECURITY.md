# TrustDesk AI — Güvenlik ve sorumlu kullanım

> **Yalnızca yerel, deneysel demonstrasyon.** Bu proje internete açık bankacılık servisi, müşteri destek otomasyonu veya finansal işlem platformu olarak kullanılamaz.

## Mevcut teknik önlemler (kaynak kodda)

- [x] **Girdi doğrulaması:** Birleşik FastAPI'de metin uzunluğu 3–2.000 karakter; model seçimi izin verilen iki modelle sınırlı.
- [x] **Otomatik işlem kapalı:** API yanıtında `automatic_routing_enabled: false`; banka hesabı, kart veya transfer işlemi gerçekleştirilmez.
- [x] **Yerel konteyner ağı:** Docker Compose yalnızca Nginx'i `127.0.0.1:8080` üzerinde yayımlar; FastAPI container iç ağında kalır.
- [x] **Container kullanıcı ayrımı:** Backend imajı ayrıcalıksız `appuser` (UID 10001) ile çalışır.
- [x] **Ağırlıklar imaja dahil edilmez:** `artifacts/` yalnızca yerel klasörden salt okunur bağlanır ve Git tarafından dışlanır.
- [x] **CORS kapsamı:** Ayrı geliştirme modundaki birleşik API sadece yerel React origin'lerini kabul edecek şekilde ayarlanmıştır.
- [x] **Nginx temel ayarları:** İstek gövdesi üst sınırı ve bazı güvenlik başlıkları mevcuttur; bunlar tek başına üretim güvenliği sağlamaz.
- [x] **Model dosyası bütünlüğü için metadata kontrolü:** Checkpoint etiket eşlemesi ön kontrolü vardır; bu kriptografik dosya kökeni kanıtı değildir.

## Bilinen eksikler / yayımlamadan önce gerekli olanlar

- [ ] Kimlik doğrulama ve yetkilendirme.
- [ ] TLS/HTTPS, ters proxy sertifika yönetimi ve ağ güvenlik modeli.
- [ ] Rate limiting, anti-abuse kontrolleri, trafik/DoS limitleri ve kapsamlı izleme.
- [ ] Girdi, log ve kişisel veri yaşam döngüsü (maskeleme, saklama, silme, erişim denetimi); KVKK gereklilikleri.
- [ ] Bağımlılık/SBOM, container ve secret scan; kütüphane güncellemeleri ile uyumluluk denetimi.
- [ ] Modelin gerçek kullanım alanında dağılım kayması, adversarial girdi, kalibrasyon ve OOD ölçümleri; insan inceleme iş akışı.
- [ ] Ayrı üretim testleri, eşzamanlı yük/stres testi, felaket kurtarma ve kurumsal devreye alma incelemesi.

## Model ve veri dosyaları

- `.joblib` dosyası pickle tabanlıdır. **Yalnızca kendi eğittiğiniz veya tamamen güvendiğiniz dosyaları** yükleyin; güvenilmeyen `.joblib` çalıştırma riski taşır.
- Transformer checkpoint dosyaları `safetensors` biçimindedir; tokenizer ve `config.json` kaynakları da doğrulanmalıdır. `trust_remote_code=False` ve `local_files_only=True` yaklaşımı kullanılır.
- `.gitignore`: `.venv`, `artifacts`, model ağırlıkları ve ortam değişkeni dosyaları dışlanır. Bu kurallar **daha önce Git'e eklenmiş sırları geri almaz**.
- GitHub'a göndermeden önce `git status`, `git ls-files` ve secret taraması kontrol edilmelidir. Sır tespiti için basit desen aramaları kapsamlı denetimin yerini tutmaz.
- Gerçek müşterilere ait mesaj, kimlik bilgisi, hesap numarası, API anahtarı veya erişim token'ı örneklerde, ekran görüntülerinde ve commit'lerde kullanılmamalıdır.

## İhlal/yanlış kullanım bildirimi

Güvenlik sorunu fark edilirse gizli anahtarları veya kişisel verileri herkese açık bir issue'ya eklemeyin. Depo sahibine özel/güvenli bir kanal üzerinden bildirin. Bu proje için resmî güvenlik müdahale SLA'sı tanımlanmamıştır.

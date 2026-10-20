# TrustDesk AI — Public yayın kontrol listesi (v0.12)

Bu liste **özel depoyu Public yapmadan önce** kontrol edilmelidir. Kodun GitHub'a push edilmesi, internet üzerinde çalışan API hizmeti başlatmaz.

## Dosyalar / belge doğruluğu

- [ ] README'nin GitHub üzerinde iki gerçek ekran görüntüsünü ve Mermaid mimari diyagramını doğru gösterdiğini kontrol et.
- [ ] Teknik iddiaları doğrula: DistilBERT **%89,06 validasyon**, TF-IDF **%83,36 validasyon**; bağımsız test sonucu iddia etme.
- [ ] `README.md` üzerinde ilk GitHub CI koşusu doğru gösteriliyor mu? [Başarılı koşu](https://github.com/fathylmaz/trustdesk-ai/actions/runs/39398907454).
- [ ] Yerel README ve web arayüzündeki metinlerde Türkçe/İngilizce karakter kodlaması doğru mu?

## Gizlilik ve güvenlik

- [ ] `git ls-files` ve tüm **commit geçmişinde** API key, kişisel e-posta, parola, müşteri verisi ve hassas içerik bulunmadığını kontrol et. `.gitignore` geçmişte commit edilmiş veriyi silmez.
- [ ] `data/evaluation/*.csv` dosyalarının tüm satırlarını manuel incele. İlk taramada görünür e-posta adresi bulunmaması kapsamlı kişisel veri kontrolü değildir.
- [ ] `docs/screenshots/*.png` dosyalarında gerçek müşteri verisi, e-posta, kişisel hesap, kullanıcı kimliği ya da token bulunmadığını insan gözüyle onayla.
- [ ] `artifacts/`, `.venv/`, `node_modules/`, `*.safetensors` ve `*.joblib` dosyalarının Git'e dahil olmadığını doğrula.
- [ ] Sadece kaynak kodu paylaşılır: Docker checkpoint dosyalarını otomatik indirmez, internet erişimi ve authentication mevcut değildir.
- [ ] GitHub hesabının e-posta gizliliği seçeneklerini ayarla; yazar bilgileri geçmiş commit'lerde de güvenli mi kontrol et.

## Lisans, katkı, görünürlük

- [x] Proje sahibi tek insan geliştirici olduğunu doğruladı; yazılım lisansı olarak **MIT** seçildi. BANKING77 veri kümesinin CC BY 4.0 lisansı ayrıdır.
- [x] MIT lisansı seçildi ve `LICENSE` belgesi hazırlandı. **GitHub’a commit/push edilmesi ayrıca doğrulanacak.**
- [x] Tek insan geliştirici/bakım sorumlusu Fatih Yılmaz olarak kullanıcı tarafından doğrulandı; `docs/CONTRIBUTORS.md` hazırlandı.
- [ ] GitHub Repository > Settings > General > Change repository visibility adımını ancak yukarıdakiler tamamlandıktan sonra kullan.
- [ ] Public yapıldıktan sonra dışarıdan açılan repo bağlantıları, README görselleri ve Actions badge'i kontrol et.

## Sürüm kontrolü ve son doğrulama

```powershell
git --no-pager diff --stat
git diff --check
git status --short
# Belgeleri ekledikten sonra:
git add LICENSE README.md docs/CONTRIBUTORS.md docs/ROADMAP.md docs/CV_GUIDANCE.md docs/RELEASE_CHECKLIST.md
git diff --cached --check
git --no-pager diff --cached --name-only
```

**Not:** `git push`, `Public` görünürlük değişikliği ve yazılım lisansı seçimi ayrı kararlardır.

## v0.12.2 ek kontrol

- [ ] MIT telif satırı (`Copyright (c) 2026 Fatih Yılmaz`), iki dilli README ve katkı belgesini GitHub üzerinde kontrol et.
- [ ] `git push` sonrası ilgili GitHub Actions işlerinin başarılı olduğunu doğrula.
- [ ] Private → Public geçişinden önce kapsamlı kişisel veri/secret/lisans ve ekran görüntüsü denetimini tamamla; depoyu onaysız açma.

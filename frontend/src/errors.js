/** Safe, user-facing error messages. Technical diagnostics belong in developer logs. */
export function friendlyApiError(error) {
  if (error?.name === 'AbortError') {
    return 'İstek beklenenden uzun sürdü. Lütfen kısa bir süre sonra tekrar deneyin.';
  }
  if (error instanceof TypeError) {
    return 'Yapay zekâ servisine ulaşılamıyor. Servis yeniden açıldığında tekrar deneyin.';
  }
  if (error?.kind === 'server') {
    return 'Sunucuda geçici bir sorun oluştu. Lütfen biraz sonra yeniden deneyin.';
  }
  if (error?.kind === 'request') {
    return 'Mesaj işlenemedi. Lütfen metni kontrol edip yeniden deneyin.';
  }
  return 'Karşılaştırma tamamlanamadı. Lütfen tekrar deneyin.';
}

export function apiResponseError(status) {
  const err = new Error(`API HTTP ${status}`);
  err.kind = status >= 500 ? 'server' : 'request';
  return err;
}

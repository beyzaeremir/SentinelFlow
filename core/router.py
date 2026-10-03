import numpy as np
from sentence_transformers import SentenceTransformer

# Hafif ve hızlı açık kaynak embedding modeli (~80 MB)
model = SentenceTransformer("all-MiniLM-L6-v2")

ONBELLEK_VERILERI = [
    {
        "soru": "yıllık izin süresi kaç gündür",
        "cevap": "Şirket politikamıza göre 1. yılını dolduranların izin hakkı 14 gündür."
    },
    {
        "soru": "fatura onay limiti ne kadar",
        "cevap": "Yönetici onayı gerektirmeyen üst limit 50.000 TL'dir."
    },
    {
        "soru": "mesai saatleri ve esnek çalışma kuralları nelerdir",
        "cevap": "Çalışma saatleri 09:00 - 18:00 arasındadır. Çarşamba günleri uzaktan çalışma hakkı mevcuttur."
    }
]

KAYITLI_SORULAR = [oge["soru"] for oge in ONBELLEK_VERILERI]
KAYITLI_VEKTORLER = model.encode(KAYITLI_SORULAR, normalize_embeddings=True)


def kosinus_benzerligi(vektor_a: np.ndarray, vektorler_b: np.ndarray) -> np.ndarray:
    return np.dot(vektorler_b, vektor_a)


def anlamsal_onbellek_kontrolu(kullanici_sorusu: str, esik_degeri: float = 0.65) -> tuple[bool, str, float, float]:
    soru_vektoru = model.encode(kullanici_sorusu, normalize_embeddings=True)
    skorlar = kosinus_benzerligi(soru_vektoru, KAYITLI_VEKTORLER)
    
    en_yuksek_indeks = int(np.argmax(skorlar))
    en_yuksek_skor = float(skorlar[en_yuksek_indeks])

    # 4 değer döndürüyoruz: bulundu_mu, cevap, maliyet, benzerlik_skoru
    if en_yuksek_skor >= esik_degeri:
        bulunan_cevap = ONBELLEK_VERILERI[en_yuksek_indeks]["cevap"]
        return True, bulunan_cevap, 0.0, round(en_yuksek_skor, 4)

    return False, "Büyük LLM modeline yönlendiriliyor...", 0.02, round(en_yuksek_skor, 4)
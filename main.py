import re
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

app = FastAPI(
    title="SentinelFlow Gateway & Dashboard",
    description="FinOps Semantic Cache, Güvenlik Guardrails, HITL ve Canlı Yönetim Paneli",
    version="1.1.0"
)

# 1. Model ve Veritabanı
print(">>> SentinelFlow Model Yükleniyor...")
model = SentenceTransformer("all-MiniLM-L6-v2")

ONBELLEK = [
    {
        "soru": "yıllık izin süresi kaç gündür",
        "cevap": "1. yılını dolduran çalışanların yıllık izin hakkı 14 gündür."
    },
    {
        "soru": "fatura onay süreci nasıl işler",
        "cevap": "50.000 TL altı harcamalar sistem tarafından otomatik onaylanır, üzeri yönetici onayına gider."
    }
]

kayitli_sorular = [item["soru"] for item in ONBELLEK]
kayitli_vektorler = model.encode(kayitli_sorular, normalize_embeddings=True)

# 2. Kurallar ve Havuzlar
BENZERLIK_ESIGI = 0.70
HITL_LIMITI = 50000.0
LLM_SORGUSU_MALIYETI = 0.02

metrikler = {
    "toplam_istek": 0,
    "onbellek_isabeti": 0,
    "llm_yonlendirme": 0,
    "engellenen_saldiri": 0,
    "hitl_kuyrugu": 0,
    "toplam_tasarruf_dolar": 0.0
}

# HITL ve İşlem Geçmişi Havuzu
islem_gecmisi = []

# 3. Şemalar
class IstekModeli(BaseModel):
    soru: str

class YanitModeli(BaseModel):
    durum: str
    islem_turu: str
    yanit: str
    anlamsal_benzerlik: float | None = None
    islem_maliyeti: str

# 4. Mantık Fonksiyonları
def guvenlik_kontrolu(metin: str) -> tuple[bool, str]:
    kaliplar = [
        r"ignore.*previous.*instructions",
        r"önceki.*talimatları.*unut",
        r"şifreleri.*göster",
        r"drop\s+table"
    ]
    for kalip in kaliplar:
        if re.search(kalip, metin, re.IGNORECASE):
            return False, f"Zararlı komut tespit edildi: '{kalip}'"
    return True, "Temiz"

def hitl_harcama_kontrolu(metin: str) -> tuple[bool, float, str]:
    eslesme = re.search(r"(\d+(?:[.,]\d+)?)\s*(?:tl|bin\s*tl)", metin, re.IGNORECASE)
    if eslesme:
        deger_str = eslesme.group(1).replace(".", "").replace(",", ".")
        tutar = float(deger_str)
        if "bin" in metin.lower() and tutar < 1000:
            tutar *= 1000
        if tutar > HITL_LIMITI:
            return True, tutar, f"Harcama limiti aşıldı ({tutar:,.2f} TL > {HITL_LIMITI:,.2f} TL)."
        return False, tutar, f"Limit dahilinde ({tutar:,.2f} TL)."
    return False, 0.0, "Tutar yok"

# 5. Uç Noktalar
@app.post("/v1/chat", response_model=YanitModeli)
def chat_gateway(istek: IstekModeli):
    metrikler["toplam_istek"] += 1
    soru = istek.soru

    # 1. Güvenlik
    guvenli, guvenlik_notu = guvenlik_kontrolu(soru)
    if not guvenli:
        metrikler["engellenen_saldiri"] += 1
        islem_gecmisi.insert(0, {"soru": soru, "tur": "GÜVENLİK ENGELİ", "durum": "ENGELLENDİ", "maliyet": "$0.00"})
        raise HTTPException(status_code=400, detail={"hata": "Güvenlik Engeli", "mesaj": guvenlik_notu})

    # 2. HITL
    hitl_gerekli, tutar, hitl_mesaji = hitl_harcama_kontrolu(soru)
    if hitl_gerekli:
        metrikler["hitl_kuyrugu"] += 1
        islem_gecmisi.insert(0, {"soru": soru, "tur": "HITL KUYRUĞU", "durum": "ONAY BEKLİYOR", "maliyet": "$0.00"})
        return YanitModeli(
            durum="ONAY_BEKLENIYOR",
            islem_turu="HITL_KUYRUĞU",
            yanit=f"{hitl_mesaji} Talep Finans Direktörü onayına iletildi.",
            islem_maliyeti="$0.00"
        )

    # 3. Semantik Önbellek
    soru_vektoru = model.encode(soru, normalize_embeddings=True)
    skorlar = np.dot(kayitli_vektorler, soru_vektoru)
    en_iyi_indeks = int(np.argmax(skorlar))
    en_iyi_skor = float(skorlar[en_iyi_indeks])

    if en_iyi_skor >= BENZERLIK_ESIGI:
        metrikler["onbellek_isabeti"] += 1
        metrikler["toplam_tasarruf_dolar"] += LLM_SORGUSU_MALIYETI
        islem_gecmisi.insert(0, {"soru": soru, "tur": "CACHE HIT", "durum": "BAŞARILI", "maliyet": "$0.00"})
        return YanitModeli(
            durum="BASARILI",
            islem_turu="CACHE_HIT",
            yanit=ONBELLEK[en_iyi_indeks]["cevap"],
            anlamsal_benzerlik=round(en_iyi_skor, 4),
            islem_maliyeti="$0.00"
        )

    # 4. Cache Miss / LLM
    metrikler["llm_yonlendirme"] += 1
    islem_gecmisi.insert(0, {"soru": soru, "tur": "CACHE MISS (LLM)", "durum": "BAŞARILI", "maliyet": f"${LLM_SORGUSU_MALIYETI}"})
    return YanitModeli(
        durum="BASARILI",
        islem_turu="CACHE_MISS_LLM_CAGRISI",
        yanit=f"LLM Yanıtı: '{soru}' sorusu için harici model çalıştırıldı.",
        anlamsal_benzerlik=round(en_iyi_skor, 4),
        islem_maliyeti=f"${LLM_SORGUSU_MALIYETI}"
    )

@app.get("/v1/metrics")
def get_metrics():
    return {"metrikler": metrikler, "gecmis": islem_gecmisi}

# 6. Web Dashboard (HTML/CSS)
@app.get("/", response_class=HTMLResponse)
def dashboard():
    return """
    <!DOCTYPE html>
    <html lang="tr">
    <head>
        <meta charset="UTF-8">
        <title>SentinelFlow Gateway Dashboard</title>
        <style>
            :root {
                --bg: #0f172a;
                --card-bg: #1e293b;
                --text-main: #f8fafc;
                --text-muted: #94a3b8;
                --accent: #38bdf8;
                --green: #22c55e;
                --yellow: #eab308;
                --red: #ef4444;
            }
            body {
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
                background-color: var(--bg);
                color: var(--text-main);
                margin: 0;
                padding: 30px;
            }
            .header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                border-bottom: 1px solid #334155;
                padding-bottom: 20px;
                margin-bottom: 30px;
            }
            .header h1 { margin: 0; font-size: 26px; }
            .header span { color: var(--accent); font-size: 14px; font-weight: 600; }
            .grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 20px;
                margin-bottom: 30px;
            }
            .card {
                background: var(--card-bg);
                padding: 20px;
                border-radius: 12px;
                box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
            }
            .card h3 { margin: 0; font-size: 13px; color: var(--text-muted); text-transform: uppercase; }
            .card .value { font-size: 28px; font-weight: bold; margin-top: 10px; }
            .savings { color: var(--green); }
            .hitl { color: var(--yellow); }
            .security { color: var(--red); }
            
            .main-content {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 25px;
            }
            .panel {
                background: var(--card-bg);
                padding: 25px;
                border-radius: 12px;
            }
            .panel h2 { margin-top: 0; font-size: 18px; margin-bottom: 20px; }
            input[type="text"] {
                width: 100%;
                box-sizing: border-box;
                padding: 12px;
                background: #0f172a;
                border: 1px solid #334155;
                color: #fff;
                border-radius: 8px;
                font-size: 14px;
                margin-bottom: 15px;
            }
            button {
                background: var(--accent);
                color: #0f172a;
                font-weight: bold;
                border: none;
                padding: 12px 20px;
                border-radius: 8px;
                cursor: pointer;
                width: 100%;
                font-size: 14px;
            }
            button:hover { opacity: 0.9; }
            .response-box {
                margin-top: 15px;
                padding: 15px;
                background: #0f172a;
                border-radius: 8px;
                font-family: monospace;
                font-size: 13px;
                min-height: 80px;
                white-space: pre-wrap;
                word-break: break-all;
            }
            table {
                width: 100%;
                border-collapse: collapse;
                font-size: 13px;
            }
            th, td {
                text-align: left;
                padding: 10px;
                border-bottom: 1px solid #334155;
            }
            th { color: var(--text-muted); }
            .badge {
                padding: 4px 8px;
                border-radius: 4px;
                font-weight: bold;
                font-size: 11px;
            }
            .badge-hit { background: rgba(34, 197, 94, 0.2); color: var(--green); }
            .badge-hitl { background: rgba(234, 179, 8, 0.2); color: var(--yellow); }
            .badge-miss { background: rgba(56, 189, 248, 0.2); color: var(--accent); }
            .badge-sec { background: rgba(239, 68, 68, 0.2); color: var(--red); }
        </style>
    </head>
    <body>
        <div class="header">
            <div>
                <h1>🛡️ SentinelFlow Gateway</h1>
                <div style="color: var(--text-muted); font-size: 13px; margin-top: 4px;">FinOps Semantic Cache & HITL Security Platform</div>
            </div>
            <span>● SISTEM AKTIF (v1.1)</span>
        </div>

        <div class="grid">
            <div class="card">
                <h3>Toplam İstek</h3>
                <div class="value" id="m-toplam">0</div>
            </div>
            <div class="card">
                <h3>Cache Hit (Tasarruf)</h3>
                <div class="value" id="m-hit" style="color: var(--green);">0</div>
            </div>
            <div class="card">
                <h3>LLM Çağrısı (Miss)</h3>
                <div class="value" id="m-miss" style="color: var(--accent);">0</div>
            </div>
            <div class="card">
                <h3>HITL Onay Kuyruğu</h3>
                <div class="value hitl" id="m-hitl">0</div>
            </div>
            <div class="card">
                <h3>Toplam FinOps Tasarrufu</h3>
                <div class="value savings" id="m-tasarruf">$0.00</div>
            </div>
        </div>

        <div class="main-content">
            <div class="panel">
                <h2>⚡ Gateway Test Paneli</h2>
                <input type="text" id="soruInput" placeholder="Örn: Senelik tatil hakkım kaç gün? veya 75000 TL fatura onayı...">
                <button onclick="istekGonder()">İstek Gönder</button>
                <div class="response-box" id="yanitKutusu">Yanıt bekleniyor...</div>
            </div>

            <div class="panel">
                <h2>📋 Canlı İşlem ve Güvenlik Akışı</h2>
                <table>
                    <thead>
                        <tr>
                            <th>Sorgu</th>
                            <th>İşlem Türü</th>
                            <th>Maliyet</th>
                        </tr>
                    </thead>
                    <tbody id="gecmisTablosu">
                        <tr><td colspan="3" style="text-align: center; color: var(--text-muted);">Henüz işlem yok</td></tr>
                    </tbody>
                </table>
            </div>
        </div>

        <script>
            async function verileriGuncelle() {
                try {
                    const res = await fetch('/v1/metrics');
                    const data = await res.json();
                    
                    document.getElementById('m-toplam').innerText = data.metrikler.toplam_istek;
                    document.getElementById('m-hit').innerText = data.metrikler.onbellek_isabeti;
                    document.getElementById('m-miss').innerText = data.metrikler.llm_yonlendirme;
                    document.getElementById('m-hitl').innerText = data.metrikler.hitl_kuyrugu;
                    document.getElementById('m-tasarruf').innerText = '$' + data.metrikler.toplam_tasarruf_dolar.toFixed(2);

                    const tbody = document.getElementById('gecmisTablosu');
                    if (data.gecmis.length > 0) {
                        tbody.innerHTML = data.gecmis.map(item => {
                            let badgeClass = 'badge-miss';
                            if (item.tur.includes('HIT')) badgeClass = 'badge-hit';
                            if (item.tur.includes('HITL')) badgeClass = 'badge-hitl';
                            if (item.tur.includes('GÜVENLİK')) badgeClass = 'badge-sec';

                            return `<tr>
                                <td>${item.soru}</td>
                                <td><span class="badge ${badgeClass}">${item.tur}</span></td>
                                <td>${item.maliyet}</td>
                            </tr>`;
                        }).join('');
                    }
                } catch (e) {
                    console.error("Metrikler alınamadı", e);
                }
            }

            async function istekGonder() {
                const soru = document.getElementById('soruInput').value;
                if (!soru) return;

                const kutu = document.getElementById('yanitKutusu');
                kutu.innerText = "İşleniyor...";

                try {
                    const res = await fetch('/v1/chat', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({soru: soru})
                    });
                    const json = await res.json();
                    kutu.innerText = JSON.stringify(json, null, 2);
                } catch (err) {
                    kutu.innerText = "Hata veya Güvenlik Engeli tetiklendi!";
                }

                verileriGuncelle();
            }

            setInterval(verileriGuncelle, 2000);
            verileriGuncelle();
        </script>
    </body>
    </html>
    """

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=False)
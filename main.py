import re
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

app = FastAPI(title="SentinelFlow Gateway", version="1.1.0")

model = SentenceTransformer("all-MiniLM-L6-v2")

CACHE_STORE = [
    {
        "query": "yıllık izin süresi kaç gündür",
        "response": "1. yılını dolduran çalışanların yıllık izin hakkı 14 gündür."
    },
    {
        "query": "fatura onay süreci nasıl işler",
        "response": "50.000 TL altı harcamalar sistem tarafından otomatik onaylanır, üzeri yönetici onayına gider."
    }
]

cached_queries = [item["query"] for item in CACHE_STORE]
cached_embeddings = model.encode(cached_queries, normalize_embeddings=True)

SIMILARITY_THRESHOLD = 0.70
HITL_SPENDING_LIMIT = 50000.0
LLM_QUERY_COST = 0.02

metrics = {
    "total_requests": 0,
    "cache_hits": 0,
    "llm_routes": 0,
    "blocked_attacks": 0,
    "hitl_queue": 0,
    "total_savings_usd": 0.0
}

audit_log = []

class QueryRequest(BaseModel):
    soru: str

class GatewayResponse(BaseModel):
    durum: str
    islem_turu: str
    yanit: str
    anlamsal_benzerlik: float | None = None
    islem_maliyeti: str

def validate_prompt_safety(text: str) -> tuple[bool, str]:
    blocked_patterns = [
        r"ignore.*previous.*instructions",
        r"önceki.*talimatları.*unut",
        r"şifreleri.*göster",
        r"drop\s+table"
    ]
    for pattern in blocked_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return False, f"Zararlı komut tespit edildi: '{pattern}'"
    return True, "OK"

def evaluate_hitl_threshold(text: str) -> tuple[bool, float, str]:
    match = re.search(r"(\d+(?:[.,]\d+)?)\s*(?:tl|bin\s*tl)", text, re.IGNORECASE)
    if match:
        amount = float(match.group(1).replace(".", "").replace(",", "."))
        if "bin" in text.lower() and amount < 1000:
            amount *= 1000
        if amount > HITL_SPENDING_LIMIT:
            return True, amount, f"Harcama limiti aşıldı ({amount:,.2f} TL > {HITL_SPENDING_LIMIT:,.2f} TL)."
        return False, amount, f"Limit dahilinde ({amount:,.2f} TL)."
    return False, 0.0, "N/A"

@app.post("/v1/chat", response_model=GatewayResponse)
def handle_chat_gateway(request: QueryRequest):
    metrics["total_requests"] += 1
    query = request.soru

    is_safe, safety_message = validate_prompt_safety(query)
    if not is_safe:
        metrics["blocked_attacks"] += 1
        audit_log.insert(0, {"query": query, "type": "GÜVENLİK ENGELİ", "cost": "$0.00"})
        raise HTTPException(status_code=400, detail={"hata": "Güvenlik Engeli", "mesaj": safety_message})

    requires_hitl, _, hitl_message = evaluate_hitl_threshold(query)
    if requires_hitl:
        metrics["hitl_queue"] += 1
        audit_log.insert(0, {"query": query, "type": "HITL KUYRUĞU", "cost": "$0.00"})
        return GatewayResponse(
            durum="ONAY_BEKLENIYOR",
            islem_turu="HITL_KUYRUĞU",
            yanit=f"{hitl_message} Talep Finans Direktörü onayına iletildi.",
            islem_maliyeti="$0.00"
        )

    query_embedding = model.encode(query, normalize_embeddings=True)
    similarity_scores = np.dot(cached_embeddings, query_embedding)
    top_index = int(np.argmax(similarity_scores))
    top_score = float(similarity_scores[top_index])

    if top_score >= SIMILARITY_THRESHOLD:
        metrics["cache_hits"] += 1
        metrics["total_savings_usd"] += LLM_QUERY_COST
        audit_log.insert(0, {"query": query, "type": "CACHE HIT", "cost": "$0.00"})
        return GatewayResponse(
            durum="BASARILI",
            islem_turu="CACHE_HIT",
            yanit=CACHE_STORE[top_index]["response"],
            anlamsal_benzerlik=round(top_score, 4),
            islem_maliyeti="$0.00"
        )

    metrics["llm_routes"] += 1
    audit_log.insert(0, {"query": query, "type": "CACHE MISS (LLM)", "cost": f"${LLM_QUERY_COST}"})
    return GatewayResponse(
        durum="BASARILI",
        islem_turu="CACHE_MISS_LLM_CAGRISI",
        yanit=f"LLM Yanıtı: '{query}' sorusu için harici model çalıştırıldı.",
        anlamsal_benzerlik=round(top_score, 4),
        islem_maliyeti=f"${LLM_QUERY_COST}"
    )

@app.get("/v1/metrics")
def get_metrics():
    return {"metrics": metrics, "audit_log": audit_log}

@app.get("/", response_class=HTMLResponse)
def render_dashboard():
    return """
    <!DOCTYPE html>
    <html lang="tr">
    <head>
        <meta charset="UTF-8">
        <title>SentinelFlow Gateway</title>
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
            .header h1 { margin: 0; font-size: 24px; }
            .header span { color: var(--accent); font-size: 13px; font-weight: 600; }
            .grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 20px;
                margin-bottom: 30px;
            }
            .card {
                background: var(--card-bg);
                padding: 20px;
                border-radius: 10px;
            }
            .card h3 { margin: 0; font-size: 12px; color: var(--text-muted); text-transform: uppercase; }
            .card .value { font-size: 26px; font-weight: 600; margin-top: 10px; }
            .main-content {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 25px;
            }
            .panel {
                background: var(--card-bg);
                padding: 25px;
                border-radius: 10px;
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
                font-weight: 600;
                border: none;
                padding: 12px 20px;
                border-radius: 8px;
                cursor: pointer;
                width: 100%;
                font-size: 14px;
            }
            .response-box {
                margin-top: 15px;
                padding: 15px;
                background: #0f172a;
                border-radius: 8px;
                font-family: monospace;
                font-size: 13px;
                min-height: 80px;
                white-space: pre-wrap;
            }
            table { width: 100%; border-collapse: collapse; font-size: 13px; }
            th, td { text-align: left; padding: 10px; border-bottom: 1px solid #334155; }
            th { color: var(--text-muted); }
            .badge { padding: 4px 8px; border-radius: 4px; font-weight: 600; font-size: 11px; }
            .badge-hit { background: rgba(34, 197, 94, 0.2); color: var(--green); }
            .badge-hitl { background: rgba(234, 179, 8, 0.2); color: var(--yellow); }
            .badge-miss { background: rgba(56, 189, 248, 0.2); color: var(--accent); }
            .badge-sec { background: rgba(239, 68, 68, 0.2); color: var(--red); }
        </style>
    </head>
    <body>
        <div class="header">
            <div>
                <h1>SentinelFlow Gateway</h1>
                <div style="color: var(--text-muted); font-size: 13px; margin-top: 4px;">FinOps Semantic Cache & Security Gateway</div>
            </div>
            <span>v1.1.0</span>
        </div>

        <div class="grid">
            <div class="card">
                <h3>Toplam İstek</h3>
                <div class="value" id="m-toplam">0</div>
            </div>
            <div class="card">
                <h3>Cache Hit</h3>
                <div class="value" id="m-hit" style="color: var(--green);">0</div>
            </div>
            <div class="card">
                <h3>LLM Çağrısı</h3>
                <div class="value" id="m-miss" style="color: var(--accent);">0</div>
            </div>
            <div class="card">
                <h3>HITL Kuyruğu</h3>
                <div class="value" id="m-hitl" style="color: var(--yellow);">0</div>
            </div>
            <div class="card">
                <h3>FinOps Tasarrufu</h3>
                <div class="value" id="m-tasarruf" style="color: var(--green);">$0.00</div>
            </div>
        </div>

        <div class="main-content">
            <div class="panel">
                <h2>Gateway Test Paneli</h2>
                <input type="text" id="soruInput" placeholder="Örn: Senelik izin süresi kaç gündür? veya 75000 TL fatura onayı">
                <button onclick="sendRequest()">İstek Gönder</button>
                <div class="response-box" id="yanitKutusu">Yanıt bekleniyor...</div>
            </div>

            <div class="panel">
                <h2>İşlem Geçmişi</h2>
                <table>
                    <thead>
                        <tr>
                            <th>Sorgu</th>
                            <th>Tür</th>
                            <th>Maliyet</th>
                        </tr>
                    </thead>
                    <tbody id="gecmisTablosu">
                        <tr><td colspan="3" style="text-align: center; color: var(--text-muted);">İşlem kaydı yok</td></tr>
                    </tbody>
                </table>
            </div>
        </div>

        <script>
            async function fetchMetrics() {
                try {
                    const res = await fetch('/v1/metrics');
                    const data = await res.json();
                    
                    document.getElementById('m-toplam').innerText = data.metrics.total_requests;
                    document.getElementById('m-hit').innerText = data.metrics.cache_hits;
                    document.getElementById('m-miss').innerText = data.metrics.llm_routes;
                    document.getElementById('m-hitl').innerText = data.metrics.hitl_queue;
                    document.getElementById('m-tasarruf').innerText = '$' + data.metrics.total_savings_usd.toFixed(2);

                    const tbody = document.getElementById('gecmisTablosu');
                    if (data.audit_log.length > 0) {
                        tbody.innerHTML = data.audit_log.map(item => {
                            let badgeClass = 'badge-miss';
                            if (item.type.includes('HIT')) badgeClass = 'badge-hit';
                            if (item.type.includes('HITL')) badgeClass = 'badge-hitl';
                            if (item.type.includes('GÜVENLİK')) badgeClass = 'badge-sec';

                            return `<tr>
                                <td>${item.query}</td>
                                <td><span class="badge ${badgeClass}">${item.type}</span></td>
                                <td>${item.cost}</td>
                            </tr>`;
                        }).join('');
                    }
                } catch (e) {
                    console.error("Metrikler okunamadı", e);
                }
            }

            async function sendRequest() {
                const query = document.getElementById('soruInput').value;
                if (!query) return;

                const box = document.getElementById('yanitKutusu');
                box.innerText = "İşleniyor...";

                try {
                    const res = await fetch('/v1/chat', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({soru: query})
                    });
                    const json = await res.json();
                    box.innerText = JSON.stringify(json, null, 2);
                } catch (err) {
                    box.innerText = "Hata veya güvenlik engeli!";
                }

                fetchMetrics();
            }

            setInterval(fetchMetrics, 2000);
            fetchMetrics();
        </script>
    </body>
    </html>
    """

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=False)ss
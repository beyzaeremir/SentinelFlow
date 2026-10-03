# 🛡️ SentinelFlow Gateway

> **FinOps Semantic Cache, Prompt Injection Guardrails & Human-in-the-Loop (HITL) Gateway**

SentinelFlow, büyük dil modelleri (LLM) ile çalışan kurumsal uygulamaların maliyetlerini düşürmek, prompt injection saldırılarını önlemek ve yüksek tutarlı finansal işlemleri insan onayına bağlamak için tasarlanmış bir yapay zekâ ağ geçididir.

## 🚀 Özellikler
- **FinOps Semantik Önbellek:** `all-MiniLM-L6-v2` modeli ve kosinüs benzerliği ile tekrar eden soruları %70+ benzerlikle önbellekten sıfır maliyetle yanıtlar.
- **Prompt Injection Kalkanı:** Zararlı prompt manipülasyonlarını ağ geçidi seviyesinde engeller.
- **Human-in-the-Loop (HITL):** 50.000 TL üzerindeki finansal talepleri durdurup yönetici onayına yönlendirir.
- **Canlı Web Paneli:** Gerçek zamanlı tasarruf ve güvenlik metriklerini gösteren yerleşik dashboard.

## 🛠️ Kurulum ve Çalıştırma
```bash
python -m venv venv
.\venv\Scripts\activate
pip install fastapi uvicorn sentence-transformers numpy pydantic
python main.py
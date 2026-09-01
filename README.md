# ✨ Astroloji RAG Asistanı

Burçlar, gezegenler, astrolojik evler ve gezegen açıları hakkında soru sorabileceğiniz, tamamen **yerel** çalışan bir RAG (Retrieval-Augmented Generation) uygulaması. Tüm işlemler (embedding ve chat modeli) [Foundry Local](https://github.com/microsoft/Foundry-Local) üzerinden bilgisayarınızda çalışır — herhangi bir veri dışarı gönderilmez.

## Mimari: 3 Ana Dosya

Proje, RAG hattının 3 temel aşamasına birebir karşılık gelen 3 ana Python dosyasından oluşur:

| # | Dosya | Görev |
|---|---|---|
| 1 | `veri_hazirlama.py` | **Veri hazırlama** — kaynak belgeyi parçalara ayırır, embedding üretir, SQLite'a yazar |
| 2 | `yapay_zeka_motoru.py` | **Yapay zeka motoru** — hybrid retrieval (semantic + keyword) ve cevap üretme mantığı |
| 3 | `kullanici_arayuzu.py` | **Kullanıcı arayüzü** — Streamlit tabanlı sohbet ekranı |

`yapay_zeka_motoru.py` hiçbir arayüz bilgisi içermez; `kullanici_arayuzu.py` ise hiçbir retrieval/generation mantığı içermez — sadece motoru çağırır. Bu ayrım, motorun ileride başka bir arayüzle (CLI, mobil, API) değiştirilebilmesini kolaylaştırır.

```
         Kullanıcı Sorusu
               │
               ▼
      Soru Embedding'e Çevrilir
               │
               ▼
┌─────────────────────────────┐
│   Hybrid Retrieval           │
│                               │
│  Semantic Similarity          │
│          +                    │
│  Keyword / Exact Matching     │
└──────────────┬────────────────┘
               │
               ▼
       En İyi Parça
               │
               ▼
      Güçlü Exact Match mi?
          /            \
        Evet            Hayır
         │                │
         ▼                ▼
Kaynak Metni Döndür   phi-3.5-mini
         │                │
         └────────┬───────┘
                  ▼
               Cevap
```

## Proje Yapısı

```
.
├── belgeler/
│   └── burclar.txt          # Kaynak astroloji verisi
│
├── veri_hazirlama.py         # (1/3) Veri hazırlama
├── yapay_zeka_motoru.py      # (2/3) Yapay zeka motoru
├── kullanici_arayuzu.py      # (3/3) Kullanıcı arayüzü
│
├── araclar/                  # Yardımcı / doğrulama scriptleri (ana teslim değil)
│   ├── db_kontrol.py          # veritabani.db içeriğini doğrular
│   ├── test_embedding.py      # Embedding modelinin çalıştığını test eder
│   └── test_chat_model.py     # Chat modelinin çalıştığını test eder
│
├── requirements.txt
├── .gitignore
└── veritabani.db              # (otomatik oluşturulur, repoya dahil değildir)
```

> `araclar/` klasöründeki scriptler projenin çekirdeği değildir; kurulumu doğrulamak ve hata ayıklamak için hazırlanmış yardımcı araçlardır.

## Nasıl Çalışır?

### 1. Veri Hazırlama (`veri_hazirlama.py`)

`belgeler/burclar.txt` dosyası okunur ve boş satırlara göre parçalara ayrılır. 800 karakteri geçen parçalar cümle sonlarına göre daha küçük parçalara bölünür (`MAX_CHUNK_CHARS`). Her parça, Foundry Local üzerinden çalışan `qwen3-embedding-0.6b` modeliyle vektöre dönüştürülüp `veritabani.db` (SQLite) içine JSON olarak kaydedilir.

### 2. Yapay Zeka Motoru (`yapay_zeka_motoru.py`)

**Retrieval:** Kullanıcı sorusu embedding'e çevrilir ve veritabanındaki tüm parçalarla **cosine similarity** hesaplanır. Buna ek olarak, soru içindeki burç/gezegen adı ve ev numarası (rakamla veya yazıyla — örn. "7. ev" / "yedinci ev") regex ile tespit edilip metinle eşleştirilerek bir **keyword skoru** hesaplanır:

```
final_score = semantic_score * 0.70 + keyword_score * 0.30
```

**Generation:** En iyi parçanın keyword skoru çok yüksekse (`exact_score >= 0.60`), LLM'e hiç sorulmadan kaynak metin doğrudan döndürülür — bu, halüsinasyon riskini sıfırlar ve daha hızlıdır. Aksi halde bulunan parçalar bağlam olarak `phi-3.5-mini` modeline verilir; model **sadece bağlamdaki bilgiye dayanarak** Türkçe cevap üretir. Bağlamda cevap yoksa *"Bu konuda elimde bilgi yok."* yanıtı döner.

Bu dosya, arayüzün çağıracağı tek yüksek seviye fonksiyonu da sağlar:

```python
cevap, baglam_parcalari = soru_sor(embed_client, chat_client, soru, k=1)
```

### 3. Kullanıcı Arayüzü (`kullanici_arayuzu.py`)

Streamlit ile kullanıcıdan soru alır, `yapay_zeka_motoru.py`'deki `soru_sor` fonksiyonunu çağırır ve sonucu (cevap + kullanılan bağlam parçalarının skorlarıyla birlikte) ekranda gösterir. Modeller `st.cache_resource` ile önbelleğe alınır, böylece her soruda yeniden yüklenmez.

## Kurulum

### 1. Gereksinimler

- Python 3.10+
- [Foundry Local](https://github.com/microsoft/Foundry-Local) kurulu olmalı

### 2. Bağımlılıkları yükleyin

```bash
pip install -r requirements.txt
```

### 3. Veritabanını oluşturun (veri hazırlama)

```bash
python veri_hazirlama.py
```

Kaynak belge güncellendiğinde bu script tekrar çalıştırılmalıdır; script eski kayıtları silip veritabanını yeniden oluşturur.

### 4. (Opsiyonel) Kurulumu doğrulayın

```bash
python araclar/test_embedding.py     # Embedding modelini test eder
python araclar/test_chat_model.py    # Chat modelini test eder
python araclar/db_kontrol.py         # Veritabanının doğru dolduğunu kontrol eder
```

### 5. (Opsiyonel) Yapay zeka motorunu terminalden test edin

Arayüze hiç girmeden, motoru doğrudan terminalden deneyebilirsiniz:

```bash
python yapay_zeka_motoru.py
```

Bir soru yazdığınızda `final_score`, `semantic_score` ve `keyword_score` ayrı ayrı gösterilir — retrieval kalitesini hızlıca değerlendirmek için kullanışlıdır.

### 6. Uygulamayı başlatın

```bash
python -m streamlit run kullanici_arayuzu.py
```

Tarayıcınızda açılan arayüzden sorularınızı yazabilirsiniz.

## Örnek Sorular

```
Akrep burcunun özellikleri nelerdir?
Venüs neyi temsil eder?
4. evde Mars ne anlama gelir?
7. evde Jüpiter ne anlama gelir?
Yedinci evde Jüpiter ne anlama gelir?
```

## Yapılandırma

İlgili dosyaların başındaki sabitler üzerinden ayarlanabilir:

| Değişken | Bulunduğu Dosya | Açıklama | Varsayılan |
|---|---|---|---|
| `EMBEDDING_MODEL` | `veri_hazirlama.py`, `yapay_zeka_motoru.py` | Embedding için kullanılan model | `qwen3-embedding-0.6b` |
| `CHAT_MODEL` | `yapay_zeka_motoru.py` | Cevap üretimi için kullanılan model | `phi-3.5-mini` |
| `MIN_SIMILARITY` | `yapay_zeka_motoru.py` | Minimum semantic similarity eşiği | `0.45` |
| `TOP_K` | `yapay_zeka_motoru.py` | Varsayılan bağlam parça sayısı | `5` |
| `EXACT_MATCH_ESIGI` | `yapay_zeka_motoru.py` | Üzerinde LLM'e gidilmeden kaynak metin döndürülen eşik | `0.60` |
| `MAX_CHUNK_CHARS` | `veri_hazirlama.py` | Bir parçanın maksimum karakter uzunluğu | `800` |

## Gizlilik ve Yerel Çalışma

Belge işleme, embedding üretimi, SQLite depolama, retrieval ve LLM çıkarımı dahil tüm işlemler bilgisayarınızda gerçekleşir. Modeller ilk indirildikten sonra normal kullanım için internet bağlantısı gerekmez.

## Kullanılan Teknolojiler

- Python
- Microsoft Foundry Local
- Phi-3.5 Mini
- Qwen3 Embedding 0.6B
- SQLite
- Streamlit
- Cosine Similarity
- Regular Expressions (Regex)

## Bilinen Sınırlamalar

- Bilgi tabanı görece küçük.
- Retrieval, ayrık bir vektör veritabanı yerine SQLite üzerinde tam tarama (full scan) ile yapılıyor; büyük veri setlerinde ölçeklenmesi gerekir.
- Keyword eşleştirme astroloji alanına özel olarak yazıldı, genel amaçlı değil.
- Üretilen cevabın kalitesi kullanılan yerel dil modeline bağlı.

## Proje Amacı

Bu proje, tamamen yerel çalışan bir Retrieval-Augmented Generation sisteminin uçtan uca pratik uygulamasını keşfetmek amacıyla geliştirilmiş eğitim amaçlı bir çalışmadır:

```
Belge → Chunking → Embedding → SQLite → Hybrid Retrieval → Bağlam Seçimi → Yerel LLM → Cevap
```

Proje özellikle **semantic similarity** ile **alana özel exact matching**'i birleştirerek retrieval güvenilirliğini artırmaya odaklanır.

## Lisans

Bu proje eğitim amaçlıdır.

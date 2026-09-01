"""
YARDIMCI SCRIPT: Embedding Modeli Testi
==========================================
Foundry Local'ın ve embedding modelinin (qwen3-embedding-0.6b) doğru
kurulup çalıştığını hızlıca doğrulamak için kullanılır. Proje kökünden
çalıştırılmalıdır:

    python araclar/test_embedding.py
"""

from foundry_local_sdk import Configuration, FoundryLocalManager


def main():
    config = Configuration(app_name="rag_test")
    FoundryLocalManager.initialize(config)
    manager = FoundryLocalManager.instance

    print("Embedding modeli indiriliyor/yükleniyor...")
    model = manager.catalog.get_model("qwen3-embedding-0.6b")
    model.download()
    model.load()
    print("Model hazır.")

    client = model.get_embedding_client()

    # Tek bir cümle için embedding
    response = client.generate_embedding("The quick brown fox jumps over the lazy dog")
    embedding = response.data[0].embedding
    print(f"Vektör boyutu: {len(embedding)}")
    print(f"İlk 5 değer: {embedding[:5]}")

    model.unload()


if __name__ == "__main__":
    main()

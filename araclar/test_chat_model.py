"""
YARDIMCI SCRIPT: Chat Modeli Testi
=====================================
Foundry Local'ın ve chat modelinin (phi-3.5-mini) doğru kurulup
çalıştığını hızlıca doğrulamak için kullanılır. Proje kökünden
çalıştırılmalıdır:

    python araclar/test_chat_model.py
"""

from foundry_local_sdk import Configuration, FoundryLocalManager


def main():
    config = Configuration(app_name="rag_test")
    FoundryLocalManager.initialize(config)
    manager = FoundryLocalManager.instance

    print("Modeller yükleniyor...")
    catalog = manager.catalog
    model = catalog.get_model("phi-3.5-mini")

    print("İndiriliyor (gerekirse)...")
    model.download()
    model.load()

    print("Model yüklendi, soru soruluyor...")
    client = model.get_chat_client()
    response = client.complete_chat([
        {"role": "user", "content": "Why is the sky blue? Answer in one sentence."}
    ])

    print("Cevap:", response.choices[0].message.content)
    model.unload()


if __name__ == "__main__":
    main()

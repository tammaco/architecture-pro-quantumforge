import os
import pickle
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
from yandex_cloud_ml_sdk import YCloudML
from examples_loader import get_few_shot_text, load_examples

load_dotenv()

YANDEX_CLOUD_FOLDER = os.getenv("YANDEX_CLOUD_FOLDER")
YANDEX_CLOUD_API_KEY = os.getenv("YANDEX_CLOUD_API_KEY")

INDEX_PATH = Path(__file__).parent.parent / "faiss_index.pkl"

TOP_K = 4
MAX_CONTEXT_LEN = 2000
TEMPERATURE = 0.3

SYSTEM_PROMPT = """Ты — помощник, который отвечает на вопросы. Отвечай коротко, по-русски, добавляй ссылку источники, если она есть в контексте.

При ответе на вопрос выполни следующие шаги:
1. Проанализируй вопрос: о чём именно спрашивают?
2. Посмотри на найденные фрагменты из базы знаний.
3. Если в фрагментах НЕТ информации, отвечай: "Я не знаю"
4. Если информация есть, сформулируй ответ своими словами, основываясь ТОЛЬКО на найденных фрагментах.
5. В конце ответа укажи источники в формате: [Источник: название_файла.txt]"""


class RAGBot:
    def __init__(self):
        self._load_index()
        self._init_sdk()
        self._load_examples()

    def _load_index(self):
        with open(INDEX_PATH, "rb") as f:
            index_data = pickle.load(f)
        self.faiss_index = index_data["faiss_index"]
        self.chunks = index_data["chunks"]

    def _init_sdk(self):
        self.sdk = YCloudML(folder_id=YANDEX_CLOUD_FOLDER, auth=YANDEX_CLOUD_API_KEY)
        self.query_embedder = self.sdk.models.text_embeddings('query')
        self.llm = self.sdk.models.completions('yandexgpt-lite')

    def _load_examples(self):
        self.examples_data = load_examples()
        self.examples_count = len(self.examples_data.get("examples", []))

    def _retrieve(self, query: str, top_k: int = TOP_K) -> list:
        result = self.query_embedder.run(query)
        query_emb = np.array(result.embedding, dtype=np.float32).reshape(1, -1)
        distances, indices = self.faiss_index.search(query_emb, top_k)

        retrieved = []
        for i, idx in enumerate(indices[0]):
            retrieved.append({
                "text": self.chunks[idx]["text"],
                "source": self.chunks[idx]["source_file"],
                "distance": float(distances[0][i])
            })
        return retrieved

    def _build_prompt(self, query: str, context_chunks: list) -> str:
        context_parts = []
        for i, chunk in enumerate(context_chunks):
            context_parts.append(f"[Фрагмент {i+1}] Источник: {chunk['source']}\n{chunk['text']}")
        context = "\n\n".join(context_parts)
        if len(context) > MAX_CONTEXT_LEN:
            context = context[:MAX_CONTEXT_LEN] + "..."
        few_shot_examples = get_few_shot_text()
        if not context_chunks or all(c['distance'] > 2.0 for c in context_chunks):
            return self._build_unknown_prompt(query)
        
        prompt = f"""{SYSTEM_PROMPT}

=== НАЙДЕННЫЕ ФРАГМЕНТЫ ИЗ БАЗЫ ЗНАНИЙ ===
{context}
=== КОНЕЦ ФРАГМЕНТОВ ===

=== ПРИМЕРЫ ПРАВИЛЬНЫХ ОТВЕТОВ (из JSON) ===
{few_shot_examples}

Вопрос: {query}

Теперь ответь, сначала написав шаги рассуждения (Шаг 1, Шаг 2...), затем ответ."""
        return prompt

    def _build_unknown_prompt(self, query: str) -> str:
        return f"""{SYSTEM_PROMPT}

=== ВНИМАНИЕ ===
В базе знаний НЕ НАЙДЕНО информации, относящейся к вопросу.

Вопрос: {query}

Шаг 1: Анализирую вопрос.
Шаг 2: Проверяю найденные фрагменты — информации нет.
Шаг 3: Не могу ответить.

Ответ: Я не знаю"""

    def ask(self, query: str) -> dict:
        retrieved = self._retrieve(query)
        prompt = self._build_prompt(query, retrieved)
        self.llm = self.llm.configure(temperature=TEMPERATURE)
        result = self.llm.run(prompt)
        sources = list(set([chunk['source'] for chunk in retrieved[:3]])) if retrieved else []

        return {
            "query": query,
            "answer": result.text,
            "sources": sources,
            "retrieved_count": len(retrieved)
        }

if __name__ == "__main__":
    import sys

    bot = RAGBot()

    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        response = bot.ask(query)
        print(f"\nВопрос: {response['query']}")
        print(f"Ответ: {response['answer']}")
        if response['sources']:
            print(f"\nИсточники: {', '.join(response['sources'])}")
    else:
        print("RAG-бот")
        print("Введите 'выход' для завершения")

        while True:
            try:
                user_input = input("Вы: ").strip()
                if user_input.lower() in ["выход", "quit", "exit"]:
                    print("Пока!")
                    break
                if not user_input:
                    continue

                response = bot.ask(user_input)
                print(f"\n {response['answer']}")
                if response['sources']:
                    print(f"\n Источники: {', '.join(response['sources'])}")
                print()
            except KeyboardInterrupt:
                print("\nПока!")
                break
            except Exception as e:
                print(f"\n Ошибка: {e}\n")
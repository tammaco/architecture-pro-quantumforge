import sys
import os
import json
import time
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent))
from rag_bot import RAGBot


load_dotenv()

GOLDEN_PATH = Path(__file__).parent.parent / "data" / "golden_questions.json"
LOG_PATH = Path(__file__).parent.parent / "logs" / "query_log.jsonl"

os.makedirs(LOG_PATH.parent, exist_ok=True)


def load_golden_questions() -> list:
    with open(GOLDEN_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("golden_questions", [])


def normalize_word(word: str) -> str:
    word = word.lower().strip()
    word = word.replace("-", " ")
    return word


def check_answer(expected: str, actual: str, chunks_found: int) -> tuple:
    expected_lower = expected.lower().strip()
    actual_lower = actual.lower().strip()
    
    if expected_lower == "я не знаю":
        if "я не знаю" in actual_lower:
            return 1, "Корректный отказ"
        else:
            return 0, "Должен был сказать 'я не знаю'"
    
    if chunks_found == 0:
        return 0, "Чанки не найдены"
    
    expected_words = set(normalize_word(w) for w in expected_lower.split())
    actual_words = set(normalize_word(w) for w in actual_lower.split())
    
    matched = len(expected_words & actual_words)
    
    if matched == len(expected_words):
        return 1, f"Найдены все слова ({matched}/{len(expected_words)})"
    elif matched > 0:
        return 1, f"Найдено {matched}/{len(expected_words)} слов"
    else:
        return 0, f"Не найдено ни одного слова"


def log_to_jsonl(entry: dict):
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def run_tests():
    bot = RAGBot()
    questions = load_golden_questions()
    
    results = {"total": len(questions), "success": 0, "failed": 0}
    
    for q in questions:
        start_time = time.time()
        response = bot.ask(q["question"])
        duration = time.time() - start_time
        
        chunks_found = response.get("retrieved_count", 0)
        actual_answer = response["answer"]
        answer_length = len(actual_answer)
        sources = response.get("sources", [])
        
        success, reason = check_answer(q["answer"], actual_answer, chunks_found)
        
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "question": q["question"],
            "expected_answer": q["answer"],
            "actual_answer": actual_answer,
            "chunks_found": chunks_found,
            "answer_length": answer_length,
            "success": success,
            "reason": reason,
            "sources": sources,
            "duration_sec": round(duration, 2)
        }
        log_to_jsonl(log_entry)
        
        if success:
            results["success"] += 1
        else:
            results["failed"] += 1
    
    return results


if __name__ == "__main__":
    results = run_tests()
    print(json.dumps(results, ensure_ascii=False))
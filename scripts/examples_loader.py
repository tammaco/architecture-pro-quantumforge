import json
from pathlib import Path


def load_examples(json_path: str = None) -> dict:
    if json_path is None:
        json_path = Path(__file__).parent.parent / "data" / "examples.json"
    
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    return data


def get_few_shot_text(json_path: str = None) -> str:
    data = load_examples(json_path)
    examples = data.get("examples", [])
    
    result = []
    for ex in examples:
        result.append(f"Пример {ex['id']}:")
        result.append(f"Вопрос: {ex['question']}")
        result.append(f"Найденные фрагменты:\n{ex['context']}")
        result.append(f"Ответ:\n{ex['answer']}")
        result.append("---")
    
    return "\n".join(result)
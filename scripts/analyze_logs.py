import json
from pathlib import Path
from datetime import datetime, timedelta

LOG_PATH = Path(__file__).parent.parent / "logs" / "query_log.jsonl"

def load_recent_entries(last_n: int = None, last_days: int = None) -> list:
    if not LOG_PATH.exists():
        print("Log not found")
        return []
    
    entries = []
    with open(LOG_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                entries.append(json.loads(line))
    
    if not entries:
        return []
    
    if last_n:
        return entries[-last_n:]
    
    if last_days:
        cutoff = datetime.now() - timedelta(days=last_days)
        return [e for e in entries if datetime.fromisoformat(e["timestamp"]) > cutoff]
    
    return entries

def analyze(last_n: int = None, last_days: int = None):
    entries = load_recent_entries(last_n, last_days)  
    entries = []
    with open(LOG_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                entries.append(json.loads(line))
    
    if not entries:
        print("Log is empty")
        return
    
    total = len(entries)
    success = sum(1 for e in entries if e["success"])
    avg_chunks = sum(e["chunks_found"] for e in entries) / total
    avg_duration = sum(e["duration_sec"] for e in entries) / total
    
    print(f"Totla queries: {total}")
    print(f"Success: {success}")
    print(f"Accuracy: {success/total*100:.1f}%")
    print(f"AVG chunks count: {avg_chunks:.1f}")
    print(f"AVG answer duration: {avg_duration:.2f} sec")
    
    print("Failed:")
    for e in entries:
        if not e["success"]:
            print(f"\nQuestion: {e['question']}")
            print(f"Reason: {e['reason']}")
            print(f"Expected: {e['expected_answer']}")
            print(f"Actual: {e['actual_answer'][:100]}...")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--last_n", type=int, help="Анализировать последние N запросов")
    parser.add_argument("--last_days", type=int, help="Анализировать за последние N дней")
    args = parser.parse_args()
    
    analyze(last_n=args.last_n, last_days=args.last_days)
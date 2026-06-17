# scripts/update_index.py
import os
import sys
import subprocess
import hashlib
import json
import logging
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

KB_PATH = Path(__file__).parent.parent / "knowledge_base"
STATE_PATH = Path(__file__).parent / "index_state.json"
LOG_PATH = Path(__file__).parent.parent / "logs" / "update.log"
VENV_PYTHON = Path(__file__).parent.parent / "venv" / "Scripts" / "python.exe"

os.makedirs(LOG_PATH.parent, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_PATH, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def get_file_hash(file_path: Path) -> str:
    with open(file_path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def load_state() -> dict:
    if STATE_PATH.exists():
        with open(STATE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"files": {}}


def save_state(state: dict):
    with open(STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def scan_changes() -> dict:
    state = load_state()
    current_files = {}
    
    for file_path in KB_PATH.glob("*.txt"):
        current_files[file_path.name] = get_file_hash(file_path)
    
    new_files = []
    modified_files = []
    deleted_files = []
    
    for filename, hash_val in current_files.items():
        if filename not in state["files"]:
            new_files.append(filename)
        elif state["files"][filename] != hash_val:
            modified_files.append(filename)
    
    for filename in state["files"]:
        if filename not in current_files:
            deleted_files.append(filename)
    
    return {
        "new": new_files,
        "modified": modified_files,
        "deleted": deleted_files,
        "has_changes": bool(new_files or modified_files or deleted_files)
    }


def run_script(script_name: str) -> bool:
    script_path = Path(__file__).parent / script_name
    
    python_path = VENV_PYTHON if VENV_PYTHON.exists() else Path(sys.executable)
    
    logger.info(f"launch: {script_name}")
    
    try:
        result = subprocess.run(
            [str(python_path), str(script_path)],
            capture_output=True,
            text=False,
            cwd=Path(__file__).parent
        )
        
        if result.stdout:
            print(result.stdout.decode('utf-8', errors='replace'))
        
        if result.returncode == 0:
            logger.info(f"{script_name} done")
            return True
        else:
            logger.error(f"{script_name} error ({result.returncode})")
            return False
            
    except Exception as e:
        logger.error(f"Error: {e}")
        return False


def update_index():
    start_time = datetime.now()
    logger.info(f"Start: {start_time}")
    
    changes = scan_changes()
    
    logger.info(f"NEW: {len(changes['new'])}")
    for f in changes['new']:
        logger.info(f"  + {f}")
    
    logger.info(f"MODIFIED: {len(changes['modified'])}")
    for f in changes['modified']:
        logger.info(f"  ~ {f}")
    
    logger.info(f"DELETED: {len(changes['deleted'])}")
    for f in changes['deleted']:
        logger.info(f"  - {f}")
    
    if not changes["has_changes"]:
        logger.info("NO CHANGES")
        logger.info(f"index updated at {start_time.strftime('%Y-%m-%d')}, 0 files added, 0 files modified, 0 files deleted, 0 errors")
        return True
        
    if not run_script("chunking.py"):
        logger.error("Error: chunking.py failed")
        return False
    
    if not run_script("embeddings.py"):
        logger.error("Error: embeddings.py failed")
        return False
    
    if not run_script("build_index.py"):
        logger.error("Error: build_index.py failed")
        return False
    
    new_state = {"files": {}}
    for file_path in KB_PATH.glob("*.txt"):
        new_state["files"][file_path.name] = get_file_hash(file_path)
    
    save_state(new_state)
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    logger.info(f"End, duration {duration:.2f} s")
    logger.info(f"index updated at {end_time.strftime('%Y-%m-%d')}, {len(changes['new'])} files added, {len(changes['modified'])} files modified, {len(changes['deleted'])} files deleted, 0 errors")
     
    return True


if __name__ == "__main__":
    success = update_index()
    sys.exit(0 if success else 1)
# Задание 1. Исследование моделей и инфраструктуры

## 1. Сравнение LLM-моделей

|LLM-модель|Качество ответов|Скорость работы|Стоимость владения и использования|Удобство и простота развёртывания|
|-|-|-|-|-|
|Hugging Face|Хорошее на малых моделях, высокое на больших (70B+)|Низкая на слабом железе, высокая на мощном|Высокая (покупка GPU, электричество, инженеры)|Низкая (требует настройки, CUDA, MLOps)|
|OpenAI|Максимальное, лучше держит длинный контекст и сложную логику|Высокая (80-150 ток/сек)|Низкая на старте, растет с нагрузкой|Высокая|
|YandexGPT|Среднее (уступает на сложных многошаговых задачах)|Средняя (достаточна для диалогов)|Низкая на старте, дешевле OpenAI|Высокая (API, SDK, российская поддержка)|

## 2. Сравнение моделей эмбеддингов

|LLM-модель|Скорость создания индекса|Качество поиска|Стоимость владения и использования|
|-|-|-|-|
|Локальные Sentence-Transformers|Высокая|Хорошее|Низкая (оборудование и электроэнергия)|
|Облачные OpenAI Embeddings|Низкая (зависит от сети)|Высокое|Высокая (растёт линейно с объёмом)|

## 3. Сравнение векторных баз ChromaDB и FAISS

|Векторных база|Скорость поиска и индексации|Сложность внедрения и поддержки|Удобство в работе|Стоимость владения (учёт инфраструктуры)|
|-|-|-|-|-|
|ChromaDB|Средняя|Низкая ("zero-config", установка через pip)|Высокое (Python API, встроенные метаданные, интеграция с LangChain)|Низкая (Open-source, затраты только на хостинг)|
|FAISS|Высокая|Высокая (нужен опыт для настройки индексов и оптимизации)|Среднее (требует ручного управления метаданными)|Высокая (500-1000 $/мес за GPU)|

## 4. Выбор рекомендуемой конфигурации сервера

|Критерий|Облачный LLM + локальная БД|Полностью локальный|Гибридный|
|-|-|-|-|
|Скорость индексации|(18K MDX + 3K Confluence)|Низкая (API-лимиты OpenAI)|Максимальная (локальный FAISS на GPU)|Высокая (локальный FAISS + только эмбеддинги на API)|
|Качество поиска|Очень высокое|Хорошее|Высокое (облачные эмбеддинги)|
|Качество генерации|Очень высокое|Среднее|Высокое (GPT на типовые запросы, локальная — для sensitive)|
|Стоимость владения|$18,000–24,000 (оплата токенов)|$12,000 (электричество + амортизация GPU)|$14,000–16,000|
|Сложность поддержки|Низкая|Высокая|Средняя|

Для варинта из задания лучше подойдёт Гибридный вариант.
Конфигурация сервера:
- CPU:	16 vCPU
- RAM:	128 GB 
- GPU:	1× NVIDIA L20 (48GB) или 2× RTX 4090
- Диск:	500 GB

# Задание 2. Подготовка базы знаний

В качестве предметной области был выбран мульфильм "ВАЛЛ-И", сюжет и персонажи были взяты с [фандом-сайта](https://pixar.fandom.com/).

Тексты были скопированы вручную, переведены и очищены от разметки.

Ключевые термины и персонажи были заменены в соотвествии со [словарём](https://github.com/tammaco/architecture-pro-quantumforge/terms_map.txt).

# Задание 3. Создание векторного индекса базы знаний

Основные шаги:
1. Разбиение на чанки

Для разбияения на чанки выбрана библиотека NLTKTextSplitter.

CHUNK_SIZE = 300
CHUNK_OVERLAP = 50

```
python chunking.py
```

Всего файлов: 31
Всего чанков: 115

Результат созранён в chunks_cache.pkl.

2. Получение эмбеддингов

Выбранная эмбеддинг-модель - YandexGPT text-search-doc.

```
python embeddings.py
```

Всего эмбеддингов: 115
Размерность: 256
Время векторизации: 27.4 с

Результат созранён в embeddings_cache.pkl.

3. Создание FAISS индекса

```
python build_index.py
```

Размерность: 256
Количество векторов: 115
Результат сохранён в faiss_index.pkl

4. Тестирование 

```
python query_test.py
```

# Задание 4. Реализация RAG-бота с техниками промптинга

Добавлены [примеры](https://github.com/tammaco/architecture-pro-quantumforge/data/examples.json) для few-shot промптинга.

Добавлен скрипт для [загрузки](https://github.com/tammaco/architecture-pro-quantumforge/scripts/examples_loader.py)  примеров.

Telegram-бот реализован, но для его работы требуется VPN/прокси из-за блокировки Telegram на территории РФ, поэтому остановилась на простом REPL-е.

# Задание 5. Запуск и демонстрация работы бота

Основные переменные вынесены в .env

```
YANDEX_CLOUD_FOLDER=[YANDEX_CLOUD_FOLDER]
YANDEX_CLOUD_API_KEY=[YANDEX_CLOUD_API_KEY]
TELEGRAM_BOT_TOKEN=[TELEGRAM_BOT_TOKEN]
CHUNK_SIZE=300
CHUNK_OVERLAP=50
CHUNK_SEPARATORS=["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""]
BATCH_SIZE=10
```

1. Активация виртуального окружения

```
venv\Scripts\activate
```

2. Установка зависимостей

```
pip install -r requirements.txt
```

3. Выполнение скриптов:

```
cd scripts
```

```
python chunking.py
```

```
python embeddings.py
```

```
python build_index.py
```

```
python rag_bot_secure.py
```

4. Вывод

Без фильтрации бот возвращал некорректный ответ по вредоносному контексту, фильтрацией бот отвечает "Я не знаю".
Скрины в [папке](https://github.com/tammaco/architecture-pro-quantumforge/screenshots) 

# Задание 6. Автоматическое ежедневное обновление базы знаний

Добавлен [файл](https://github.com/tammaco/architecture-pro-quantumforge/scripts/index_state.json) для хранения состояния индекса.

Добавлен [скрипт](https://github.com/tammaco/architecture-pro-quantumforge/scripts/update_index.py) для обновления индекса и логирования. 

Результаты обновления сохраняются в [файл](https://github.com/tammaco/architecture-pro-quantumforge/logs/update.log).

Пример [лога](https://github.com/tammaco/architecture-pro-quantumforge/screenshots/пример_лога.png) лога (удаление 1 файла).

[Диаграмма](https://github.com/tammaco/architecture-pro-quantumforge/diagram.drawio)

Вызов скрипта по обновлению базы знаний можно сделать в БД (вызов job) ежедневно в 6 утра:

```
EXEC msdb.dbo.sp_add_job
    @job_name = N'RAG_Index_Updater',
    @enabled = 1;

EXEC msdb.dbo.sp_add_jobstep
    @job_name = N'RAG_Index_Updater',
    @step_name = N'Run update_index.py',
    @subsystem = N'CmdExec',
    @command = N'update_index.py',
    @retry_attempts = 3,
    @retry_interval = 5;

EXEC msdb.dbo.sp_add_jobschedule
    @job_name = N'RAG_Index_Updater',
    @name = N'Daily_6AM',
    @freq_type = 4, 
    @freq_interval = 1,
    @active_start_time = 60000;  

EXEC msdb.dbo.sp_add_jobserver
    @job_name = N'RAG_Index_Updater';
```

# Задание 7. Аналитика покрытия и качества базы знаний

1. Добавлен [«золотой набор»](https://github.com/tammaco/architecture-pro-quantumforge/data/golden_questions.json) вопросов для проверки бота.
2. Добавлен скрит, который вызывает бот со списокм «золотого набора» и анализирует вхождения ответов:

```
cd scripts
python query_logger.py
```

3. Логи сохраняются в формате jsonl в [файле](https://github.com/tammaco/architecture-pro-quantumforge/logs/query_log.jsonl).

4. Добавлен скрипт анализа логов.
Запуск можно осуществить с параметрами для последний n-запросов или за последние N дней.

```
cd scripts
python analyze_logs.py --last_n 10 
```

Для теста был исправлен один верный ответ на неправильный, результат:

```
Totla queries: 11
Success: 10
Accuracy: 90.9%
AVG chunks count: 4.0
AVG answer duration: 1.78 sec
Failed:

Question: Какие люди были на "Очевидности"?
Reason: Не найдено ни одного слова
Expected: Вероника Антонио
Actual: Шаг 1: Вопрос о пассажирах на борту «Очевидности».
Шаг 2: В найденных фрагментах указано, что на «Оч...
```

Также на основе анализа в ответы были добавлены некоторые слова.

Было:

```
    {
      "question": "Что такое Пиплия?",
      "answer": "планета"
    }
``` 

Стало:

```
    {
      "question": "Что такое Пиплия?",
      "answer": "планета дом Вали"
    }
``` 

5. Добавлена [Диаграмма](https://github.com/tammaco/architecture-pro-quantumforge/sequence_diagram.puml) последовательности RAG-бота
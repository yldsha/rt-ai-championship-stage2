# Отчет по тестированию RAG-системы: Java Dataset

## 1. Общая информация об эксперименте
* **Дата проведения:** 22 июня 2026 года
* **Источник датасета:** [The Algorithms - Java](https://github.com/TheAlgorithms/Java)
* **Аппаратная платформа:** Apple Silicon (Device: `mps`)
* **Используемая модель эмбеддингов:** `BAAI/bge-m3`
* **Векторная база данных:** ChromaDB (Коллекция: `test_java_bge`)
* **Файл бэкапа чанков:** `backups/test_java_chunks.jsonl`

---

## 2. Метрики производительности (Performance Metrics)

### Индексация (Indexing)
| Метрика | Значение | Комментарий |
| :--- | :--- | :--- |
| **Всего извлечено чанков** | 10 384 | Успешно распарсено Tree-sitter |
| **Общее время индексации** | 2787.68 сек (~46.5 мин) | Генерация эмбеддингов + запись в ChromaDB |
| **Средняя скорость** | ~3.72 чанка / сек | На графическом чипе `mps` |
| **Ошибки парсинга** | 0 | Полный успех, пропусков файлов нет |

### Поиск (Retrieval)
| Метрика | Значение | Описание |
| :--- | :--- | :--- |
| **Latency (Задержка поиска)** | **0.1287 сек** (~128.7 мс) | Время от отправки запроса до получения Top-K чанков |

> **Вывод по производительности:** Скорость извлечения чанков (парсер AST) работает мгновенно (около 1 секунды на весь репозиторий). Основное время уходит на расчет эмбеддингов моделью `bge-m3`. При этом поисковая задержка в 128 мс полностью укладывается в рамки комфортного real-time взаимодействия.

---

## 3. Качественный анализ поиска (Qualitative Analysis)


### Тестовый запрос 1: "как реализовать авторизацию"

#### Найденные чанки ответов (Top-3):
1. **src/main/java/com/thealgorithms/others/PasswordGen.java:PasswordGen.PasswordGen:21**
*(Score: 0.5):*
```java
private PasswordGen() {
    }...

```

2. **src/test/java/com/thealgorithms/ciphers/a5/A5KeyStreamGeneratorTest.java:A5KeyStreamGeneratorTest.testInitialization:27**
*(Score: 0.4997):*

```java
@Test
    void testInitialization() {
        // Verify that the internal state is set up correctly
        assertNotNull(keyStreamGenerator, "KeyStreamGenerator should be initialized");
    }...

```
3. **src/main/java/com/thealgorithms/scheduling/FairShareScheduling.java:FairShareScheduling.addUser:40**
*(Score: 0.4993):*
```java
public void addUser(String userName) {
        users.putIfAbsent(userName, new User(userName));
    }...

```
### Тестовый запрос 2: "как отсортировать данные?"
#### Найденные чанки ответов (Top-3):
1. **src/main/java/com/thealgorithms/sorts/QuickSort.java:QuickSort:32**
*(Score: 0.5):*
```java
class QuickSort implements SortAlgorithm {

    @Override
    public <T extends Comparable<T>> T[] s...
```
2. **src/main/java/com/thealgorithms/sorts/StrandSort.java:StrandSort.strandSort:33**
*(Score: 0.4981):*
```java
private static <T extends Comparable<? super T>> List<T> strandSort(List<T> list) {
        if (list...

```
3. **src/main/java/com/thealgorithms/sorts/StrandSort.java:StrandSort.sort:19**
*(Score: 0.4966):*
```java
@Override
    public <T extends Comparable<T>> T[] sort(T[] array) {
        List<T> unsortedList = ...
```

---

## 4. Примеры структуры чанков (Chunk Samples)

### Пример 1: Чанк типа `class`

* **"chunk_id"**: "src/main/java/com/thealgorithms/audiofilters/EMAFilter.java:EMAFilter:15", 
* **"path"**: "src/main/java/com/thealgorithms/audiofilters/EMAFilter.java", 
* **"language"**: "java", 
* **"symbol"**: "EMAFilter", 
* **"chunk_type"**: "class", 
* **"start_line"**: 15, 
* **"end_line"**: 54, 
* **"code"**: 
```java
"public class EMAFilter {\n    private final double alpha;\n    private double emaValue;\n\n    /**\n     * Constructs an EMA filter with a given smoothing factor.\n     *\n     * @param alpha Smoothing factor (0 < alpha <= 1)\n     * @throws IllegalArgumentException if alpha is not in (0, 1]\n     */\n    public EMAFilter(double alpha) {\n        if (alpha <= 0 || alpha > 1) {\n            throw new IllegalArgumentException(\"Alpha must be between 0 and 1.\");\n        }\n        this.alpha = alpha;\n        this.emaValue = 0.0;\n    }\n\n    /**\n     * Applies the EMA filter to an audio signal array.\n     * EMA formula:\n     * EMA = alpha * currentSample + (1 - alpha) * previousEMA\n     *\n     * @param audioSignal Array of audio samples to process\n     * @return Array of processed (smoothed) samples\n     */\n    public double[] apply(double[] audioSignal) {\n        if (audioSignal == null || audioSignal.length == 0) {\n            return new double[0];\n        }\n        double[] emaSignal = new double[audioSignal.length];\n        emaValue = audioSignal[0];\n        emaSignal[0] = emaValue;\n        for (int i = 1; i < audioSignal.length; i++) {\n            emaValue = alpha * audioSignal[i] + (1 - alpha) * emaValue;\n            emaSignal[i] = emaValue;\n        }\n        return emaSignal;\n    }\n}", 
```
* **"docstring"**: 
```java
"/**\n * Exponential Moving Average (EMA) Filter for smoothing audio signals.\n *\n * <p>\n * This filter applies an exponential moving average to a sequence of audio\n * signal values, making it useful for smoothing out rapid fluctuations.\n * The smoothing factor (alpha) controls the degree of smoothing.\n *\n * <p>\n * Based on the definition from\n * <a href=\"https://en.wikipedia.org/wiki/Moving_average\">Wikipedia link</a>.\n */"
```
### Пример 1: Чанк типа `method`
* **"chunk_id"**: "src/main/java/com/thealgorithms/audiofilters/EMAFilter.java:EMAFilter.EMAFilter:25"
* **"path"**: "src/main/java/com/thealgorithms/audiofilters/EMAFilter.java", 
* **"language"**: "java", 
* **"symbol"**: "EMAFilter.EMAFilter", 
* **"chunk_type"**: "method", 
* **"start_line"**: 25, 
* **"end_line"**: 31, 
* **"code"**: 
```java
"public EMAFilter(double alpha) {\n        if (alpha <= 0 || alpha > 1) {\n            throw new IllegalArgumentException(\"Alpha must be between 0 and 1.\");\n        }\n        this.alpha = alpha;\n        this.emaValue = 0.0;\n    }", 
```
* **"docstring":** 
```java
"/**\n     * Constructs an EMA filter with a given smoothing factor.\n     *\n     * @param alpha Smoothing factor (0 < alpha <= 1)\n     * @throws IllegalArgumentException if alpha is not in (0, 1]\n     */"}
```

---
## 5. Логи (logs)

2026-06-22 01:25:34,250 [INFO] Extracting semantic chunks from repository /Users/dmitriy/rt-ai-championship-stage2/dataset/Java...
2026-06-22 01:25:35,489 [INFO] Successfully extracted 10384 semantic chunks
2026-06-22 01:25:35,536 [INFO] Backup saved to backups/test_java_chunks.jsonl
2026-06-22 01:25:35,543 [INFO] Using device: mps
2026-06-22 01:25:35,543 [INFO]
Loading model BAAI/bge-m3...
2026-06-22 01:25:36,198 [INFO] Loading SentenceTransformer model from BAAI/bge-m3.
2026-06-22 01:25:43,429 [INFO] Connecting to ChromaDB (folder RAG/chroma_db)...
2026-06-22 01:25:43,521 [INFO] FORCE MODE: Purging existing collection 'test_java_bge' for a clean re-indexing...
2026-06-22 01:25:43,588 [INFO] Found 0 chunks already in the database.
2026-06-22 01:25:43,588 [INFO]
Generating embeddings and saving to ChromaDB in batches...
2026-06-22 02:12:01,932 [INFO]
Done! Successfully processed and saved 10384 chunks.
2026-06-22 02:12:01,936 [INFO] Time: 2787.681 seconds


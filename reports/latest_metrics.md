# VidScribe Pipeline Performance & Token Audit Report
- **Execution Timestamp:** 2026-10-08 18:27:48
- **Target Video:** `Cricket World Cup 2011 India vs Aus Highlights ｜ Thrilling Match`
- **LLM Provider:** `gemini` (`gemini-3.1-flash-lite`)
- **Total Pipeline Latency:** `133.89s`
- **Total Estimated Cost:** `$0.00077 USD`
- **Overall Throughput:** `3.47 tokens/second`

## 1. Granular Phase Metrics

| Phase / Task | Calls | Input Tokens | Output Tokens | Total Tokens | Cost (USD) | Latency (s) | Throughput (TPS) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1a. PySceneDetect & Keyframes** | 1 | 0 | 0 | 0 | $0.00000 | 5.12s | - |
| **1b. Whisper Speech Recognition** | 1 | 0 | 0 | 0 | $0.00000 | 91.41s | - |
| **1c. ChromaDB Vector Upsert** | 1 | 0 | 0 | 0 | $0.00000 | 18.33s | - |
| **1d. Tier 2 Gemini Vision** | 0 | 0 | 0 | 0 | $0.00000 | 0.00s | - |
| **2a. ChromaDB Vector Retrieval** | 1 | 0 | 0 | 0 | $0.00000 | 0.20s | - |
| **2b. Multimodal Gemini RAG** | 1 | 5,405 | 73 | 5,478 | $0.00043 | 10.04s | 7.3 |
| **3. Full Video Summarization** | 1 | 3,050 | 391 | 3,441 | $0.00035 | 8.78s | 44.5 |
| **TOTAL** | **6** | **8,455** | **464** | **8,919** | **$0.00077** | **133.89s** | **3.5** |

## 2. Key Performance Indicators
- **Audio Real-Time Factor (RTF):** Faster than real-time playback if > 1.0x.
- **Vector Search Latency:** Typically sub-50ms via persistent ChromaDB.
- **Multimodal Image Tokens:** Gemini accounts for ~258 tokens per inspected keyframe.
- **Timestamp Grounding:** Citations verified in format `[MM:SS]`.

## 3. Query & Retrieval Grounding Evaluation
- **User Query:** *"Who won the match and what was the score?"*
- **Citations Cited:** `5` segment(s)
- **Grounded Timestamp Format:** `Verified`
- **Response Latency:** `10.048s`

### Generated Response
> India won the match against Australia by 5 wickets, as confirmed by the on-screen graphic at [04:17 - 04:31]. The commentary also notes this as a "magic win for India," marking their entry into the semi-finals of the World Cup [04:17 - 04:31].
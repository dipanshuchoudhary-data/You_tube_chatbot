﻿# YouTube Chatbot
# YouTube Chatbot

A lightweight chatbot that allows users to ask natural language questions about the content of any **YouTube video**, by simply entering the **video ID**. It fetches the transcript using YouTube APIs (or `youtube_transcript_api`) and answers questions based on the video's textual content using an LLM.

---

## Features

- Ask Questions About Videos**: Get answers without watching the full video.
- Uses Video ID**: No need to paste full URLs or download files.
- Transcript-Based**: Pulls the official transcript (if available) using YouTube APIs.
- LLM-Powered**: Uses a language model ( Hugging Face model) to answer context-aware questions.

---

## How It Works

1. User provides a **YouTube video ID**
2. The script fetches the **transcript** (if available)
3. Transcript is split into chunks and converted into **embeddings**
4. The user asks a question
5. A context-aware answer is generated using **retrieval-augmented generation (RAG)** with an LLM

---

## Tech Stack

- **Python**
- `youtube-transcript-api`
- **LangChain**
- **FAISS** for similarity search
- **LLM**:  Hugging Face Transformers

---

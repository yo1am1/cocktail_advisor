## Quick Start

# Cocktail Advisor Chat

A **Streamlit + FastAPI** application for cocktail recommendations using a **FAISS** vector store. This README explains how to build, set up, and start the application.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Project Structure](#project-structure)
3. [Installation and Setup](#installation-and-setup)
4. [Running the Backend (FastAPI)](#running-the-backend-fastapi)
5. [Running the Frontend (Streamlit)](#running-the-frontend-streamlit)
6. [Usage](#usage)

---

## Prerequisites

1. **Python 3.9+** (recommended)
2. A virtual environment tool (e.g. `venv`, `pipenv`, or `Poetry`)
3. **FAISS** (either `faiss-cpu` or `faiss-gpu` depending on your setup)
4. **Streamlit** and **FastAPI** installed (they should be in `requirements.txt`)
5. An **OpenAI** API key or other embeddings provider (modify code if needed)

## Project Structure

```
cocktail-advisor/
│
├─ app/
│ ├─ api_utils.py
│ ├─ embeddings.py
│ ├─ faiss_utils.py
│ ├─ prompts.py
│ └─ ...
│
├─ recommender_api.py # FastAPI server endpoints
├─ streamlit_app.py # Streamlit UI code
├─ requirements.txt
├─ README.md # This file
└─ data/
└─ faiss/
└─ vector_cocktails/... # Where your FAISS index is stored
```

- **`recommender_api.py`**: Main FastAPI file, defines endpoints for `/chat` and `/retrieve_recommendations`.
- **`streamlit_app.py`**: The Streamlit UI entry point (or your custom `py` with `st.chat_input` logic).
- **`app/faiss_utils.py`**: Contains logic to load the FAISS index and store or search user memory.
- **`app/embeddings.py`**: Helper for retrieving embeddings (e.g. from OpenAI).
- **`app/prompts.py`**: Contains the prompt templates used in the chain.
- **`requirements.txt`**: Dependencies for the entire project.

---

## Installation and Setup

### 1. Clone or Download

```bash
git clone https://github.com/yo1am1/cocktail_advisor.git
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 3. Install Dependencies

Install required packages (adjust if you use Poetry or pipenv):

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### **4. Ensure FAISS is Installed** :

If not already in `requirements.txt`, install either:

```
pip install faiss-cpu
# or
pip install faiss-gpu

```

## Running the Backend (FastAPI)

### **1. Inside** the project root, run:

```bash
python recommender_api.py
```

This starts the FastAPI server at `http://127.0.0.1:8000`.

### 2. Verify it’s running by visiting:

```
http://127.0.0.1:8000/docs
```

You should see the automatically generated FastAPI docs.

## Running the Frontend (Streamlit)

### 1. In a **new terminal** (while t	he FastAPI server is still running), **activate** your same virtual environment again.

### 2. Run your Streamlit code. For example:

```bash
streamlit run chatbot_index.py
```

### 3. By default, Streamlit runs at:

```html
http://localhost:8501
```

## Usage

* **Username is Required** :
  The Streamlit app will not proceed until you enter your name in the text input. This ensures each user is identified uniquely.
* **Asking for Cocktails** :
  Once you enter your name, you can type a question like:

> "Can you suggest a cocktail with rum and lemon?"
> The Streamlit app will:

1. Call `/retrieve_recommendations` to find relevant cocktails from FAISS.
2. Provide that context to `/chat` for the final LLM-based reply.
3. Display a structured answer in the chat interface.

* **Storing or Retrieving Memory** :
  The backend uses a chain of `tool` calls to decide if the user’s query should store new memory (e.g., “I love cocktails with mint”) or retrieve relevant results from user memory.
* **Category & Alcoholic** :
  The filtering logic in `recommender_api.py` can optionally filter by category or alcoholic type if your prompts produce those fields.

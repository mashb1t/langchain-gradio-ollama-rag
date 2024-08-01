# LangChain Document QA

This example provides an interface for asking questions to a PDF document.

## Setup

1. Ensure you have the `llama3.1` model installed:

```
ollama pull llama3.1
```

2. Install the Python Requirements.

```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```
python main.py
```

A prompt will appear, where questions may be asked:

```
Query: How many locations does WeWork have?
```

# Distributed RAG proof of concept

## Setup

### Remote Vector Database Servers

Prerequisites: Python, Docker, Git

In a terminal:
```sh
git clone https://github.com/chroma-core/chroma.git
cd chroma
docker compose up
```

The server API will be available on port 8000.

### Workstation/Client

Clone this repo:
```sh
git clone https://github.com/Kwaai-AI-Lab/kwaai-net.git
cd kwaai-net/distributed-rag
```

Create a python virtualenv:
```sh
python -m venv --prompt rag .venv
. .venv/bin/activate
pip install -r requirements.txt
```

Create a `nodes.txt` file with the IP and port of each remote server:
```
192.168.1.10:8000
192.168.1.11:8000
192.168.1.12:8000
```

Create a directory `chunks` for local storage of the original chunks.

Load the remote databases with embeddings from the documents:
```sh
python load.py
```
This will load documents from Wikipedia resulting from the query "Artificial Intelligence" and divide them evenly onto the remote databases.

Alternatively, documents can be loaded from a zim archive file.
```sh
python load.py wikipedia_en_all_maxi_2024-01.zim
```

Query the remote databases:
```sh
python query.py "Can models make moral judgements?"
```
This will retrieve the 5 nearest vectors from each remote database and sort the final list by distance.


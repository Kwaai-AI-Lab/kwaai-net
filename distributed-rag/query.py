import chromadb
from langchain_huggingface import HuggingFaceEmbeddings
import os
import sys


remote_nodes = []


def get_collections():
  clients = []
  for host, port in remote_nodes:
    client = chromadb.HttpClient(host=host, port=port)
    clients.append(client)

  collection_name = 'wikipedia-shard'
  collections = [c.get_or_create_collection(collection_name) for c in clients]
  return collections

def get_embedding_model():
  model_name = "sentence-transformers/all-mpnet-base-v2"
  model_kwargs = {'device': 'cpu'}
  encode_kwargs = {'normalize_embeddings': False}
  hf  = HuggingFaceEmbeddings(
          model_name=model_name,
          model_kwargs=model_kwargs,
          encode_kwargs=encode_kwargs
        )
  return hf

def make_query(query):
  results = []

  embedding_model = get_embedding_model()
  embedded_query = embedding_model.embed_query(query)

  for collection in get_collections():
    result = collection.query(query_embeddings=[embedded_query], n_results=5, include=['distances',])
    results.extend(zip(result['distances'][0],result['ids'][0]))

  results.sort()
  return results

def init():
  with open('nodes.txt') as fh:
    for line in fh:
      host, port = line.strip().split(':')
      remote_nodes.append((host, port))

def main(query):
  init()
  results = make_query(query)
  for _, id in results:
    with open(os.path.join('chunks', id)) as fh:
      print(fh.read())
    print('\n----------\n\n')

if __name__ == '__main__':
  if len(sys.argv) != 2:
    print('Usage: python query.py "Put your query here"')
    sys.exit(1)
  main(sys.argv[1])

import chromadb
from langchain_community.document_loaders import WikipediaLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, SpacyTextSplitter, NLTKTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
import os
from uuid import uuid4


remote_nodes = []


def save_chunk(chunk, id):
  with open(os.path.join('chunks', id), 'w') as fh:
    fh.write(chunk)

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

def get_embeddings():
  loader = WikipediaLoader(query='Artificial Intelligence', load_max_docs=10)
  documents = loader.load()
  text_splitter = SpacyTextSplitter(chunk_size=1000, chunk_overlap=100)

  embedding_model = get_embedding_model()

  for document in documents:
    chunks = text_splitter.split_text(document.page_content)
    for chunk in chunks:
      embedding = embedding_model.embed_query(chunk)
      id = uuid4().hex
      save_chunk(chunk, id)
      yield (embedding, id)

def build_database():
  clients = []
  for host, port in remote_nodes:
    client = chromadb.HttpClient(host=host, port=port)
    clients.append(client)

  collection_name = 'wikipedia-shard'
  for client in clients:
    try:
      client.delete_collection(collection_name)
    except:
      pass
  collections = [c.create_collection(collection_name) for c in clients]
  shard_count = len(clients)

  index = 0
  for embedding, id in get_embeddings():
    collections[index].add(embeddings=[embedding], ids=[id])
    print('Adding embedding %s to server %d' % (id, index))
    index = (index + 1) % shard_count

def init():
  with open('nodes.txt') as fh:
    for line in fh:
      host, port = line.strip().split(':')
      remote_nodes.append((host, port))

def main():
  init()
  build_database()

if __name__ == '__main__':
  main()


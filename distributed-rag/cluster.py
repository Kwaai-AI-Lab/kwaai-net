import chromadb

remote_nodes = []

def display_cluster_info():
  clients = []
  for host, port in remote_nodes:
    client = chromadb.HttpClient(host=host, port=port)
    clients.append(client)

  for client in clients:
    settings = client.get_settings()
    print('Server %s:%d' % (settings['chroma_server_host'], settings['chroma_server_http_port']))
    for collection in client.list_collections():
      print('- %s: %d' % (collection.name, collection.count()))

def init():
  with open('nodes.txt') as fh:
    for line in fh:
      host, port = line.strip().split(':')
      remote_nodes.append((host, port))

def main():
  init()
  display_cluster_info()

if __name__ == '__main__':
  main()

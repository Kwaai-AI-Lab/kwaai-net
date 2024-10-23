import chromadb

remote_nodes = []

def get_cluster_info():
  clients = []
  for host, port in remote_nodes:
    client = chromadb.HttpClient(host=host, port=port)
    clients.append(client)

  cluster = {'servers':[]}
  for client in clients:
    settings = client.get_settings()
    server = {
      'host':settings['chroma_server_host'],
      'port':settings['chroma_server_http_port'],
      'collections': {}
    }
    for collection in client.list_collections():
      server['collections'][collection.name] = collection.count()
    cluster['servers'].append(server)

  return cluster

def display_cluster_info():
  cluster = get_cluster_info()
  for server in cluster['servers']:
    print('Server %s:%d' % (server['host'], server['port']))
    for name, count in server['collections'].items():
      print('- %s: %d' % (name, count))

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

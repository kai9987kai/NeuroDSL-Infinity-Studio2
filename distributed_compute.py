import socket
import threading
import json
import pickle
import time
from typing import List, Dict

class ComputeServer:
    """Central server for the distributed training cluster."""
    def __init__(self, host='0.0.0.0', port=9999):
        self.host = host
        self.port = port
        self.nodes = {} # {addr: {'status': 'idle', 'last_seen': time}}
        self.running = False
        self.server_thread = None

    def start(self):
        self.running = True
        self.server_thread = threading.Thread(target=self._listen, daemon=True)
        self.server_thread.start()

    def _listen(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind((self.host, self.port))
            s.listen()
            while self.running:
                conn, addr = s.accept()
                threading.Thread(target=self._handle_node, args=(conn, addr), daemon=True).start()

    def _handle_node(self, conn, addr):
        with conn:
            data = conn.recv(4096)
            if data:
                msg = json.loads(data.decode())
                if msg.get('type') == 'register':
                    self.nodes[addr] = {'status': 'idle', 'last_seen': time.time(), 'spec': msg.get('spec')}
                elif msg.get('type') == 'update':
                    if addr in self.nodes:
                        self.nodes[addr]['status'] = msg.get('status')
                        self.nodes[addr]['last_seen'] = time.time()

    def get_active_nodes(self) -> List[Dict]:
        return [{"addr": addr, "info": info} for addr, info in self.nodes.items() if time.time() - info['last_seen'] < 60]

class ComputeClient:
    """Remote node client that connects to the studio server."""
    def __init__(self, server_host, server_port=9999):
        self.server_host = server_host
        self.server_port = server_port

    def register(self, specs: dict):
        self._send({'type': 'register', 'spec': specs})

    def send_status(self, status: str):
        self._send({'type': 'update', 'status': status})

    def _send(self, msg: dict):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.connect((self.server_host, self.server_port))
                s.sendall(json.dumps(msg).encode())
        except Exception as e:
            print(f"Connection Error: {e}")

if __name__ == "__main__":
    # Example client usage
    client = ComputeClient('localhost')
    client.register({'gpu': 'RTX 4090', 'ram': '64GB'})
    while True:
        client.send_status('idle')
        time.sleep(10)

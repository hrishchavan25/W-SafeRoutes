import socket
import sys

def get_lan_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # doesn't need to be reachable
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]
    except Exception:
        ip = None
    finally:
        s.close()
    if not ip:
        # fallback: inspect interfaces
        try:
            import netifaces
            for iface in netifaces.interfaces():
                addrs = netifaces.ifaddresses(iface).get(netifaces.AF_INET, [])
                for a in addrs:
                    addr = a.get('addr')
                    if addr and not addr.startswith('127.') and not addr.startswith('169.'):
                        return addr
        except Exception:
            pass
    return ip or '127.0.0.1'

if __name__ == '__main__':
    print(get_lan_ip())

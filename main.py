from config import load_config
from nosos.nosos import NOSOS
import sys
import signal

def main():
    config = load_config()
    nosos = NOSOS(config)

    # Обработка сигналов завершения
    signal.signal(signal.SIGINT, lambda s, f: nosos.shutdown())
    signal.signal(signal.SIGTERM, lambda s, f: nosos.shutdown())

    def signal_handler(sig, frame):
        nosos.shutdown()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        nosos.run()
    except KeyboardInterrupt:
        nosos.shutdown()
    finally:
        nosos.shutdown()


if __name__ == "__main__":
    main()
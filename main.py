from config import load_config
from efis.nosos import NOSOS

def main():
    config = load_config()
    nosos = NOSOS(config)
    nosos.run()

if __name__ == "__main__":
    main()

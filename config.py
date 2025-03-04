import os
import sys
import yaml

if getattr(sys, 'frozen', False):
    BASE_DIR = sys._MEIPASS
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CONFIG_PATH = os.path.join(BASE_DIR, 'config.yaml')

def load_config():
    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    # При необходимости можно провести предобработку конфига
    for zone in config.get('alerts', {}).get('zones', []):
        zone['alert_level'] = zone.get('alert_level', 'INFO')
    if 'limits' in config.get('alerts', {}):
        config['alerts']['limits']['alert_level'] = config['alerts']['limits'].get('alert_level', 'WARNING')
    return config

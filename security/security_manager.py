from datetime import datetime

class SecurityManager:
    def __init__(self, config):
        self.config = config['security']
        self.log_file = self.config['log_file']

    def is_admin(self, user_id: str) -> bool:
        return str(user_id) in self.config['admins']

    def log_event(self, event_type: str, user_id: str, command: str):
        entry = f"{datetime.now().isoformat()} | {event_type} | User: {user_id} | Command: {command}\n"
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(entry)

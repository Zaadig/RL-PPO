import json
import os

class Logger:
    def __init__(self, log_file):
        self.log_file = log_file
        self.data = {}

    def log(self, key, value):
        if key not in self.data:
            self.data[key] = []
        self.data[key].append(value)

    def save(self):
        with open(self.log_file, 'w') as f:
            json.dump(self.data, f, indent=4)

    def load(self):
        if os.path.exists(self.log_file):
            with open(self.log_file, 'r') as f:
                self.data = json.load(f)

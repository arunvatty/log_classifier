import json

important_keywords = ['error', 'fail', 'exception', 'critical', 'crash', 'anr', 'not responding', 'fatal']

def auto_label_log(log_line):
    log_lower = log_line.lower()
    return "Important" if any(keyword in log_lower for keyword in important_keywords) else "Not Important"

with open("data/logs.log", "r") as f:
    log_lines = f.readlines()

labeled_logs = [{"log": line.strip(), "label": auto_label_log(line)} for line in log_lines]

with open("data/labeled_logs.json", "w") as f:
    json.dump(labeled_logs, f, indent=2)

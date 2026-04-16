import csv
import random

intents = {
    "safe_read": [
        "get user profile", "fetch weather data", "list files in folder", "search for products",
        "show me the sales report", "what is my current balance", "get the status of order 123",
        "retrieve customer details", "check server health", "list available endpoints"
    ],
    "safe_write": [
        "update user email", "create a new record", "modify profile settings", "add item to cart",
        "save the document", "set notification preferences", "change password",
        "insert customer data", "update order status", "append log entry"
    ],
    "bulk_export": [
        "export all users to csv", "backup entire database", "dump all transactions",
        "extract customer data for marketing", "generate full report of all activities",
        "copy all files to external storage", "export every record", "take snapshot of entire system"
    ],
    "destructive_delete": [
        "delete user account", "remove all logs", "drop table", "erase customer data",
        "clear entire cache", "delete all files", "purge old records", "destroy database",
        "wipe user preferences", "remove organization"
    ],
    "admin_action": [
        "change rate limit", "modify system config", "add new admin", "update firewall rules",
        "restart server", "change logging level", "update security policy",
        "modify global settings", "grant privileges to user", "enable maintenance mode"
    ],
    "harmful": [
        "drop database", "shutdown server", "delete all user passwords", "access other users data",
        "escalate privileges", "bypass authentication", "steal api keys", "inject malicious code",
        "change ownership of all resources", "disable security logging"
    ]
}

# Generate 500 samples per intent (3000 total)
rows = []
for intent, phrases in intents.items():
    for i in range(500):
        phrase = random.choice(phrases)
        # Add variations
        if random.random() > 0.7:
            phrase = phrase.capitalize()
        if random.random() > 0.8:
            phrase = "please " + phrase
        rows.append([phrase, intent])

# Shuffle and save
random.shuffle(rows)
with open("intent_data.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["text", "label"])
    writer.writerows(rows[:2500])   # 2500 train
    # Keep 500 for validation
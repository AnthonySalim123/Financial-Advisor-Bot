import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.database import get_database

print("Initializing database...")
db = get_database()
print("✅ Database initialized at data/stockbot.db")

# Create default user
user_id = db.create_user("demo_user", "demo@example.com", "demo123")
if user_id > 0:
    print(f"✅ Created demo user with ID: {user_id}")
else:
    print("ℹ️ Demo user already exists")

print("Database setup complete!")

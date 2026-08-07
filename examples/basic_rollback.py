"""
examples/basic_rollback.py

Run this file directly to SEE the undo engine work:
    python examples/basic_rollback.py

We simulate 3 "real world" systems using plain Python dictionaries/lists,
so you don't need any real database, email, or API keys to see it work.
"""

import sys
import os

# Allow running this script directly without installing the package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agent_acid.core import ReversibleTool, TransactionContext, AgentTransactionEngine


# ---------------------------------------------------------------
# STEP 1: Fake "real world" systems (stand-ins for a DB, email, etc.)
# ---------------------------------------------------------------
fake_database = {}
fake_sent_emails = []
fake_inventory = {"widgets": 100}


# ---------------------------------------------------------------
# STEP 2: Define tools, each with an execute() and a compensate()
# ---------------------------------------------------------------

# Tool A: create a user record in the "database"
def create_user_execute(kwargs):
    user_id = kwargs["user_id"]
    fake_database[user_id] = {"name": kwargs["name"], "email": kwargs["email"]}
    return {"user_id": user_id}

def create_user_compensate(kwargs, result):
    fake_database.pop(result["user_id"], None)

create_user_tool = ReversibleTool(
    name="create_user",
    description="Creates a user record in the database",
    execute=create_user_execute,
    compensate=create_user_compensate,
)


# Tool B: send a welcome email
def send_email_execute(kwargs):
    email_record = {"to": kwargs["email"], "subject": "Welcome!"}
    fake_sent_emails.append(email_record)
    return email_record

def send_email_compensate(kwargs, result):
    # "Undo" an email by logging a retraction (you can't unsend an email,
    # but you CAN record that it should be ignored/retracted)
    if result in fake_sent_emails:
        fake_sent_emails.remove(result)
        fake_sent_emails.append({"to": result["to"], "subject": "[RETRACTED] Welcome!"})

send_email_tool = ReversibleTool(
    name="send_welcome_email",
    description="Sends a welcome email to the new user",
    execute=send_email_execute,
    compensate=send_email_compensate,
)


# Tool C: reduce inventory count (this one will FAIL on purpose)
def update_inventory_execute(kwargs):
    item = kwargs["item"]
    qty = kwargs["qty"]
    if fake_inventory[item] < qty:
        # Simulate a real failure: not enough stock
        raise ValueError(f"Not enough stock for '{item}' (requested {qty}, have {fake_inventory[item]})")
    fake_inventory[item] -= qty
    return {"item": item, "new_qty": fake_inventory[item], "deducted": qty}

def update_inventory_compensate(kwargs, result):
    fake_inventory[result["item"]] += result["deducted"]

update_inventory_tool = ReversibleTool(
    name="update_inventory",
    description="Deducts stock from inventory",
    execute=update_inventory_execute,
    compensate=update_inventory_compensate,
)


# ---------------------------------------------------------------
# STEP 3: Run a SUCCESSFUL transaction first
# ---------------------------------------------------------------
engine = AgentTransactionEngine()

print("\n########## SCENARIO 1: Everything succeeds ##########")
ctx1 = TransactionContext()
success_plan = [
    (create_user_tool, {"user_id": "u1", "name": "Alice", "email": "alice@test.com"}),
    (send_email_tool, {"email": "alice@test.com"}),
    (update_inventory_tool, {"item": "widgets", "qty": 5}),
]
engine.execute_plan(ctx1, success_plan)

print("Database state:", fake_database)
print("Emails sent:", fake_sent_emails)
print("Inventory:", fake_inventory)


# ---------------------------------------------------------------
# STEP 4: Run a FAILING transaction (not enough stock on step 3)
# ---------------------------------------------------------------
print("\n########## SCENARIO 2: Step 3 fails -> auto rollback ##########")
ctx2 = TransactionContext()
failing_plan = [
    (create_user_tool, {"user_id": "u2", "name": "Bob", "email": "bob@test.com"}),
    (send_email_tool, {"email": "bob@test.com"}),
    (update_inventory_tool, {"item": "widgets", "qty": 9999}),  # will fail: not enough stock
]
engine.execute_plan(ctx2, failing_plan)

print("Database state AFTER rollback (Bob should be GONE):", fake_database)
print("Emails AFTER rollback (Bob's email should be retracted):", fake_sent_emails)
print("Inventory AFTER rollback (should be unchanged):", fake_inventory)

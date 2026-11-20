import sys, os
sys.path.append(r"C:/Users/Admin/OneDrive/Desktop/trip-planner-agent/trip-planner-agent")

try:
    from fastapi.testclient import TestClient
    from trip_planner.app import app
except Exception as e:
    print("Import error:", e)
    raise

client = TestClient(app)
resp = client.post("/ask", json={"question": "Hello"})
print("Status:", resp.status_code)
print("Body:", resp.json())

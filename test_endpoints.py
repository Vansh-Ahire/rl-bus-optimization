import requests
import json

def test():
    try:
        r = requests.get("http://localhost:7860/tasks")
        print(f"Tasks: {json.dumps(r.json(), indent=2)[:500]}...")
        
        r = requests.post("http://localhost:7860/grader", json={"task_id": "task_1"})
        print(f"Grader task_1: {r.json()}")
        
        r = requests.post("http://localhost:7860/grader", json={"task_id": "task1"})
        print(f"Grader task1: {r.json()}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test()

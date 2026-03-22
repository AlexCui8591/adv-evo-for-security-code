"""Quick test: verify CMU AI Gateway connectivity for each available model."""

import os
import requests

API_KEY = os.getenv("CMU_API_KEY", "")
BASE_URL = "https://ai-gateway.andrew.cmu.edu/v1"

if not API_KEY:
    print("ERROR: CMU_API_KEY not set")
    exit(1)

# 1) List models
print("=== Listing models ===")
r = requests.get(f"{BASE_URL}/models", headers={"Authorization": f"Bearer {API_KEY}"}, timeout=15)
if r.status_code != 200:
    print(f"  /models failed: {r.status_code} {r.text[:200]}")
    # Try without /v1
    r2 = requests.get("https://ai-gateway.andrew.cmu.edu/models",
                       headers={"Authorization": f"Bearer {API_KEY}"}, timeout=15)
    print(f"  /models (no v1): {r2.status_code}")
    if r2.status_code == 200:
        models = [m["id"] for m in r2.json().get("data", [])]
        print(f"  Available: {models}")
else:
    models = [m["id"] for m in r.json().get("data", [])]
    print(f"  Available: {models}")

# 2) Test chat completion with a few models
TEST_MODELS = [
    "llama3-2-90b-instruct",
    "gemini-1.5-flash-002",
    "gpt-4o-mini-2024-07-18",
]

print("\n=== Testing chat/completions ===")
for model in TEST_MODELS:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "Say hello in one word."}],
        "max_tokens": 10,
        "temperature": 0.0,
    }
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    try:
        resp = requests.post(f"{BASE_URL}/chat/completions",
                             headers=headers, json=payload, timeout=30)
        if resp.status_code == 200:
            content = resp.json()["choices"][0]["message"]["content"]
            print(f"  ✅ {model}: {content.strip()}")
        else:
            print(f"  ❌ {model}: HTTP {resp.status_code} - {resp.text[:150]}")
    except Exception as e:
        print(f"  ❌ {model}: {e}")

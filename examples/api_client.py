import json
import urllib.request

url = "http://127.0.0.1:8000/generate"

payload = {
    "prompt": (
        "System: You are a helpful coding assistant for unit testing.\n"
        "User: My tests are flaky in CI. Give me a quick plan.\n"
        "Assistant:"
    ),
    "max_new_tokens": 80,
    "do_sample": True,
    "temperature": 0.9,
    "top_k": 40,
}

req = urllib.request.Request(
    url,
    data=json.dumps(payload).encode("utf-8"),
    headers={"Content-Type": "application/json"},
    method="POST",
)

with urllib.request.urlopen(req) as resp:
    body = json.loads(resp.read().decode("utf-8"))
    print("PROMPT:")
    print(body["prompt"])
    print("\nCOMPLETION:")
    print(body["completion"])

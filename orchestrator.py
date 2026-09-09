"""
Nexvon local orchestrator (CLI).

Routes free-text input across local Ollama models:
  - tool / structured logic  → qwen2.5-coder
  - general chat             → llama3.2
  - vision (image path)      → qwen2.5vl

For the HTTP API used by NexvonUI, see app/main.py.
"""

import json
import re

import ollama

# Explicit client pointed at the local Ollama server.
client = ollama.Client(host="http://127.0.0.1:11434")

MODEL_CODER = "qwen2.5-coder:1.5b"
MODEL_CHAT = "llama3.2:3b"
MODEL_VISION = "qwen2.5vl:3b"


def add_numbers(a: float, b: float) -> float:
    return a + b


tools = [
    {
        "type": "function",
        "function": {
            "name": "add_numbers",
            "description": "Add two numbers together",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "number"},
                    "b": {"type": "number"},
                },
                "required": ["a", "b"],
            },
        },
    }
]


def _extract_json_block(text: str):
    """Pull a JSON object out of raw text, stripping ```json fences if present."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```", 2)
        text = text[1] if len(text) > 1 else text[0]
        if text.lstrip().lower().startswith("json"):
            text = text.lstrip()[4:]
    text = text.strip().strip("`").strip()
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return None


def run_tool_agent(prompt: str):
    print(f"\n[Tool Agent] Using {MODEL_CODER}...")
    response = client.chat(
        model=MODEL_CODER,
        messages=[{"role": "user", "content": prompt}],
        tools=tools,
    )

    tool_calls = response.get("message", {}).get("tool_calls", [])

    if not tool_calls:
        content = response.get("message", {}).get("content", "")
        parsed = _extract_json_block(content)
        if parsed and parsed.get("name") == "add_numbers":
            tool_calls = [
                {
                    "function": {
                        "name": "add_numbers",
                        "arguments": parsed.get("arguments", {}),
                    }
                }
            ]

    if tool_calls:
        for call in tool_calls:
            if call["function"]["name"] == "add_numbers":
                args = call["function"]["arguments"]
                res = add_numbers(**args)
                print(f"-> Executed Tool: add_numbers({args['a']}, {args['b']}) = {res}")
    else:
        print("Response:", response["message"]["content"])


def analyze_image(image_path: str, prompt: str = "Describe what is in this image"):
    print(f"\n[Vision Engine] Using {MODEL_VISION}...")
    response = client.chat(
        model=MODEL_VISION,
        messages=[
            {
                "role": "user",
                "content": prompt,
                "images": [image_path],
            }
        ],
    )
    print("Vision Output:\n", response["message"]["content"])


def general_chat(prompt: str):
    print(f"\n[Chat Engine] Using {MODEL_CHAT}...")
    response = client.chat(
        model=MODEL_CHAT,
        messages=[{"role": "user", "content": prompt}],
    )
    print("Chat Output:\n", response["message"]["content"])


IMAGE_PATH_RE = re.compile(
    r"([A-Za-z]:\\[^\s]+\.(?:png|jpg|jpeg|gif|bmp|webp)|/[^\s]+\.(?:png|jpg|jpeg|gif|bmp|webp))",
    re.IGNORECASE,
)


def classify_intent(user_input: str) -> str:
    system = (
        "Classify the user's message into exactly one word: "
        "'tool' if it asks to add/sum/calculate two numbers together, "
        "'chat' for anything else (questions, conversation, requests). "
        "Reply with only that single word, nothing else."
    )
    response = client.chat(
        model=MODEL_CHAT,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user_input},
        ],
    )
    label = response["message"]["content"].strip().lower()
    return "tool" if "tool" in label else "chat"


def route(user_input: str):
    match = IMAGE_PATH_RE.search(user_input)
    if match:
        path = match.group(1)
        prompt = user_input.replace(path, "").strip() or "Describe what is in this image"
        analyze_image(path, prompt)
        return

    intent = classify_intent(user_input)
    print(f"[Router] Classified as: {intent}")
    if intent == "tool":
        run_tool_agent(user_input)
    else:
        general_chat(user_input)


def main():
    print("Orchestrator ready. Just type naturally — the router decides which model handles it.")
    print("(Paste an image path to trigger vision. Type exit/quit to stop.)\n")

    while True:
        try:
            raw = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not raw:
            continue
        if raw.lower() in ("exit", "quit"):
            break

        route(raw)


if __name__ == "__main__":
    main()

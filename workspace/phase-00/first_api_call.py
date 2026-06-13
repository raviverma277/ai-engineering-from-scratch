import os
import json
import urllib.request


def call_with_sdk():
    try:
        import anthropic
    except ImportError:
        print("Install the SDK: pip install anthropic")
        return

    client = anthropic.Anthropic()
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=256,
        messages=[{"role": "user", "content": "What is a neural network in one sentence?"}]
    )
    # Handle response content (could be TextBlock or ThinkingBlock)
    for block in response.content:
        if hasattr(block, 'text'):
            print(f"SDK response: {block.text}")
    print(f"Tokens used: {response.usage.input_tokens} in, {response.usage.output_tokens} out")


def call_raw_http():
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Set ANTHROPIC_API_KEY environment variable first")
        return

    url = "https://api.anthropic.com/v1/messages"
    headers = {
        "Content-Type": "application/json",
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
    }
    body = json.dumps({
        "model": "claude-sonnet-4-6",
        "max_tokens": 256,
        "messages": [{"role": "user", "content": "What is a neural network in one sentence?"}],
    }).encode()

    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    with urllib.request.urlopen(req) as resp:
        result = json.loads(resp.read())
        # Handle response content
        for block in result.get('content', []):
            if block.get('type') == 'text':
                print(f"Raw HTTP response: {block['text']}")
        print(f"Tokens used: {result['usage']['input_tokens']} in, {result['usage']['output_tokens']} out")


if __name__ == "__main__":
    print("=== API Calls ===\n")
    print("1. Using the SDK:")
    call_with_sdk()
    print("\n2. Using raw HTTP:")
    call_raw_http()

import os
from anthropic import Anthropic

# Check if API key is set
api_key = os.environ.get("ANTHROPIC_API_KEY")
if not api_key:
    print("ERROR: ANTHROPIC_API_KEY not set!")
    print("Set it with: export ANTHROPIC_API_KEY='your-key'")
    exit(1)

print(f"✓ API key found (length: {len(api_key)})")

# Try to create client
client = Anthropic(api_key=api_key)
print("✓ Client created successfully")

# Make a test request
message = client.messages.create(
    model="claude-opus-4-6",
    max_tokens=100,
    messages=[{"role": "user", "content": "Say 'Hello'"}]
)
print(f"✓ API call successful: {message.content[0].text}")
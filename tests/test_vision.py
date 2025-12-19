import requests
import base64
import pathlib

# Read and encode image
image_path = pathlib.Path("/home/nimesh/Pictures/Screenshots/Screenshot from 2025-12-17 09-19-41.png")

print(f"Image exists: {image_path.exists()}")
print(f"Image size: {image_path.stat().st_size / 1024:.1f} KB")

with image_path.open("rb") as f:
    image_b64 = base64.b64encode(f.read()).decode("utf-8")

print(f"Base64 length: {len(image_b64)}")

# Test API call
payload = {
    "model": "llava:7b-v1.6",
    "prompt": "Extract all text from this image",
    "images": [image_b64],
    "stream": False
}

print("\nSending request to Ollama...")
try:
    r = requests.post("http://localhost:11434/api/generate", json=payload, timeout=120)
    print(f"Status: {r.status_code}")
    
    if r.status_code != 200:
        print(f"Error response: {r.text}")
    else:
        result = r.json()
        print(f"\nSuccess! Response:\n{result['response']}")
        
except Exception as e:
    print(f"Exception: {e}")

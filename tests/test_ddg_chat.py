import json
import urllib.request
import urllib.parse
import re

def ask_llm(prompt):
    print(f"Querying LLM with prompt: {prompt}")
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': '*/*',
        'Accept-Language': 'en-US,en;q=0.5',
        'x-client-extra': '1'
    }
    
    try:
        # Step 1: Get vqd token
        status_url = "https://duckduckgo.com/duckchat/v1/status"
        req = urllib.request.Request(status_url, headers={**headers, 'x-request-vqd': '1'})
        with urllib.request.urlopen(req, timeout=5) as response:
            vqd_token = response.headers.get('x-vqd-4')
            if not vqd_token:
                # Try reading body or search in headers
                print("Headers:", response.headers.items())
                return None
        
        print(f"Token obtained: {vqd_token}")
        
        # Step 2: Send Chat query
        chat_url = "https://duckduckgo.com/duckchat/v1/chat"
        payload = {
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": prompt}]
        }
        data = json.dumps(payload).encode('utf-8')
        
        chat_headers = {
            **headers,
            'x-vqd-4': vqd_token,
            'Content-Type': 'application/json',
            'Accept': 'text/event-stream'
        }
        
        req2 = urllib.request.Request(chat_url, data=data, headers=chat_headers, method='POST')
        with urllib.request.urlopen(req2, timeout=5) as response2:
            body = response2.read().decode('utf-8', errors='ignore')
            
        # Parse stream of events (Server-Sent Events)
        output = []
        for line in body.split("\n"):
            if line.startswith("data:"):
                data_str = line[5:].strip()
                if data_str == "[DONE]":
                    break
                try:
                    event_data = json.loads(data_str)
                    message = event_data.get("message")
                    if message:
                        output.append(message)
                except Exception:
                    pass
        return "".join(output)
        
    except Exception as e:
        print(f"LLM query failed: {e}")
        return None

if __name__ == "__main__":
    res = ask_llm("Explain Valence and Arousal in Russell's circumplex model in 2 sentences in Korean.")
    print("LLM Response:\n", res)

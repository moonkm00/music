import json
import urllib.request
import urllib.parse

def get_llm_recommendations(api_key, targetV, targetE):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"
    
    prompt = f"""You are a music recommendation system. Recommend exactly 3 K-POP, 3 POP, and 3 J-POP songs (total 9 songs) suitable for a photo with the following Russell emotion model coordinates:
Valence (Positivity): {targetV} (0.0 to 1.0)
Energy (Arousal): {targetE} (0.0 to 1.0)

Your response must be a valid JSON object matching this structure:
{{
  "K-POP": [
    {{"title": "Song Title", "artist": "Artist Name", "genre": "Genre", "valence": 0.8, "energy": 0.9, "desc": "Short description of why it fits in Korean"}}
  ],
  "POP": [
    {{"title": "Song Title", "artist": "Artist Name", "genre": "Genre", "valence": 0.8, "energy": 0.9, "desc": "Short description of why it fits in Korean"}}
  ],
  "J-POP": [
    {{"title": "Song Title", "artist": "Artist Name", "genre": "Genre", "valence": 0.8, "energy": 0.9, "desc": "Short description of why it fits in Korean"}}
  ]
}}
Each array must contain exactly 3 unique, real, highly popular songs. Ensure "desc" is in polite Korean (한국어 존댓말). Do not wrap in markdown code blocks. Just return raw JSON.
"""
    
    payload = {
        "contents": [
            {
                "parts": [
                    {
                        "text": prompt
                    }
                ]
            }
        ],
        "generationConfig": {
            "responseMimeType": "application/json"
        }
    }
    
    data = json.dumps(payload).encode('utf-8')
    headers = {'Content-Type': 'application/json'}
    
    try:
        req = urllib.request.Request(url, data=data, headers=headers, method='POST')
        with urllib.request.urlopen(req, timeout=10) as response:
            res_body = response.read().decode('utf-8')
        
        res_json = json.loads(res_body)
        text_content = res_json['candidates'][0]['content']['parts'][0]['text']
        return json.loads(text_content)
    except Exception as e:
        print(f"Gemini LLM call failed: {e}")
        return None

if __name__ == "__main__":
    # Test with a mock key or instruction
    print("Gemini API caller defined.")

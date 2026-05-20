import urllib.request
import urllib.parse
import re

def searchYoutubeVideos(query, count=3):
    print(f"Searching YouTube for: {query}")
    encoded_query = urllib.parse.quote(query)
    url = f"https://www.youtube.com/results?search_query={encoded_query}"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=5) as response:
            html = response.read().decode('utf-8', errors='ignore')
        
        # In YouTube search results JSON inside HTML:
        # "videoIds":["xxxx"] or "videoId":"xxxx"
        video_ids = re.findall(r'"videoId"\s*:\s*"([a-zA-Z0-9_-]{11})"', html)
        if not video_ids:
            # Fallback regex for link hrefs
            video_ids = re.findall(r'/watch\?v=([a-zA-Z0-9_-]{11})', html)
            
        results = []
        seen = set()
        for vid in video_ids:
            # Skip common duplicates/ads
            if vid not in seen:
                seen.add(vid)
                results.append(vid)
                if len(results) >= count:
                    break
        return results
    except Exception as e:
        print(f"YouTube search failed: {e}")
        return []

if __name__ == "__main__":
    res = searchYoutubeVideos("NewJeans How Sweet")
    print("Found video IDs:", res)

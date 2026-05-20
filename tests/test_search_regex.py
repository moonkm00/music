import urllib.request
import urllib.parse
import re

def searchSpotifyTracksRegex(query):
    # Avoid printing problematic characters directly
    print("Searching Spotify tracks...")
    fullQuery = f"site:open.spotify.com/track/ {query}"
    url = "https://html.duckduckgo.com/html/?" + urllib.parse.urlencode({'q': fullQuery})
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=5) as response:
            html = response.read().decode('utf-8', errors='ignore')
        
        # Regex to find links matching open.spotify.com/track/...
        # In DDG HTML, the links inside href typically contain uddg=encoded_url
        links = re.findall(r'uddg=(https?%3A%2F%2Fopen\.spotify\.com%2Ftrack%2F[a-zA-Z0-9%_-]+)', html)
        
        results = []
        seen = set()
        for enc_url in links:
            dec_url = urllib.parse.unquote(enc_url)
            # Normalize url (remove query params if any)
            dec_url = dec_url.split('?')[0]
            if dec_url not in seen:
                seen.add(dec_url)
                
                # Let's extract anchor text (Title) near this link
                # DDG typical html: <a class="result__snippet" href="...uddg=dec_url">...</a> or similar
                # Let's search the HTML to extract the text of this result.
                # Since regex is basic, we can extract the title by looking for the closest previous result__a class
                title = "Spotify Track"
                try:
                    # Let's find the match for the encoded URL in the HTML and search backward for '<a class="result__a"'
                    pos = html.find(enc_url)
                    if pos != -1:
                        sub = html[max(0, pos-2000):pos]
                        # Find the last occurrence of class="result__a"
                        titles = re.findall(r'class="result__a"[^>]*>([^<]+)</a>', sub)
                        if titles:
                            title = titles[-1].strip()
                except Exception:
                    pass
                
                track_id = dec_url.split('/track/')[-1]
                results.append((title, dec_url, track_id))
                if len(results) >= 5:
                    break
        return results
    except Exception as e:
        print(f"Search failed: {e}")
        return []

if __name__ == "__main__":
    res = searchSpotifyTracksRegex("Rose APT.")
    for title, url, tid in res:
        try:
            print(f"Title: {title} | URL: {url} | ID: {tid}")
        except Exception:
            print(f"URL: {url} | ID: {tid}")

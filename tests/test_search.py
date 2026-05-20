import urllib.request
import urllib.parse
import re
from bs4 import BeautifulSoup

def searchSpotifyTracks(query):
    print(f"Searching Spotify tracks for: {query}")
    fullQuery = f"site:open.spotify.com/track/ {query}"
    url = "https://html.duckduckgo.com/html/?" + urllib.parse.urlencode({'q': fullQuery})
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=5) as response:
            html = response.read()
        
        soup = BeautifulSoup(html, 'html.parser')
        results = []
        
        # DuckDuckGo HTML search results are in elements with class 'result__snippet' and 'result__url'
        for a in soup.find_all('a', class_='result__url'):
            href = a.get('href', '')
            # Extract actual URL from DDG redirect url
            match = re.search(r'uddg=(https?://open\.spotify\.com/track/[a-zA-Z0-9]+)', href)
            if match:
                spotifyUrl = urllib.parse.unquote(match.group(1))
                # Find the parent container to get the title
                parent = a.find_parent('div', class_='links_main')
                title = "Spotify Track"
                if parent:
                    title_elem = parent.find('a', class_='result__a')
                    if title_elem:
                        title = title_elem.get_text(strip=True)
                results.append((title, spotifyUrl))
                if len(results) >= 5:
                    break
        return results
    except Exception as e:
        print(f"Search failed: {e}")
        return []

if __name__ == "__main__":
    res = searchSpotifyTracks("Rosé APT.")
    for title, url in res:
        print(f"Title: {title} | URL: {url}")

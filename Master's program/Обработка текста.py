import requests
import re
from collections import Counter

url = "https://www.gutenberg.org/ebooks/996.txt.utf-8"
r = requests.get(url)
if r.status_code == 200:
    text = r.text
    words = re.findall(r'\b\w+\b', text.lower())
    word_counts = Counter(words)
    for word, count in word_counts.most_common(20):
        print(f"{word}: {count}")
else:
    print("Не удалось загрузить текст")

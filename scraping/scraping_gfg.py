import requests
from bs4 import BeautifulSoup
import os
import re
from urllib.parse import urlparse

def extract_gfg_article(url, output_dir="dataset_gfg_dbms"):
    headers = {
        "User-Agent": "Mozilla/5.0"
    }

    try:
        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            print(f" Failed to fetch page. Status code: {response.status_code}")
            return

        soup = BeautifulSoup(response.text, "html.parser")

        # Find the main article content based on current GFG structure
        article = soup.find("article")
        if not article:
            print(" No <article> tag found on the page.")
            return

        content = ""
        for tag in article.find_all(["p", "pre", "ul", "ol", "h2", "h3"]):
            if tag.name == "p":
                content += tag.get_text(strip=True) + "\n\n"
            elif tag.name in ["h2", "h3"]:
                content += f"\n### {tag.get_text(strip=True)}\n"
            elif tag.name in ["ul", "ol"]:
                for li in tag.find_all("li"):
                    content += f"• {li.get_text(strip=True)}\n"
                content += "\n"
            elif tag.name == "pre":
                content += f"\n```python\n{tag.get_text(strip=True)}\n```\n"

        if not content.strip():
            print(" No text content extracted.")
            return

        os.makedirs(output_dir, exist_ok=True)

        # ✅ Generate slug from URL path (excluding params and anchors)
        parsed_url = urlparse(url)
        path_parts = parsed_url.path.rstrip("/").split("/")
        slug = path_parts[-1] if path_parts[-1] else path_parts[-2]
        slug = re.sub(r'[^\w\-]', '_', slug)  # Clean the slug
        file_name = os.path.join(output_dir, f"{slug}.txt")

        with open(file_name, "w", encoding="utf-8") as f:
            f.write(content)

        print(f" Article saved to: {file_name}")

    except Exception as e:
        print(f" Error: {e}")

# Example usage
url = "https://www.geeksforgeeks.org/commonly-asked-dbms-interview-questions-set-2/?ref=lbp"
extract_gfg_article(url)

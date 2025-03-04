import re

def clean_html_tags(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r'<span class="material-symbols-rounded">.*?</span>', '', text)
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\s+([.,!?;:])', r'\1', text)
    return text.strip()

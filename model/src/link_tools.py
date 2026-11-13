"""
link_tools.py - find and separate links in messages (importable).

Typical use
-----------
    from link_tools import extract_links, split_message, strip_links, add_link_columns

    extract_links("Verify: gtbank-secure.xyz/login now")   # ['gtbank-secure.xyz/login']
    strip_links("Verify: gtbank-secure.xyz/login now")     # 'Verify: [URL] now'
    split_message(text)    # {'text': ..., 'text_no_links': ..., 'links': [...], 'link_info': [...]}
    df = add_link_columns(df, "text")   # adds text_no_links, links, n_links (and feature columns)

What counts as a link
---------------------
  * anything starting with http:// , https:// or www.
  * bare domains ('ubasecure.ng', 't.ly/abc12') when the ending is a known TLD
  * email addresses are NOT links (they are skipped)
"""
import re
from urllib.parse import urlparse

# URL shorteners: the real destination is hidden, which scammers like
SHORTENERS = {"bit.ly", "tinyurl.com", "t.ly", "cutt.ly", "rb.gy", "is.gd", "goo.gl", "ow.ly", "t.co", "shorturl.at"}

# Known endings, used only to decide whether a *bare* domain (no http/www) is really a link.
# This keeps 'e.g.' or 'No.5' from being picked up. Add more if your data needs them.
_TLDS = ("com ng co io net org info xyz top online site click live icu help vip cc ly gy gd me app dev biz shop store link "
         "gov edu uk us ca au in ph es fr br pt de za ke gh ae sa tv ws").split()
_TLD_PATTERN = "|".join(sorted(_TLDS, key=len, reverse=True))         # longest first

_EMAIL = r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+"
_URL = (r"(?:https?://|www\.)[^\s<>\"']+"                                       # explicit links
        rf"|(?<![\w@.-])(?:[\w-]+\.)+(?:{_TLD_PATTERN})(?![\w-])(?:/[^\s<>\"']*)?")  # bare domains (+ optional path)
_PATTERN = re.compile(rf"(?P<email>{_EMAIL})|(?P<url>{_URL})", re.I)
_TRAILING = ".,;:!?)]}\"'"                                                      # punctuation that follows a link, not part of it


def find_links(text):
    """Return [(start, end, link_text), ...] in the order they appear."""
    found = []
    for m in _PATTERN.finditer(text):
        if m.group("email"):                      # emails are matched only so they can be skipped
            continue
        raw = m.group("url")
        link = raw.rstrip(_TRAILING)              # '...claim.' -> '...claim'
        found.append((m.start(), m.start() + len(link), link))
    return found


def extract_links(text):
    """List of links in a message (empty list if none)."""
    return [link for _, _, link in find_links(text)]


def strip_links(text, placeholder="[URL]"):
    """Replace every link with a placeholder ('' removes it). Tidies spacing afterwards."""
    out, last = [], 0
    for start, end, _ in find_links(text):
        out += [text[last:start], placeholder]
        last = end
    out.append(text[last:])
    return re.sub(r"\s+", " ", "".join(out)).strip()


def parse_link(link):
    """Break one link into features useful for fraud detection."""
    url = link if re.match(r"^(https?://)", link, re.I) else "http://" + link      # urlparse needs a scheme
    p = urlparse(url)
    host = (p.hostname or "").lower().removeprefix("www.")
    labels = host.split(".")
    return {
        "link": link,
        "host": host,                                            # e.g. gtbank-secure.xyz
        "tld": labels[-1] if labels else "",                     # last label: xyz, com, ng ...
        "has_scheme": bool(re.match(r"^https?://", link, re.I)),
        "uses_https": link.lower().startswith("https://"),
        "path": p.path or "",
        "is_shortener": host in SHORTENERS,
        "n_subdomain_labels": max(len(labels) - 2, 0),           # high = decoy-subdomain trick
        "has_hyphen": "-" in host,
        "has_digit_in_host": any(c.isdigit() for c in host),
        "has_query": bool(p.query),
        "length": len(link),
    }


def split_message(text, placeholder="[URL]"):
    """Separate one message into its text and its links."""
    links = extract_links(text)
    return {
        "text": text,
        "text_no_links": strip_links(text, placeholder),
        "links": links,
        "link_info": [parse_link(l) for l in links],
    }


def add_link_columns(df, text_col="text", placeholder="[URL]"):
    """
    Add link columns to a pandas DataFrame (returns a copy):
      text_no_links, links (list), n_links, has_link,
      plus first-link features: link_tld, link_is_shortener, link_has_hyphen, link_subdomains.
    """
    out = df.copy()
    parts = out[text_col].map(split_message)
    out["text_no_links"] = parts.map(lambda d: d["text_no_links"])
    out["links"] = parts.map(lambda d: d["links"])
    out["n_links"] = out["links"].map(len)
    out["has_link"] = (out["n_links"] > 0).astype(int)
    first = parts.map(lambda d: d["link_info"][0] if d["link_info"] else {})
    out["link_tld"] = first.map(lambda i: i.get("tld", ""))
    out["link_is_shortener"] = first.map(lambda i: int(i.get("is_shortener", False)))
    out["link_has_hyphen"] = first.map(lambda i: int(i.get("has_hyphen", False)))
    out["link_subdomains"] = first.map(lambda i: i.get("n_subdomain_labels", 0))
    return out


if __name__ == "__main__":                       # quick demo:  python link_tools.py
    demo = ["Your MTN data bundle expired. Renew now for only N100: mtn.co.uk/renew.",
            "GTBank: verify at https://gtbank.com.secure-x7k2.top/login?id=Ab3 or email hr@companyjobs.ng",
            "Track it here: bit.ly/3xYz9Q and www.jumia.com.ng",
            "How far? See you Saturday, e.g. 5pm. No.5 Allen Avenue."]
    for msg in demo:
        r = split_message(msg)
        print("\nMESSAGE :", msg, "\nNO LINKS:", r["text_no_links"], "\nLINKS   :", r["links"])
        for info in r["link_info"]:
            print("  ->", {k: info[k] for k in ("host", "tld", "is_shortener", "n_subdomain_labels", "has_hyphen")})

"""
Build scam_dataset_realistic.csv

Steps
  1. Archive scam rows : replace fake .example/.invalid/.test links with phishing-style links
  2. Archive benign rows: add legitimate links to ~55%, and strip the 'Org:' prefix on ~45%
  3. Generate new Nigerian-context rows (benign + scam, English / Pidgin / es-fr-pt-hi)
  4. Add the 94 real messages (kept as split = real_test)
  5. Clean text, build text_masked, assign splits by template, save, report shortcut checks
"""
import re, hashlib, unicodedata
import pandas as pd
import aug_lib as L
rng = L.rng

syn  = pd.read_csv("/mnt/user-data/uploads/internet_scams_archive.csv")
real = pd.read_csv("/mnt/user-data/outputs/scam_messages_nigeria.csv")
real = real[real.is_truncated == 0]

# ============================================================ 1. archive SCAM links
FAKE_URL = re.compile(r"(?:https?://)?[\w-]+(?:\.[\w-]+)*\.(?:example|invalid|test)\b(?:/\S*)?")
PREFIX   = re.compile(r"^([A-Z][^:]{2,40}):")
REAL_BRANDS = {**L.BANKS, **L.TELCOS, **L.SHOPS, **L.GLOBAL, **L.GOVS}

def scam_base(text):
    """Pick the brand a scam message pretends to be, so its link can imitate that brand."""
    for name, (dom, base) in REAL_BRANDS.items():
        if re.search(rf"\b{re.escape(name)}\b", text, re.I):
            return dom, base
    m = PREFIX.match(text)
    if m:
        org = m.group(1).split(" from ")[-1]               # 'Security notice from Ledger & Co. Bank' -> 'Ledger & Co. Bank'
        if len(org.split()) <= 4:
            return None, L.org_slug(org)[:16]
    return None, None

def swap_link(m, text):
    url = m.group(0)
    tail = (re.search(r"[.,;:!?)\]]+$", url) or [""])[0]   # keep punctuation that followed the link
    dom, base = scam_base(text)
    return L.scam_url(dom, base) + tail

def fix_scam(text):
    return FAKE_URL.sub(lambda m: swap_link(m, text), text)

# ============================================================ 2. archive BENIGN links + prefix removal
LINK_PHRASES = {
 "legit_delivery_update":     ("/track",   ["Track it here: {u}", "Tracking: {u}", "Live tracking at {u}"]),
 "legit_bank_statement":      ("",         ["View it at {u}", "Statement: {u}", "Log in at {u}"]),
 "legit_marketing_promo":     ("",         ["Shop now: {u}", "Browse the sale at {u}", "See the offer: {u}"]),
 "legit_job_posting":         ("/careers", ["Apply at {u}", "Full details: {u}", "Careers page: {u}"]),
 "legit_support_reply":       ("/help",    ["Order status: {u}", "Help centre: {u}", "More help at {u}"]),
 "legit_social_notification": ("",         ["Open: {u}", "See what's new: {u}", "View: {u}"]),
}
NAME = r"([A-Z][\w&']*(?: [A-Z&][\w&.']*){0,3})"            # 1-4 capitalised words, e.g. 'ValueLane Store'
ORG_PATTERNS = [re.compile(rf"^{NAME} ?:"), re.compile(rf"^{NAME} is hiring"), re.compile(rf"{NAME} Support\.$")]

def benign_org(text):
    for p in ORG_PATTERNS:
        m = p.search(text)
        if m: return m.group(1).strip()
    return rng.choice(L.COS)[0]

def fix_benign(text, pattern):
    org = benign_org(text)
    strip_prefix = ORG_PATTERNS[0].match(text) and rng.random() < 0.60   # only real org prefixes, not 'Following up on ticket X:'
    if strip_prefix:                                        # 'Org: your parcel...' -> 'Your parcel... - Org'
        body = PREFIX.sub("", text).strip()
        text = body[0].upper() + body[1:]
    if rng.random() < 0.55 and pattern in LINK_PHRASES:     # add a legitimate link
        path, phrases = LINK_PHRASES[pattern]
        link = rng.choice(phrases).format(u=L.legit_url(("www." if rng.random()<.3 else "") + L.org_slug(org) + ".com", path))
        if rng.random() < 0.7 or ". " not in text:
            text = f"{text} {link}"
        else:                                               # sometimes put the link after the first sentence
            first, rest = text.split(". ", 1); text = f"{first}. {link} {rest}"
    if strip_prefix:
        text = f"{text} - {org}"
    return text

# ============================================================ clean + mask (same as before)
def clean_text(s):
    s = unicodedata.normalize("NFC", s)
    s = s.replace("’","'").replace("‘","'").replace("“",'"').replace("”",'"')
    s = re.sub(r"\s*—\s*", " - ", s)
    return re.sub(r"\s+", " ", s).strip()

URL, EMAIL = re.compile(r"(?:https?://)?(?:[\w-]+\.)+[a-z]{2,}(?:/[^\s]*)?", re.I), re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
PHONE = re.compile(r"\+?\d[\d\s\-]{6,}\d")
def mask(s): return PHONE.sub("[NUM]", URL.sub("[URL]", EMAIL.sub("[EMAIL]", s)))

# ============================================================ assemble archive rows
a = syn.copy()
a["text"] = [fix_scam(t) if lab == "Scam" else fix_benign(t, p)
             for t, lab, p in zip(a.message_text, a.label, a.scam_pattern)]
a = pd.DataFrame({"text": a.text, "label": a.label.map({"Scam":1,"Benign":0}), "source": "synthetic",
                  "language": a.language, "category": a.scam_pattern, "split": a.split,
                  "template_id": a.generation_template_id})

# ============================================================ 3. generate new rows
names   = list(L.T)
weights = [L.T[n][3] for n in names]
def make_rows(n_benign, n_scam):
    rows = []
    for label, n in ((0, n_benign), (1, n_scam)):
        pool = [k for k in names if L.T[k][0] == label]
        w    = [L.T[k][3] for k in pool]
        for _ in range(n):
            f = rng.choices(pool, w)[0]
            _, lang, kind, _, variants = L.T[f]
            v = rng.randrange(len(variants))
            text = variants[v].format(**L.ctx(kind))
            rows.append(dict(text=text, label=label, source="augmented", language=lang,
                             category=f, template_id=f"aug:{f}:{v}"))
    return pd.DataFrame(rows)

new = make_rows(n_benign=6000, n_scam=2600)
new["text"] = new.text.map(clean_text)
new = new.drop_duplicates("text")                           # exact repeats add nothing

# split the new rows by TEMPLATE (a template never appears in two splits)
def bucket(tid):
    h = int(hashlib.md5(tid.encode()).hexdigest(), 16) % 10
    return "test" if h == 0 else "val" if h == 1 else "train"
new["split"] = new.template_id.map(bucket)

# ============================================================ 4. real rows
b = pd.DataFrame({"text": real.text, "label": real.label, "source": "real", "language": "English",
                  "category": real.category, "split": "real_test", "template_id": pd.NA})

# ============================================================ 5. finish
df = pd.concat([a, new, b], ignore_index=True)
df["text"]        = df.text.map(clean_text)
df = df.drop_duplicates("text").reset_index(drop=True)   # final safety net
df["text_masked"] = df.text.map(mask)
df["has_url"]     = df.text.str.contains(URL).astype(int)
df.insert(0, "id", range(1, len(df) + 1))
df = df[["id","text","text_masked","label","source","language","category","split","template_id","has_url"]]
df.to_csv("/mnt/user-data/outputs/scam_dataset_realistic.csv", index=False, encoding="utf-8-sig")

# ------------------------------------------------------------ report: did the shortcuts shrink?
print(df.groupby(["source","label"]).size().unstack(), "\n")
print("has_url rate by label     :", df.groupby("label").has_url.mean().round(2).to_dict())
pre = lambda s: s.str.match(r"^[A-Z][^:]{2,40}:").mean().round(2)
print("'Org:' prefix rate        :", {k: pre(g.text) for k, g in df.groupby("label")})
print("non-English share         :", {k: round((g.language!="English").mean(),3) for k, g in df.groupby("label")})
print("em-dash style ' - ' rate  :", {k: round(g.text.str.contains(" - ").mean(),2) for k, g in df.groupby("label")})
print("avg length (chars)        :", df.groupby("label").text.apply(lambda s: s.str.len().mean()).round(0).to_dict())
print("unique masked / rows      :", df.text_masked.nunique(), "/", len(df))
tld = df.text.str.extract(r"https?://[^\s/]+?\.([a-z]{2,6})(?:/|\s|$|[.,])")[0]
print("\nTLDs by label (top):\n", pd.crosstab(tld.fillna("-"), df.label).sort_values(1, ascending=False).head(12).T.to_string())
print("\nsplit x label:\n", pd.crosstab(df.split, df.label).to_string())
print("\nnew-row categories (counts):", new.category.value_counts().to_dict())

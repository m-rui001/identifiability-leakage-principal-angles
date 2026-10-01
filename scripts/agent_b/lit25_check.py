# B round 13f: paced arXiv probe for section 25.9's question -- has anyone already shown that
# delay-embedded / smoothed-index bases manufacture spurious transient growth in data-driven
# non-normal analysis?  Rate limit is real (got "Rate exceeded"), so sleep between queries.
import urllib.request, urllib.parse, xml.etree.ElementTree as ET, time, sys

QUERIES = [
    ("delay embedding + transient growth", 'all:"transient growth" AND all:"delay embedding"'),
    ("spurious + DMD", 'all:"spurious" AND all:"DMD"'),
    ("optimal growth + standardized", 'all:"optimal growth" AND all:"standardized" AND all:"transient"'),
    ("pseudospectrum + estimated operator", 'all:"pseudospectrum" AND all:"estimated operator"'),
    ("non-normal + climate + artifact", 'all:"non-normal" AND all:"climate" AND all:"artifact"'),
    ("moving average + regression + inflated", 'all:"moving average" AND all:"lag-1" AND all:"non-normal"'),
    ("embedding + Koopman + redundancy", 'all:"Koopman" AND all:"redundant" AND all:"observables"'),
    ("transient growth + ENSO", 'all:"transient growth" AND all:"ENSO"'),
]

ns = {"a": "http://www.w3.org/2025/Atom", "o": "http://a9.com/-/spec/opensearch/1.1/"}
out = []
for label, q in QUERIES:
    url = ("https://export.arxiv.org/api/query?" +
           urllib.parse.urlencode({"search_query": q, "start": 0, "max_results": 12,
                                   "sortBy": "relevance", "sortOrder": "descending"}))
    txt, ok = "", False
    for attempt in range(4):
        try:
            txt = urllib.request.urlopen(url, timeout=45).read().decode("utf-8", "ignore")
        except Exception as e:
            txt = "ERR %s" % e
        if txt.startswith("<?xml") or "<feed" in txt[:2000]:
            ok = True
            break
        time.sleep(20 + 20 * attempt)
    out.append("=== %-38s | %s" % (label, q))
    if not ok:
        out.append("   NO-RESULT (%s)" % txt[:80].replace("\n", " "))
        continue
    r = ET.fromstring(txt)
    out.append("   TOTAL = %s" % r.find("{http://a9.com/-/spec/opensearch/1.1/}totalResults").text)
    for e in r.findall("a:entry", ns):
        i = e.find("a:id", ns).text.split("/abs/")[-1]
        t = " ".join(e.find("a:title", ns).text.split())
        p = e.find("a:published", ns).text[:10]
        out.append("   %s | %s | %s" % (p, i, t[:150]))
    time.sleep(12)
print("\n".join(out))

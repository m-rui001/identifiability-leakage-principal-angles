"""Metadata check for the three records the reference list still lacks.

Each of these is mentioned in the manuscript's text but was not in the verified set:
  * Kutz et al., AIAA Journal 2016 (cited in the positioning item on operator identification);
  * Temple 1923 (the classical residual/bracketing eigenvalue bound, which is the natural
    comparator for the certificate's a posteriori side);
  * Szego/Gautschi monographs for the Gauss-rule error-sign statement used in prop:gauss.
If a query returns no article-level record the manuscript must NOT cite it, and the item is
reported to the user as unverifiable rather than silently invented.
"""
import io
import json
import os
import sys
import time
import urllib.parse
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "cref_extra.txt")
UA = "B-citation-check/1.0 (research use; one query per candidate reference)"

QUERIES = [
    "kutz brunton protopapas probert dynamic mode decomposition theoretical analysis generalizations 2016",
    "dynamic mode decomposition data-driven modeling of complex systems SIREAM Kutz 2016",
    "temple algebraic numerical variational method quantum mechanical eigenvalue problems 1923",
    "temple bounds eigenvalues Proceedings London Mathematical Society 1953 bracketing",
    "gautschi computational aspects of classical orthogonal polynomials 2004",
    "szego orthogonal polynomials 1939 american mathematical society",
    "markov chebyshev moment problem positive measures quadrature",
    "golub van loan matrix computations",
]


def p(s):
    sys.stdout.write(s.encode("ascii", "replace").decode("ascii") + "\n")
    sys.stdout.flush()


lines = []
for q in QUERIES:
    url = ("https://api.crossref.org/works?rows=4&query.bibliographic="
           + urllib.parse.quote(q) + "&select=DOI,title,container-title,issued,author,page,volume")
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            code, body = r.status, r.read().decode("utf-8", "replace")
    except Exception as e:  # noqa: BLE001
        code, body = getattr(e, "code", "ERR"), ""
    p("=== QUERY: " + q)
    lines.append("=== QUERY: " + q)
    if code != 200 or not body:
        p("    http=%s -> no data" % code)
        lines.append("    http=%s -> no data" % code)
    else:
        for it in json.loads(body)["message"]["items"]:
            au = "; ".join(" ".join([x.get("given", ""), x.get("family", "")]).strip() or "?"
                           for x in it.get("author", []))
            s = "    %s | %s | %s | %s | %s vol=%s p=%s" % (
                it.get("issued", {}).get("date-parts", [[None]])[0][0],
                it.get("DOI"), au[:90],
                (it.get("title", ["?"]) or ["?"])[0][:95],
                ", ".join(it.get("container-title", []) or ["?"])[:48],
                it.get("volume", "?"), it.get("page", "?"))
            p(s)
            lines.append(s)
    time.sleep(2.5)

with io.open(OUT, "w", encoding="utf-8", newline="\n") as f:
    f.write("\n".join(lines) + "\n")
p("wrote " + OUT)

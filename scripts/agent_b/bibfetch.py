"""Fetch the metadata needed to WRITE thebibliography entries, not just to check existence.

Two routes, both metadata-only (no PDFs):
  * arXiv Atom API in id_list mode --- one request for all 22 identifiers, which returns the
    full author list. The previous round verified that these IDs resolve (B/b65_abs_verify_*.txt)
    but did not record authors, and a bibitem needs them.
  * Crossref /works/<DOI> --- full metadata for the classical journal entries, including the
    COMPLETE author list (B/b65_crossref*.txt printed only the first author, which is enough to
    identify a record and not enough to typeset one).

Everything printed here is the server's own field, unedited, so the bibitem lines can be copied
from this file rather than from memory. A DOI-verified record is what a reference list needs even
when the full text is behind a paywall; a title match is not a read paper, and no claim in the
note is sourced from these entries' contents.
"""
import io
import json
import os
import sys
import time
import urllib.request
import xml.etree.ElementTree as ET

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "bibfetch.txt")

ARXIV = ["1204.5052", "1302.7203", "1707.06000", "2106.09806", "2110.13328", "2211.15643",
         "2307.14688", "2309.11864", "2311.02701", "2401.17757", "2501.05860", "2506.02816",
         "2603.27715", "2603.29001", "2604.02041", "2604.11464", "2606.09369", "2606.27285",
         "2608.04360", "2608.24203", "2608.24529", "2609.24947"]

DOIS = [
    ("golubwelsch1969", "10.1090/s0025-5718-69-99647-1"),
    ("gueisenstat1995", "10.1137/s0895479892241287"),
    ("bunch1978", "10.1007/bf01396012"),
    ("cuppen1980", "10.1007/bf01396757"),
    ("daviskahan1963", "10.1016/0022-247x(63)90001-5"),
    ("daviskahan1965", "10.1016/0022-247x(65)90066-1"),
    ("daviskahan1970", "10.1137/0707001"),
    ("wedin1972", "10.1007/bf01932678"),
    ("bjorck1973", "10.2307/2005662"),
    ("knyazevharresh2002", "10.1137/s1064827500377332"),
    ("knyazev2006", "10.1137/060649070"),
    ("drmac2000", "10.1137/s0895479897320824"),
    ("welch1974", "10.1109/tit.1974.1055219"),
    ("gautschi1996", "10.1017/s0962492900002622"),
    ("frisch1933", "10.2307/1907330"),
    ("lovell1963", "10.1080/01621459.1963.10480682"),
    ("ding2021", "10.1016/j.spl.2020.108945"),
    ("ljung2010", "10.1016/j.arcontrol.2009.12.001"),
    ("kato1995", "10.1007/978-3-642-66282-9"),
    ("sun1998", "10.1137/s0895479895291303"),
]


def get(url, ua):
    req = urllib.request.Request(url, headers={"User-Agent": ua})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.status, r.read().decode("utf-8", "replace")
    except Exception as e:  # noqa: BLE001
        code = getattr(e, "code", None)
        return code if code else "ERR", repr(e)


def main():
    log = []
    def p(s):
        log.append(s)
        # the console on this machine is GBK: printing a server's own non-ASCII field
        # ("Åstrand", "Pötschka") used to abort the whole crawl mid-file, so the console
        # gets an ASCII-safe rendering and the file keeps the exact bytes.
        sys.stdout.write(s.encode("ascii", "replace").decode("ascii") + "\n")
        sys.stdout.flush()

    # ---- arXiv, one id_list request ----
    url = ("https://export.arxiv.org/api/query?id_list=" + ",".join(ARXIV)
           + "&max_results=" + str(len(ARXIV)))
    code, body = get(url, "citation-metadata-verification/1.0 (research use)")
    p("=== arXiv id_list: http=%s bytes=%s" % (code, len(body) if isinstance(body, str) else body))
    if not isinstance(body, str) or code != 200:
        p("FAILED - nothing written; a 400/429 is never a bibliographic result")
    else:
        ns = {"a": "http://www.w3.org/2005/Atom"}
        root = ET.fromstring(body)
        entries = root.findall("a:entry", ns)
        p("entries=%d (asked %d)" % (len(entries), len(ARXIV)))
        for e in entries:
            eid = e.findtext("a:id", "", ns).split("/abs/")[-1]
            title = " ".join(e.findtext("a:title", "", ns).split())
            auth = [" ".join(a.findtext("a:name", "", ns).split())
                    for a in e.findall("a:author", ns)]
            pub = e.findtext("a:published", "", ns)[:10]
            p("%s | %s | %s | %s" % (pub, eid, "; ".join(auth), title))
        p("")

    # ---- Crossref, DOI by DOI ----
    for key, doi in DOIS:
        code, body = get("https://api.crossref.org/works/" + doi,
                "B-citation-check/1.0 (research use; contact via Agent B workspace)")

        if code != 200 or not isinstance(body, str):
            p("### %s | DOI %s | http=%s -> NOT VERIFIED" % (key, doi, code))
            time.sleep(2.0)
            continue
        m = json.loads(body)["message"]
        au = m.get("author", [])
        names = [" ".join([x.get("given", ""), x.get("family", "")]).strip() or "?" for x in au]
        p("### %s | %s" % (key, doi))
        p("    authors: %s" % "; ".join(names))
        p("    title  : %s" % (m.get("title", ["?"])[0]))
        p("    journal: %s  vol=%s  page=%s  issued=%s"
          % (", ".join(m.get("container-title", []) or ["?"]),
             m.get("volume", "?"), m.get("page", "?"),
             m.get("issued", {}).get("date-parts", [[None]])[0][0]))
        p("    pubtype: %s  publisher=%s" % (m.get("type", "?"), m.get("publisher", "?")))
        time.sleep(2.0)

    with io.open(OUT, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(log) + "\n")
    p("\nwrote %s" % OUT)


if __name__ == "__main__":
    sys.exit(main())

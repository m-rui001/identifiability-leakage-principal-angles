"""Author lists for the 22 arXiv references, taken from arxiv.org/abs rather than the API.

Why this host: export.arxiv.org/api/query just answered this round's id_list request with
http=429 and 32 bytes ("Rate exceeded."), and a 429 is a statement about my request rate, never
about the literature (B/arx05.sh is built around that distinction). The abstract pages on
arxiv.org carry the same information as <meta name="citation_author"> tags and are served by a
different host, which was already the route that verified these 22 identifiers resolve
(B/b65_abs_verify_1.txt, B/b65_abs_verify_2.txt).

What this is for: a bibitem needs the FULL author list, and the previous round recorded only
title/abstract/primary subject. Nothing here is read as evidence about a paper's contents ---
these are metadata for the reference list, and the note's positioning claims rest on the abstracts
already quoted in Section "Positioning".
"""
import io
import os
import re
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "absauthors.txt")

ARXIV = ["1204.5052", "1302.7203", "1707.06000", "2106.09806", "2110.13328", "2211.15643",
         "2307.14688", "2309.11864", "2311.02701", "2401.17757", "2501.05860", "2506.02816",
         "2603.27715", "2603.29001", "2604.02041", "2604.11464", "2606.09369", "2606.27285",
         "2608.04360", "2608.24203", "2608.24529", "2609.24947"]

UA = "Mozilla/5.0 (citation metadata for a reference list; one request per paper)"
log = []


def p(s):
    log.append(s)
    sys.stdout.write(s.encode("ascii", "replace").decode("ascii") + "\n")
    sys.stdout.flush()


def meta(html, name):
    return re.findall(r'<meta\s+name="' + name + r'"\s+content="([^"]*)"', html)


def main():
    nbad = 0
    for aid in ARXIV:
        url = "https://arxiv.org/abs/" + aid
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=60) as r:
                code, html = r.status, r.read().decode("utf-8", "replace")
        except Exception as e:  # noqa: BLE001
            code, html = getattr(e, "code", "ERR"), ""
        au = meta(html, "citation_author")
        ti = meta(html, "citation_title")
        yr = meta(html, "citation_date") or meta(html, "citation_online_date")
        if code != 200 or not au:
            nbad += 1
            p("%s | http=%s | AUTHORS NOT RECOVERED (n=%d)" % (aid, code, len(au)))
        else:
            p("%s | %s | %s | %s" % (yr[0][:10] if yr else "?", aid,
                                     "; ".join(x.strip() for x in au),
                                     (ti[0] if ti else "?").strip()))
        sys.stdout.flush()
        time.sleep(4.0)

    p("\nmissing=%d of %d" % (nbad, len(ARXIV)))
    with io.open(OUT, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(log) + "\n")
    p("wrote " + OUT)
    return 0 if nbad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

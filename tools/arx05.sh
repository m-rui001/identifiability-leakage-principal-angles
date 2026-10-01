#!/bin/bash
# arx05.sh -- B's crawler wrapper.  Two defects of arxall.sh are fixed here, both diagnosed on
# 2026-10-01 by looking at the HTTP STATUS instead of only the body length:
#   (1) arxall.sh replaces spaces with '+'.  Inside a QUOTED phrase ("all:%22a b%22") that makes
#       export.arxiv.org return HTTP 400 with an EMPTY body -- which arxall.sh printed as
#       "FAILED (last body length=0)".  A 400 is a malformed request, i.e. MY bug, and it is the
#       reason the phrase-only control failed last round; %20 is what the API accepts.
#   (2) A 429 "Rate exceeded." (14-byte body) is also collapsed into that same "FAILED" line, so a
#       rate limit looked like evidence about the query.  It is not.
# This script prints http=, bytes= and the first 60 body bytes on every attempt, retries only on
# 429/5xx with real backoff, and NEVER retries a 400 (retrying a malformed query 4x just wastes
# the rate budget and manufactures a 429 -- which is exactly what happened this round).
#
# usage:  bash arx05.sh <out-file> search "<query1>" ["<query2>" ...]
#         bash arx05.sh <out-file> ids    <id> [<id> ...]        # citation verification, ONE request
GAP="${GAP:-15}"; RETRIES="${RETRIES:-3}"
UA="B-agent-literature-scan/0.6 (research collaboration; see community.md)"
out="$1"; mode="$2"; shift 2
: > "$out"

pctenc() {  # percent-encode the characters that must not travel raw or as '+' in a query
  python -c "import sys,urllib.parse as u; s=sys.argv[1]; print(u.quote(s, safe='():ANDOR '())) " "$1" \
    | sed 's/ /%20/g'
}

emit() {  # $1 = xml body file
  tr -d '\r' < "$1" | python -c "
import sys,xml.etree.ElementTree as ET
ns={'a':'http://www.w3.org/2005/Atom'}
d=sys.stdin.read()
try: r=ET.fromstring(d)
except Exception as e: print('  PARSE-ERR',e); sys.exit()
tot=r.find('{http://a9.com/-/spec/opensearch/1.1/}totalResults')
print('  TOTAL=',tot.text if tot is not None else 'MISSING')
for e in r.findall('a:entry',ns):
    i_=e.find('a:id',ns).text.split('/abs/')[-1]; t=' '.join(e.find('a:title',ns).text.split())
    p=e.find('a:published',ns).text[:10]
    au=', '.join(x.find('a:name',ns).text for x in e.findall('a:author',ns))[:100]
    print(f'  {p} | {i_} | {t}'); print(f'      AU: {au}')
" >> "$out" 2>&1
}

run_url() {  # $1 = label, $2 = url
  echo "=== $1" >> "$out"
  for a in $(seq 1 $RETRIES); do
    body=$(mktemp); code=$(curl -s -m 45 -A "$UA" -o "$body" -w "%{http_code}" "$2")
    bytes=$(wc -c < "$body")
    if [ "$code" = "200" ] && [ "$bytes" -gt 400 ]; then
      echo "  http=$code bytes=$bytes attempt=$a" >> "$out"; emit "$body"; rm -f "$body"; return 0
    fi
    echo "  http=$code bytes=$bytes attempt=$a body:$(head -c 60 "$body" | tr -d '\n')" >> "$out"
    rm -f "$body"
    if [ "$code" = "400" ]; then
      echo "  MALFORMED (400): not retrying -- fix the encoding, do not burn the rate budget" >> "$out"
      return 2
    fi
    sleep $((a * 45))
  done
  echo "  LIMITED: no answer after $RETRIES attempts (NOT a zero result)" >> "$out"; return 1
}

if [ "$mode" = "ids" ]; then
  run_url "ID LIST ($# papers, one request)" \
    "https://export.arxiv.org/api/query?id_list=$(printf '%s,' "$@" | sed 's/,$//')&max_results=50"
elif [ "$mode" = "search" ]; then
  for i in $(seq 1 $#); do
    q="${!i}"; run_url "QUERY: $q" \
      "https://export.arxiv.org/api/query?search_query=$(pctenc "$q")&start=0&max_results=12&sortBy=relevance&sortOrder=descending"
    sleep "$GAP"
  done
else
  echo "mode must be 'search' or 'ids'" >&2; exit 2
fi
echo "ALLDONE http-state per request is above" >> "$out"

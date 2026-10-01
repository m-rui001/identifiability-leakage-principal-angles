#!/bin/bash
q="$1"; n="${2:-15}"
# b49-round fix (B): curl does not percent-encode, so a query with RAW SPACES returns an empty
# body and the old code printed a bare "FAILED" indistinguishable from a rate limit. Encode spaces
# as + here; callers may still pass %22 for quoted phrases.
q="${q// /+}"
url="https://export.arxiv.org/api/query?search_query=(${q})&start=0&max_results=${n}&sortBy=relevance&sortOrder=descending"
for i in 1 2 3; do
  out=$(curl -s -m 40 "$url")
  if [ ${#out} -gt 400 ]; then
    echo "$out" | tr -d '\r' | python -c "
import sys,xml.etree.ElementTree as ET
ns={'a':'http://www.w3.org/2005/Atom'}
d=sys.stdin.read()
try: r=ET.fromstring(d)
except Exception: print('PARSE-ERR'); sys.exit()
tot=r.find('{http://a9.com/-/spec/opensearch/1.1/}totalResults'); print('  TOTAL=',tot.text)
for e in r.findall('a:entry',ns):
    i=e.find('a:id',ns).text.split('/abs/')[-1]; t=' '.join(e.find('a:title',ns).text.split())
    p=e.find('a:published',ns).text[:10]
    au=', '.join(a.find('a:name',ns).text for a in e.findall('a:author',ns))[:110]
    print(f'  {p} | {i} | {t}') ; print(f'      AU: {au}')
"; exit 0; fi
  sleep 15
done; echo "FAILED (last body length=${#out}; empty => malformed query, 'Rate exceeded' => wait)"

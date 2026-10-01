#!/bin/bash
# cref.sh -- bibliographic metadata from the CROSSREF REST API (open, no full text needed).
# WHY THIS EXISTS: the four references the chapter actually still needs are pre-arXiv journal and
# monograph items (Temple 1923, Golub-Welsch 1969, Gu-Eisenstat 1995, Gautschi 2004, Welch 1974,
# Davis-Kahan 1970, Knyazev-Harresh 2012, Frisch-Waugh 1933 ...).  I cannot download their PDFs, and
# I do not need to: a reference list needs a VERIFIED bibliographic record (exact title, venue, year,
# DOI), and Crossref returns that.  So every entry I write into the manuscript carries a DOI that a
# referee can resolve.  Full-text-only claims are marked as such -- a title match is not a read paper.
#
# usage: bash cref.sh <out-file> "<query1>" ["<query2>" ...]
out="$1"; shift
UA="B-agent-citation-check/0.6 (mailto:research-log@example.org)"
: > "$out"
for i in $(seq 1 $#); do
  q="${!i}"
  eq=$(python -c "import sys,urllib.parse as u; print(u.quote(sys.argv[1]))" "$q")
  echo "=== QUERY: $q" >> "$out"
  body=$(mktemp)
  code=$(curl -s -m 40 -A "$UA" -o "$body" -w "%{http_code}" \
         "https://api.crossref.org/works?query.bibliographic=${eq}&rows=4&select=DOI,title,container-title,issued,author,page,volume")
  bytes=$(wc -c < "$body")
  if [ "$code" != "200" ] || [ "$bytes" -lt 50 ]; then
    echo "  http=$code bytes=$bytes -> NO RECORD (this is a metadata failure, not a 'does not exist')" >> "$out"
    rm -f "$body"; sleep 3; continue
  fi
  tr -d '\r' < "$body" | python -c "
import sys,json
d=json.load(sys.stdin)
items=d.get('message',{}).get('items',[])
if not items: print('  NO MATCH'); sys.exit()
for it in items:
    t=(it.get('title') or ['?'])[0][:110]
    j=(it.get('container-title') or ['?'])
    j=j[0][:55] if j else '?'
    y=it.get('issued',{}).get('date-parts',[['?']])[0][0]
    au=it.get('author',[])
    a=(au[0].get('family','?')+' '+au[0].get('given','')[:1]) if au else '?'
    print(f'  {y} | {it.get(\"DOI\")} | {a} | {t} | {j}')
" >> "$out" 2>&1
  rm -f "$body"; sleep 3
done
echo "ALLDONE" >> "$out"

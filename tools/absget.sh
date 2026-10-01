#!/bin/bash
for id in "$@"; do
  curl -s -m 30 -A "Mozilla/5.0" "https://arxiv.org/abs/$id" -o _t.html
  PYTHONIOENCODING=utf-8 python -c "
import re,html
d=open('_t.html',encoding='utf-8',errors='replace').read()
t=re.search(r'<h1 class=\"title[^>]*>(.*?)</h1>',d,re.S)
a=re.search(r'<blockquote class=\"abstract[^>]*>(.*?)</blockquote>',d,re.S)
c=re.search(r'tablecell comments[^>]*>(.*?)</td>',d,re.S)
s=re.search(r'tablecell subjects[^>]*>(.*?)</td>',d,re.S)
cl=lambda x,n=1500: html.unescape(re.sub('<[^>]+>',' ',x.group(1))).replace('  ',' ').strip()[:n] if x else ''
print('>>> $id |', cl(t,240))
print('ABS:', cl(a))
if c: print('CMT:', cl(c,200))
if s: print('SUBJ:', cl(s,150))
print()
"
  sleep 4
done

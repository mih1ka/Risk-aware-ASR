"""176-term Whisper-base/Whisper-small reference-aligned tier error rates + paired CI.
Requires private ``fareez_base_word_labels.csv`` and ``fareez_small_word_labels.csv``.
No audio, GPU, transcript, model training or online access. Save ONLY aggregate output.
"""
import argparse
import re
import numpy as np
import pandas as pd
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument('--base-labels',type=Path,required=True)
p.add_argument('--small-labels',type=Path,required=True)
p.add_argument('--glossary',type=Path,default=None,help='Optional external 176-term glossary; default derives from repo 190-term glossary')
p.add_argument('--repo',type=Path,default=Path(__file__).resolve().parents[2])
p.add_argument('--out',type=Path,default=Path(__file__).with_name('y176_whisper_base_small_tier_paired.csv'))
a=p.parse_args()
removed={'ramipril','reactine','rosuvastatin','spiriva','ventilin','corticosteroids','cortico','steroid','steroids','opioids','coronavirus','clotting','rheumatological','antalgic'}
glossary=a.glossary or a.repo/'data/glossary/tiered_glossary.csv'
g=pd.read_csv(glossary,keep_default_na=False)
if a.glossary is None:
 assert len(g)==190 and removed.issubset(set(g.term))
 g=g.loc[~g.term.isin(removed)]
assert len(g)==176, 'STOP: 176-term glossary not reproduced'
lookup=dict(zip(g.term.str.lower(),g.tier.astype(int)))
def norm(z):return re.sub(r"[^a-z0-9']",'',str(z).lower())
def prep(path,enc):
 d=pd.read_csv(path,keep_default_na=False)
 need={'encounter_id','gold_word','is_error'}
 assert need.issubset(d),f'Missing {need-set(d)} in {path}'
 d=d.loc[d.gold_word.astype(str).str.strip().ne('')].copy()
 d['tier176']=d.gold_word.map(lambda w:lookup.get(norm(w),1))
 d['is_error']=pd.to_numeric(d.is_error,errors='raise').astype(int)
 counts=d.groupby(['encounter_id','tier176']).is_error.agg(['sum','count'])
 counts=counts.reindex(pd.MultiIndex.from_product([enc,[1,2,3]]),fill_value=0)
 assert set(d.encounter_id)==set(enc)
 return counts['sum'].to_numpy().reshape(len(enc),3),counts['count'].to_numpy().reshape(len(enc),3)
b=pd.read_csv(a.base_labels,usecols=['encounter_id'])
s=pd.read_csv(a.small_labels,usecols=['encounter_id'])
enc=sorted(set(b.encounter_id))
assert len(enc)==272 and set(enc)==set(s.encounter_id)
be,bw=prep(a.base_labels,enc)
se,sw=prep(a.small_labels,enc)
assert np.array_equal(bw,sw),'STOP: Gold reference tier counts differ across base/small.'
assert int((be[:,1:]).sum())==180, 'STOP: Base critical-error count differs from verified R5.'
rng=np.random.default_rng(42)
sel=rng.integers(0,272,size=(2000,272))
rows=[]
for j,tier in enumerate([1,2,3]):
 n=int(bw[:,j].sum()); ebase=int(be[:,j].sum()); esmall=int(se[:,j].sum())
 denom=bw[sel,j].sum(1)
 assert (denom>0).all()
 bp=100*be[sel,j].sum(1)/denom
 sp=100*se[sel,j].sum(1)/denom
 diff=bp-sp
 blo,bhi=np.percentile(bp,[2.5,97.5]); slo,shi=np.percentile(sp,[2.5,97.5]); dlo,dhi=np.percentile(diff,[2.5,97.5])
 rows.append(dict(tier=tier,reference_words=n,base_errors=ebase,small_errors=esmall,
   base_error_pct=100*ebase/n,small_error_pct=100*esmall/n,base_ci_low_pct=blo,base_ci_high_pct=bhi,
   small_ci_low_pct=slo,small_ci_high_pct=shi,base_minus_small_pp=100*(ebase-esmall)/n,
   paired_diff_ci_low_pp=dlo,paired_diff_ci_high_pp=dhi,bootstrap_samples=2000,seed=42))
pd.DataFrame(rows).to_csv(a.out,index=False)
print('Wrote paired 176-term tier table:',a.out)
print(pd.DataFrame(rows).round(3).to_string(index=False))

"""Full 176-term Fareez evaluation. Strict stop on Rashi's 10% checkpoint.
CPU-only. Tier lookup is a validated port of jellyfish's Rust Metaphone,
validated against 352,479 historical 190-term tier outputs exactly.
No model retraining, no paper edits.
"""
import sys, os, json, re
from pathlib import Path
import pandas as pd, numpy as np
from metaphone_port import metaphone
from importlib.util import spec_from_file_location, module_from_spec
SEED=42
import argparse
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--scored',type=Path,required=True)
p.add_argument('--labels',type=Path,required=True)
p.add_argument('--repo',type=Path,default=Path.cwd(),help='Risk-aware-ASR repository root, checked out at upstream/main')
p.add_argument('--out',type=Path,default=Path('experiments/blockers_yashvi'))
p.add_argument('--private',type=Path,required=True,help='PRIVATE location outside Git for transcript excerpt rating sheets')
args=p.parse_args()
GLOSSARY190=args.repo/'data/glossary/tiered_glossary.csv'
SCORED=args.scored
LABELS=args.labels
OUT=args.out
PRIVATE=args.private
if OUT.resolve().is_relative_to(PRIVATE.resolve()) or PRIVATE.resolve().is_relative_to(args.repo.resolve()):
  raise ValueError('Private sheets must be saved outside the Git repository')
OUT.mkdir(parents=True,exist_ok=True)
DROP={"ramipril", "reactine", "rosuvastatin", "spiriva", "ventilin", "corticosteroids", "cortico", "steroid", "steroids", "opioids", "coronavirus", "clotting", "rheumatological", "antalgic"}
WEIGHTS={1:1.0,2:3.25,3:4.25}

def norm(x):return re.sub(r"[^a-z0-9']",'',str(x).lower())

def build_lookup(g):
  lookup=dict(zip(g.term.str.lower(),g.tier.astype(int)))
  ph={}
  for term,tier in lookup.items():
    c=metaphone(term)
    if c:ph[c]=max(ph.get(c,0),tier)
  return lookup,ph

def tier_word(x,lookup,ph):
  n=norm(x)
  if n in lookup:return lookup[n]
  return ph.get(metaphone(n),1)

def apply_tier(s,lookup,ph):
  uni=s.drop_duplicates().tolist()
  lookup_cached={x:tier_word(x,lookup,ph) for x in uni}
  return s.map(lookup_cached).to_numpy(dtype=np.int8)

# An exact 190-term tier match ensures Metaphone reimplementation matches the archived scoring.
g190=pd.read_csv(GLOSSARY190,keep_default_na=False)
assert len(g190)==190
assert set(DROP).issubset(set(g190.term))
g176=g190.loc[~g190.term.isin(DROP)].copy()
assert len(g176)==176
# Compute 176-term vocabulary in memory; do not commit reference-word lists.
old_l,old_ph=build_lookup(g190)
new_l,new_ph=build_lookup(g176)
df=pd.read_csv(SCORED,keep_default_na=False)
assert len(df)==352479
raw_tier=apply_tier(df.base_word,old_l,old_ph)
assert np.array_equal(raw_tier,df.tier_whisper.to_numpy()),'Metaphone failed to reproduce original tier_whisper'
print('EXACT METAPHONE CHECK: 352,479/352,479 words match archived 190-term tiers',flush=True)
# Apply requested legacy six-row scope. No full-scoring data is written to Git.
none=df.base_word.eq('None')
assert int(none.sum())==6 and int(df.loc[none,'base_is_critical_error'].sum())==0
df=df.loc[~none].reset_index(drop=True)
assert len(df)==352473 and df.encounter_id.nunique()==272
n=len(df);idx=np.arange(n)
# Recompute output and reference tiers from the 176 file.
df['tier_whisper']=apply_tier(df.base_word,new_l,new_ph).astype(int)
df['gold_tier']=df.gold_word.apply(lambda x: new_l.get(norm(x),1) if norm(x) else 0).astype(int)
df['tier_weight']=df.tier_whisper.map(WEIGHTS).to_numpy(float)
df['risk_score']=df.learned_uncertainty.to_numpy(float)*df.tier_weight.to_numpy(float)
df['base_is_critical_error']=((df.gold_tier>=2)&(df.base_is_error==1)).astype(int)
critical=df.base_is_critical_error.to_numpy(dtype=np.int64)
reference=pd.read_csv(LABELS,keep_default_na=False)
gold=reference.gold_word.astype(str).str.strip().ne('')
reference=reference.loc[gold].copy()
reference['tier176']=reference.gold_word.map(lambda x:new_l.get(norm(x),1)).astype(int)
reference['critical176']=((reference.tier176>=2)&(reference.is_error==1)).astype(int)
all_count=int(reference.critical176.sum())
del_count=int(((reference.error_type=='deletion')&(reference.critical176==1)).sum())
scorable=int(critical.sum())
print('176-TERM CRITICAL ERRORS: total',all_count,'scorable',scorable,'deleted',del_count,flush=True)
assert (all_count,scorable,del_count)==(180,171,9), 'STOP: 176 reference counts do not reproduce Rashi target'
confidence=df.base_confidence.to_numpy(float)
u=df.learned_uncertainty.to_numpy(float)
risk=df.risk_score.to_numpy(float)
dis=df.disagreement.to_numpy(dtype=np.int8)
tier=df.tier_whisper.to_numpy(dtype=np.int8)
g=(tier>=2)
# 176 baseline rank methods, all deterministic lexicographic.
orders={
 'Raw confidence':np.lexsort((idx,confidence)),
 'Learned uncertainty':np.lexsort((idx,-u)),
 'RiskAware-ASR':np.lexsort((idx,-risk)),
 'Disagreement + confidence':np.lexsort((idx,confidence,-dis)),
}
base10={key:int(critical[rank[:int(n*.1)]].sum()) for key,rank in orders.items()}
print('176 10% BASELINES',base10,'budget',int(n*.10),flush=True)
if base10['Raw confidence']!=88 or base10['RiskAware-ASR']!=100:
  raise RuntimeError('STOP: Rashi R5 176 checkpoint mismatch; no new results generated')
print('RASHI R5 CHECK PASSED, computing remaining rankings',flush=True)
# Gate was passed before constructing extra rankings or metrics.
ids_g=np.flatnonzero(g)
orders.update({
 'A1 Glossary-only strict':ids_g,
 'A2 Glossary-restricted confidence strict':ids_g[np.lexsort((ids_g,confidence[g]))],
 'A3 Glossary first then confidence filled':np.lexsort((idx,confidence,~g)),
 'B (1-confidence) x tier':np.lexsort((idx,-((1-confidence)*df.tier_weight.to_numpy(float)))),
 'D Tier gate then learned uncertainty':np.lexsort((idx,-u,~g)),
 'C1 Disagreement then RiskAware score':np.lexsort((idx,-risk,-dis)),
})

def metrics(b,method,order):
  k=int(n*b/100)
  if method=='A1 Glossary-only strict':k=len(order)
  if method=='A2 Glossary-restricted confidence strict':k=min(k,len(order))
  selected=order[:k]
  caught=int(critical[selected].sum())
  return {'method':method,'budget_pct':b,'flagged_words':k,'critical_caught':caught,'recall_all_pct':100*caught/all_count,'recall_scorable_pct':100*caught/scorable,'precision_pct':100*caught/k if k else np.nan,'flags_per_critical':k/caught if caught else np.nan}
rows=[metrics(b,name,order) for name,order in orders.items() for b in (5,10,20)]
full=pd.DataFrame(rows);full.to_csv(OUT/'y176_full_5_10_20.csv',index=False)
print('FULL 10% RESULTS\n'+full[full.budget_pct.eq(10)][['method','flagged_words','critical_caught','recall_all_pct']].to_string(index=False),flush=True)
# paired whole-encounter bootstrap helper from initial verified implementation
spec=spec_from_file_location('blocker_shared',Path(__file__).with_name('shared_blocker_helpers.py'))
m=module_from_spec(spec);spec.loader.exec_module(m)
enc=np.unique(df.encounter_id.to_numpy())
refcounts=reference.groupby('encounter_id').critical176.sum().reindex(enc,fill_value=0).to_numpy(dtype=np.int64)
assert sum(refcounts)==180
ref_override=(refcounts,all_count,del_count)
boots=m.boot_gaps(df,orders,'RiskAware-ASR',LABELS,critical,all_count,
   restricted_names={'A2 Glossary-restricted confidence strict'},
   unbudgeted_names={'A1 Glossary-only strict'},ref_override=ref_override)
boots.to_csv(OUT/'y176_paired_bootstrap_vs_riskaware.csv',index=False)
# RiskAware - raw (requested direction) = negative of raw - risk, CI reversed.
raw=boots.loc[boots.method.eq('Raw confidence')].copy()
raw['method']='RiskAware-ASR';raw['reference']='Raw confidence';
raw['bootstrap_mean_gap_pp']=-raw['bootstrap_mean_gap_pp']
lo=raw.ci_lower_pp.copy();hi=raw.ci_upper_pp.copy();raw['ci_lower_pp']=-hi;raw['ci_upper_pp']=-lo
raw.to_csv(OUT/'y176_riskaware_vs_raw_5_10_20.csv',index=False)
# Compare C1 vs Disagreement+confidence via paired resampling.
c1_vs_dis=m.boot_gaps(df,{'C1 Disagreement then RiskAware score':orders['C1 Disagreement then RiskAware score'],
                          'Disagreement + confidence':orders['Disagreement + confidence']},
                          'Disagreement + confidence',LABELS,critical,all_count,ref_override=ref_override)
c1_vs_dis.to_csv(OUT/'y176_c1_vs_disagreement_bootstrap.csv',index=False)
# Safe aggregated notes.
with (OUT/'RESULTS.md').open('w') as f:
 f.write('# Full 176-term Metaphone Fareez re-run\n\n')
 f.write('Rebuilt ASR-word and reference tiers from the 190-term source glossary, omitting 14 terms using an exact verified CPU Metaphone port; tier-free learned uncertainty unchanged.\n\n')
 f.write('Original 190-term mapping independently matched 352,479/352,479 archived ASR tiers. The six literal "None" tokens were excluded to match the requested 352,473 legacy scope.\n\n')
 f.write('Rashi R5 10% gate **PASSED**: raw confidence 88, RiskAware 100; total=180, scorable=171, deleted=9; budget=35,247.\n\n')
 f.write('A1 flags all glossary output matches at every budget (non-comparable review cost); A2 stops early if glossary matches are fewer than the budget.\n\n')
 f.write('All scores, weights, ranked selections, and critical labels were recomputed for the 176-term glossary; the ASR model and learned uncertainty classifier were not retrained.\n\n')
 f.write('All uncertainty intervals below are paired whole-encounter bootstraps with seed 42, 2,000 resamples, fixed proportional review budget within each resample.\n\n')
 def table(d):
  return d.to_markdown(index=False,floatfmt='.3f')
 f.write('## Full method table\n\n'+table(full)+'\n\n')
 f.write('## Paired bootstrap difference vs RiskAware (method minus RiskAware, percentage points)\n\n'+table(boots)+'\n\n')
 f.write('## RiskAware vs raw confidence\n\n'+table(raw)+'\n\n')
 f.write('## C1 vs disagreement + confidence\n\n'+table(c1_vs_dis)+'\n\n')
# Stage private ratings ONLY after outputs computed successfully.
private=PRIVATE;private.mkdir(parents=True,exist_ok=True)
# The prior sample code reads current gold_tier and risk_score from the updated df and updated orders.
sheet,key=m.create_rating_sheets(df,orders,private)
# Adapt rated glossary to new 176 terms (avoid original 190 term sheet).
ratings_g=g176.copy();rng=np.random.default_rng(SEED)
chosen=ratings_g.iloc[rng.choice(len(ratings_g),30,replace=False)]
chosen=chosen.iloc[rng.permutation(30)].reset_index(drop=True)
items=[f'T{i:03d}' for i in range(1,31)]
pd.DataFrame({'item_id':items,'term':chosen.term,'assigned_tier_1_2_3':['']*30}).to_csv(private/'y4_glossary_30term_rater_sheet.csv',index=False)
pd.DataFrame({'item_id':items,'source_glossary_tier':chosen.tier}).to_csv(private/'y4_glossary_30term_answer_key.csv',index=False)
m.dummy_kappa_test().to_csv(OUT/'y4_dummy_kappa_test_ONLY.csv',index=False)
# The caller must keep --private outside the Git working tree; no rating sheets are staged.
print('SUCCESS files under',OUT, 'and PRIVATE',private,flush=True)

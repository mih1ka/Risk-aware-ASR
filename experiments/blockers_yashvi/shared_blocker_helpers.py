"""Reused pure analysis utilities; no model training or remote access."""
from __future__ import annotations
import re
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import cohen_kappa_score
SEED=42
BOOT=2000
BUDGETS=(5,10,20)
EXCLUDED_176={"ramipril", "reactine", "rosuvastatin", "spiriva", "ventilin", "corticosteroids", "cortico", "steroid", "steroids", "opioids", "coronavirus", "clotting", "rheumatological", "antalgic"}


def norm(w):
    return re.sub(r"[^a-z0-9']", "", str(w).lower())


def numeric(df, col, kind=float):
    if col not in df:
        raise ValueError(f"Required scored CSV column absent: {col}")
    a = pd.to_numeric(df[col], errors="raise").to_numpy(dtype=kind)
    if not np.isfinite(a).all():
        raise ValueError(f"Missing or infinite values in column: {col}")
    return a


def prepare_boot(df):
    enc, enc_id=np.unique(df["encounter_id"].to_numpy(),return_inverse=True)
    E=len(enc)
    if E!=272:raise ValueError(f"Expected 272 encounters; got {E}")
    words=np.bincount(enc_id,minlength=E).astype(np.int64)
    crit=numeric(df,"base_is_critical_error",int)
    # Include deleted critical words in the reference denominators.
    # Requires reference labels rather than making up per-encounter deletions.
    return enc,enc_id,words,crit


def ref_counts(labels_path,encounters,exclude_176=False):
    if labels_path is None or not labels_path.is_file():
        raise FileNotFoundError("Reference labels CSV is mandatory for paired encounter bootstrap and Y3; supply --labels")
    labels=pd.read_csv(labels_path,keep_default_na=False)
    for col in ["encounter_id","gold_word","tier","is_error","error_type"]:
        if col not in labels:raise ValueError(f"Reference labels lack column {col}")
    gold=labels[labels["gold_word"].astype(str).str.strip().ne("")].copy()
    gold["tier"]=pd.to_numeric(gold["tier"],errors="raise").astype(int)
    gold["is_error"]=pd.to_numeric(gold["is_error"],errors="raise").astype(int)
    if exclude_176:
        excluded=gold["gold_word"].astype(str).map(norm).isin(EXCLUDED_176)
        gold.loc[excluded,"tier"]=1
    crit=gold["tier"].ge(2)&gold["is_error"].eq(1)
    delcrit=crit&gold["error_type"].astype(str).eq("deletion")
    allcounts=crit.astype(int).groupby(gold["encounter_id"]).sum().reindex(encounters,fill_value=0).to_numpy(dtype=np.int64)
    return allcounts,int(crit.sum()),int(delcrit.sum())


def boot_gaps(df, orders, comparator, labels_path, crit, ncrit, restricted_names=frozenset(),
              unbudgeted_names=frozenset(), ref_override=None, n_boot=BOOT):
    """Fast exact paired whole-encounter bootstrap. Requires CPU numba."""
    from numba import njit

    @njit
    def count_at_cutoffs(order, word_enc, labels, frequencies, targets, full_pool, restricted):
        if full_pool or restricted:
            pool=0
            for j in range(order.shape[0]):
                pool += frequencies[word_enc[order[j]]]
            if full_pool:
                for j in range(3):
                    targets[j]=pool
            else:
                for j in range(3):
                    if targets[j]>pool:
                        targets[j]=pool
        caught_at=np.zeros(3, dtype=np.int64)
        acc=0
        caught=0
        next_cut=0
        for j in range(order.shape[0]):
            pos=order[j]
            w=frequencies[word_enc[pos]]
            if w==0:
                continue
            nxt=acc+w
            while next_cut<3 and targets[next_cut] <= nxt:
                caught_at[next_cut]=caught+(targets[next_cut]-acc)*labels[pos]
                next_cut+=1
            caught += w*labels[pos]
            acc=nxt
            if next_cut==3:
                break
        if next_cut!=3:
            raise ValueError("Incomplete ranking under resampled budget")
        return caught_at

    enc,enc_idx,word_counts,_=prepare_boot(df)
    ref,tot,deleted=(ref_counts(labels_path,enc) if ref_override is None else ref_override)
    if tot!=ncrit:raise ValueError(f"Bootstrap reference total {tot} != expected {ncrit}")
    critical=np.asarray(crit,dtype=np.int64)
    enc_idx=np.asarray(enc_idx,dtype=np.int64)
    rng=np.random.default_rng(SEED)
    weights=np.array([np.bincount(rng.integers(0,len(enc),size=len(enc)),minlength=len(enc))
                      for _ in range(n_boot)],dtype=np.int64)
    all_words=weights @ word_counts
    all_critical=weights @ ref
    if np.any(all_critical==0):raise ValueError("Zero reference critical errors in a bootstrap replicate")
    recalls={}
    for name,order in orders.items():
        order=np.asarray(order,dtype=np.int64)
        values=np.zeros((n_boot,3),dtype=float)
        for k in range(n_boot):
            budget=np.array([int(all_words[k]*b/100) for b in BUDGETS],dtype=np.int64)
            flags=count_at_cutoffs(order,enc_idx,critical,weights[k],budget,
                                   name in unbudgeted_names,name in restricted_names)
            values[k]=100*flags/all_critical[k]
        recalls[name]=values
    rows=[]
    for name in orders:
        if name==comparator:continue
        gap=recalls[name]-recalls[comparator]
        for j,budget in enumerate(BUDGETS):
            lo,hi=np.percentile(gap[:,j],[2.5,97.5])
            rows.append(dict(method=name,reference=comparator,budget_pct=budget,
                             bootstrap_mean_gap_pp=float(gap[:,j].mean()),
                             ci_lower_pp=float(lo),ci_upper_pp=float(hi),
                             resamples=n_boot,seed=SEED))
    return pd.DataFrame(rows)


def create_rating_sheets(df,base_orders,output):
    # Original row order within each encounter, using saved original positions.
    for col in ["base_word_index"]:
        if col not in df:raise ValueError(f"Cannot reconstruct +/-8-word context: scored CSV lacks {col}")
    rng=np.random.default_rng(SEED)
    n=len(df);k=int(n*.10)
    maskrisk=np.zeros(n,bool);maskrisk[base_orders["RiskAware-ASR"][:k]]=True
    maskraw=np.zeros(n,bool);maskraw[base_orders["Raw confidence"][:k]]=True
    groups={"RiskAware only":np.flatnonzero(maskrisk&~maskraw),
            "Raw confidence only":np.flatnonzero(maskraw&~maskrisk),
            "Neither flagged control":np.flatnonzero(~maskrisk&~maskraw)}
    for name,pool in groups.items():
        if len(pool)<20:raise ValueError(f"Insufficient Y4(a) candidates in {name}: {len(pool)}")
    samples=[(int(x),name) for name,pool in groups.items() for x in rng.choice(pool,size=20,replace=False)]
    rng.shuffle(samples)
    # Each input row is a hypothesis token; the ordered sequence must stay within an encounter.
    sorted_df=df.reset_index(drop=True).sort_values(["encounter_id","base_word_index"],kind="stable")
    indices_by_enc={eid:group.index.to_numpy(dtype=int) for eid,group in sorted_df.groupby("encounter_id",sort=False)}
    contexts={}
    sampled_rows={x for x,_ in samples}
    for eid,ii in indices_by_enc.items():
        words=df.iloc[ii]["base_word"].astype(str).to_list()
        for pos,row in enumerate(ii):
            if row in sampled_rows:
                left=" ".join(words[max(0,pos-8):pos])
                mid="["+words[pos]+"]"
                right=" ".join(words[pos+1:pos+9])
                contexts[row]=" ".join([left,mid,right]).strip()
    if len(contexts)!=60:raise AssertionError(f"Only {len(contexts)} of 60 contexts recovered")
    user=[];key=[]
    for i,(j,name) in enumerate(samples,start=1):
        item=f"R{i:03d}"
        user.append({"item_id":item,"ASR_context_target_in_brackets":contexts[j],
                     "could_change_diagnosis_medicine_dose_allergy_or_followup_yes_no":""})
        key.append({"item_id":item,"source":name,"encounter_id":str(df.at[j,"encounter_id"]),
                    "base_word_index":int(df.at[j,"base_word_index"]),"ASR_word":str(df.at[j,"base_word"]),
                    "gold_word":str(df.at[j,"gold_word"]),"gold_tier":int(df.at[j,"gold_tier"]),
                    "risk_score":float(df.at[j,"risk_score"])})
    pd.DataFrame(user).to_csv(output/"y4_flagged_word_rater_sheet_PRIVATE.csv",index=False)
    pd.DataFrame(key).to_csv(output/"y4_flagged_word_answer_key_PRIVATE.csv",index=False)
    return pd.DataFrame(user),pd.DataFrame(key)


def check_ratings(df,source_key):
    """Actual reviewer responses only; accepts columns item_id,rater_id,yes_no."""
    req={"item_id","rater_id","yes_no"}
    if not req.issubset(df):raise ValueError(f"Ratings require {req}")
    raters=sorted(df["rater_id"].unique())
    if len(raters)!=2:raise ValueError("Cohen's kappa requires exactly two raters")
    a=df[df.rater_id==raters[0]].set_index("item_id")["yes_no"].str.strip().str.lower()
    b=df[df.rater_id==raters[1]].set_index("item_id")["yes_no"].str.strip().str.lower()
    joint=pd.concat([a.rename("a"),b.rename("b")],axis=1).dropna()
    if not set(joint.a).union(joint.b).issubset({"yes","no"}):raise ValueError("Responses must be yes/no")
    if len(joint)!=60:raise ValueError(f"Need 60 paired real ratings; found {len(joint)}")
    kappa=float(cohen_kappa_score(joint.a,joint.b,labels=["no","yes"]))
    combined=joint.join(source_key.set_index("item_id")["source"],how="inner")
    bysource=[]
    for label,g in combined.groupby("source"):
        bysource.append({"source":label,"n_items":len(g),"rater1_yes_pct":100*(g.a=="yes").mean(),
                         "rater2_yes_pct":100*(g.b=="yes").mean(),"kappa_all":kappa})
    return pd.DataFrame(bysource),kappa


def check_glossary_three_author_agreement(ratings):
    """Use ONLY genuine author tier ratings: item_id, rater_id, tier (1/2/3)."""
    required={"item_id","rater_id","tier"}
    if not required.issubset(ratings):raise ValueError(f"Tier ratings require {required}")
    raters=sorted(ratings["rater_id"].unique())
    if len(raters)!=3:raise ValueError("Exactly three distinct authors required")
    pivot=ratings.pivot(index="item_id",columns="rater_id",values="tier")
    if pivot.shape!=(30,3) or pivot.isna().any().any():
        raise ValueError("Need all 30 terms rated by all three authors")
    for rater in raters:
        if not set(pd.to_numeric(pivot[rater])).issubset({1,2,3}):
            raise ValueError("Glossary tiers must be 1, 2 or 3")
    out=[]
    from itertools import combinations
    for a,b in combinations(raters,2):
        x=pivot[a].astype(int);y=pivot[b].astype(int)
        out.append({"rater_a":a,"rater_b":b,"n_terms":len(pivot),
                    "cohen_kappa":float(cohen_kappa_score(x,y,labels=[1,2,3])),
                    "exact_agreement_pct":float(100*(x==y).mean())})
    return pd.DataFrame(out)


def dummy_kappa_test():
    """TEST DATA ONLY: not scientific ratings; no real conclusions."""
    rng=np.random.default_rng(SEED)
    truth=rng.choice(["yes","no"],size=60,p=[.35,.65]);r2=truth.copy()
    flip=rng.choice(60,size=11,replace=False)
    r2[flip]=np.where(r2[flip]=="yes","no","yes")
    k=float(cohen_kappa_score(truth,r2,labels=["yes","no"]))
    return pd.DataFrame([{"TEST_ONLY":True,"dummy_n":60,"dummy_kappa":k,"dummy_seed":SEED}])

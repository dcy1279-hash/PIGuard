"""Read-only replay of archived outputs; NOT independent model inference."""
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import sys
import zipfile

import eval_m3_gradient_text_full as m


def same(expected, actual):
    assert set(expected)==set(actual),(set(expected)^set(actual))
    for key,value in expected.items():
        saved=actual[key]
        if value is None:
            assert saved=="",(key,value,saved)
        elif isinstance(value,bool):
            assert saved==str(value),(key,value,saved)
        elif isinstance(value,(int,float)):
            assert math.isclose(float(saved),value,rel_tol=1e-11,abs_tol=1e-11),(key,value,saved)
        else:
            assert saved==value,(key,value,saved)


def main(archive,source):
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        read=lambda name:json.loads(z.read(name))
        csvread=lambda name:list(csv.DictReader(io.StringIO(z.read(name).decode())))
        manifest=read("manifest.json");plan=read("text_plan.json")
        assert manifest["complete"] is True and manifest["technical_verified"] is True
        m.validate_plan(plan);assert plan==m.prepare(source)
        assert manifest["plan_sha256"]==hashlib.sha256(z.read("text_plan.json")).hexdigest()
        assert manifest["script_sha256"]==hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest()
        for key in ("protocol","config","checkpoint_sha256","model_file_sha256",
                    "source_file_sha256","pilot_sha256","generator_sha256"):
            assert manifest[key]==plan[key],key
        eligibility=csvread("eligibility.csv")
        assert len(eligibility)==75
        for r,s in zip(plan["originals"],eligibility):
            same(dict(sample_id=r["sample_id"],category=r["category"],
                original_prediction=r["source_original_prediction"],label=r["label"],eligibility=m.reason(r)),s)
        feasibility=csvread("feasibility.csv");rankings=csvread("word_rankings.csv")
        outcomes=csvread("outcomes.csv");logs=csvread("query_trace.csv")
        assert [r["sample_id"] for r in feasibility]==[r["sample_id"] for r in plan["selected"]]
        assert len(outcomes)==22*12 and len(logs)==manifest["candidate_logical_queries"]
        reconstructed=[];physical=0;pointer=0
        for record,probe in zip(plan["selected"],feasibility):
            sid=record["sample_id"]
            assert int(probe["native_prediction"])==record["label"]==1
            assert all(probe[k]=="True" for k in ("native_logits_match","native_tokens_match","gradient_finite_nonzero"))
            assert float(probe["gradient_norm"])>0 and math.isfinite(float(probe["gradient_norm"]))
            assert int(probe["gradient_forward_passes"])==int(probe["gradient_backward_passes"])==1
            words=[]
            for row in rankings:
                if row["sample_id"]!=sid:continue
                words.append(dict(start=int(row["start"]),end=int(row["end"]),token=row["token"],
                    visible=row["visible"]=="True",overlapping_tokens=int(row["overlapping_tokens"]),
                    score=float(row["score"]) if row["score"] else None))
            assert [(w["start"],w["end"],w["token"]) for w in words]==m.rules.eligible_words(record["original"])
            assert sum(w["visible"] for w in words)==int(probe["visible_words"])
            assert all(w["score"] is None or math.isfinite(w["score"]) for w in words)
            schedules=m.orders(words,sid)
            for word,row in zip(words,[r for r in rankings if r["sample_id"]==sid]):
                for method in m.STRATEGIES:
                    rank=schedules[method].index(word["start"])+1 if word["visible"] else None
                    assert row[method+"_rank"]==(str(rank) if rank is not None else "")
            saved=[r for r in logs if r["sample_id"]==sid]
            lookup={};audits={}
            for row in saved:
                pred=int(row["prediction"]);prob=float(row["probability"])
                assert pred in (0,1) and math.isfinite(prob) and 0<=prob<=1
                value=(pred,prob)
                assert row["text"] not in lookup or lookup[row["text"]]==value
                lookup[row["text"]]=value
                audits[row["text"]]=dict(token_count=int(row["token_count"]),truncated=int(row["truncated"]),edited_span_visibility=row["edited_span_visibility"])
            replay,trace,calls,_=m.evaluate_sample(record,words,lambda t:lookup[t],lambda t,s:audits[t])
            for expected,row in zip(replay,outcomes[pointer:pointer+12]):same(expected,row)
            pointer+=12
            assert len(trace)==len(saved)
            for expected,row in zip(trace,saved):same(expected,row)
            physical+=calls;reconstructed.extend(replay)
        rows,random_rows,pairs=m.summaries(reconstructed,[r["sample_id"] for r in plan["selected"]])
        for filename,expected in (("budget_summary.csv",rows),("random_seed_summary.csv",random_rows),("paired_comparisons.csv",pairs)):
            saved=csvread(filename);assert len(saved)==len(expected)
            for a,b in zip(expected,saved):same(a,b)
        assert physical==manifest["candidate_physical_inputs"]==460
        assert len(plan["selected"])==manifest["text_records"]==22
        assert all(manifest[k]==22 for k in ("original_native_forward_passes","gradient_ranking_forward_passes","gradient_ranking_backward_passes"))
        assert manifest["parameter_updates"]==0
        print("PASS: CRC, source/cohort, hashes, 22 technical records, 264 outcomes, 1453 trial replay,")
        print("48 budget rows, 4 multiseed summaries, 44 paired comparisons, shared physical cache.")
        print("Actual gradients/predictions were NOT rerun; replay uses archived scores and model outputs.")
        print("Truncated attempts:",sum(int(r["truncated"]) for r in logs))
        print("Non-full edit visibility:",sum(r["edited_span_visibility"]!="full" for r in logs))
        print("Ranking seconds:",sum(float(r["ranking_elapsed_seconds"]) for r in feasibility))
        by={(r["sample_id"],r["strategy"]):r for r in reconstructed}
        gradient_only=[sid for sid in sorted(r["sample_id"] for r in plan["selected"]) if by[sid,"gradient"]["success"] and not by[sid,"left_to_right"]["success"]]
        failed=[r["sample_id"] for r in plan["selected"] if not by[r["sample_id"],"gradient"]["success"]]
        print("Gradient-only versus left at 10:",gradient_only)
        print("Gradient failed at 10:",failed)
        print("Archive SHA256:",hashlib.sha256(Path(archive).read_bytes()).hexdigest())


if __name__=="__main__":main(Path(sys.argv[1]),Path(sys.argv[2]))

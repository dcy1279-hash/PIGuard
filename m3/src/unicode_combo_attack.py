"""Model-independent score-oracle Unicode search. No tensor/model/gradient access.

The only target information accepted by attack() is oracle(text)->(label,p1).
Single edits probe positions, then a width-two beam combines up to three edits.
Controls use the same anchors/stage caps but never rank nodes by target scores.
"""
from itertools import combinations, zip_longest
import hashlib
import math
import random

from eval_m3_unicode_stability import CYRILLIC, eligible_words

MAX_QUERIES = 30
MAX_EDITS = 3
ANCHORS = 8
STAGE_LIMITS = (8, 8, 10)
REFINEMENT_QUERIES = 3
BEAM_WIDTH = 2
SEARCH_QUERY_CAP = MAX_QUERIES - REFINEMENT_QUERIES  # includes original query
RANDOM_SEEDS = tuple(range(5230, 5240))
STRATEGIES = ("single_only", "sequential_combo", "feedback_combo") + tuple(f"random_combo_{s}" for s in RANDOM_SEEDS)
CONFIG = dict(max_total_queries=MAX_QUERIES, max_edited_codepoints=MAX_EDITS,
    original_query_counted=True, max_refinement_queries=REFINEMENT_QUERIES,
    search_cap_including_original=SEARCH_QUERY_CAP, anchors=ANCHORS,
    stage_candidate_caps=list(STAGE_LIMITS), beam_width=BEAM_WIDTH,
    position_order="round robin words left-to-right, mappable letters within word left-to-right",
    mutation="fixed Cyrillic confusables; distinct positions; reversible substitutions only",
    combo_support="same first up-to-eight round-robin positions for all combination methods",
    feedback="rank observed unsuccessful nodes by injection probability, lower first; ties positions",
    random_seeds=list(RANDOM_SEEDS), random_order="SHA256(seed:sample_id:depth) local RNG",
    single_control="up to 29 round-robin single substitutions, no combination; may explore outside anchors",
    refinement="at most three deletion-of-edit checks; no claim of global minimum",
    threat_model="simulated score-visible black box; label+probability only, no gradients/weights",
    independent_holdout=False)


def positions(text):
    """Breadth across words before revisiting a word; text-only static scope."""
    words = [[a+i for i,c in enumerate(token) if c in CYRILLIC]
             for a,b,token in eligible_words(text)]
    return [p for group in zip_longest(*words) for p in group if p is not None]


def apply_edits(text, edits):
    edits=tuple(sorted(edits))
    legal=set(positions(text))
    if len(set(edits))!=len(edits) or len(edits)>MAX_EDITS or not set(edits)<=legal:
        raise ValueError("Invalid/protected/duplicate Unicode edit")
    chars=list(text)
    for pos in edits: chars[pos]=CYRILLIC[text[pos]]
    changed="".join(chars)
    if sum(a!=b for a,b in zip(text,changed))!=len(edits) or len(changed)!=len(text):
        raise AssertionError("Unexpected text transformation")
    if changed.translate(str.maketrans({v:k for k,v in CYRILLIC.items()}))!=text:
        raise ValueError("Original must be ASCII for exact reversibility")
    return changed


def random_order(nodes, seed, sid, depth):
    nodes=list(nodes)
    local=int.from_bytes(hashlib.sha256(f"{seed}:{sid}:{depth}".encode()).digest()[:8],"big")
    random.Random(local).shuffle(nodes)
    return nodes


def children(parents, anchors):
    """Round robin two parents, deduplicated canonical edit sets."""
    lists=[[(tuple(sorted((*p,a))),p) for a in anchors if a not in p] for p in parents]
    seen=set();result=[]
    for group in zip_longest(*lists):
        for item in group:
            if item is None: continue
            node,parent=item
            if node not in seen:
                seen.add(node);result.append((node,parent))
    return result


def attack(text, oracle, strategy, sample_id="input"):
    if strategy not in STRATEGIES: raise ValueError("Unknown frozen strategy")
    if not text.isascii(): raise ValueError("Frozen experiment requires ASCII originals")
    support=positions(text);anchors=support[:ANCHORS]
    if len(support)<3: raise ValueError("Insufficient candidate positions")
    seed=int(strategy.rsplit("_",1)[1]) if strategy.startswith("random_combo_") else None
    trace=[];winner=None;first_success=0;best=None

    def query(edits,phase,parent=()):
        nonlocal best
        if len(trace)>=MAX_QUERIES: raise RuntimeError("Total query cap exceeded")
        current=apply_edits(text,edits)
        label,probability=oracle(current)
        if label not in (0,1) or not math.isfinite(probability) or not 0<=probability<=1:
            raise ValueError("Invalid score-visible oracle response")
        row=dict(query_index=len(trace)+1,phase=phase,depth=len(edits),
            positions=list(edits),parent_positions=list(parent),text=current,
            prediction=int(label),injection_probability=float(probability))
        trace.append(row)
        if best is None or (probability,len(edits),tuple(edits))<(best["injection_probability"],best["depth"],tuple(best["positions"])):
            best=row
        return row

    original=query((),"original")
    if original["prediction"]!=1:
        raise RuntimeError("Original no longer detected: stop, do not count as new success")
    if strategy=="single_only":
        for pos in support[:MAX_QUERIES-1]:
            row=query((pos,),"single")
            if row["prediction"]==0:
                winner=row;first_success=row["query_index"];break
    else:
        previous=[]
        for depth,cap in enumerate(STAGE_LIMITS,1):
            if depth==1:
                proposals=[((p,),()) for p in anchors]
            elif strategy=="feedback_combo":
                ranked=sorted(previous,key=lambda r:(r["injection_probability"],tuple(r["positions"])))[:BEAM_WIDTH]
                proposals=children([tuple(r["positions"]) for r in ranked],anchors)
            else:
                proposals=[(tuple(sorted(c)),()) for c in combinations(anchors,depth)]
            if seed is not None:
                proposals=random_order(proposals,seed,sample_id,depth)
            observed=[]
            for edits,parent in proposals[:cap]:
                if len(trace)>=SEARCH_QUERY_CAP: break
                row=query(edits,"single" if depth==1 else "combination",parent)
                observed.append(row)
                if row["prediction"]==0:
                    winner=row;first_success=row["query_index"];break
            if winner is not None: break
            previous=observed
            if not previous: break

    first_winner=winner
    refinement_calls=0
    if winner is not None and strategy!="single_only":
        # Removal order fixed once. A limited greedy reduction, not a proof of minimality.
        for pos in reversed(winner["positions"]):
            if len(winner["positions"])<=1 or refinement_calls>=REFINEMENT_QUERIES: break
            reduced=tuple(p for p in winner["positions"] if p!=pos)
            checked=query(reduced,"refinement",tuple(winner["positions"]))
            refinement_calls+=1
            if checked["prediction"]==0: winner=checked
    selected=winner if winner is not None else best
    result=dict(strategy=strategy,random_seed=seed,success=winner is not None,
        total_queries=len(trace),search_queries=len(trace)-refinement_calls,
        refinement_queries=refinement_calls,first_success_query=first_success,
        first_success_edits=first_winner["depth"] if first_winner else 0,
        first_success_positions=first_winner["positions"] if first_winner else [],
        first_success_text=first_winner["text"] if first_winner else "",
        final_edits=winner["depth"] if winner else 0,final_positions=winner["positions"] if winner else [],
        final_text=winner["text"] if winner else "",candidate_positions=len(support),
        anchor_positions=anchors,best_probability=best["injection_probability"],
        best_positions=best["positions"],best_text=best["text"],
        selected_probability=selected["injection_probability"],
        original_probability=original["injection_probability"],
        global_minimum_proven=False)
    return result,trace

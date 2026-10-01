"""
CreateScore Batch & Cost Planner (corrected version)
====================================================
A Streamlit app for planning a CreateScore evaluation run: given a candidate
pool (a raw count or an uploaded job-order payload), it computes the stratified
train/test split, a chunked run schedule over the test set, a wall-clock time
estimate, and a projected escalation / token-cost comparison against an
"escalate every criterion" reference baseline.

This is a planning tool, not a re-implementation of the DAG: none of the
routing, scoring or posterior logic runs here. Every default below is a value
measured in the corrected evaluation (61 test candidates, one run, one
escalation-context policy: the fixed-window slice, A1) and every default can be
overridden. The projections are proportional extrapolations; they are not
predictions of agreement, and the escalation rates in particular depend on the
corpus and on the calibrated threshold.

Run with:  streamlit run createscore_planner_app.py
"""

import json

import pandas as pd
import streamlit as st

# --------------------------------------------------------------------------
# Constants measured in the corrected evaluation (rerun_fixed.py / analyze_rerun.py)
# --------------------------------------------------------------------------
DEFAULT_TRAIN_FRAC = 0.70
DEFAULT_N_CRITERIA = 5
DEFAULT_CHUNK_SIZE = 10

# Routing (identical under every context policy; threshold 0.2253 frozen on the training fold)
CRITERION_ESCALATION_RATE = 68 / 305          # 22.3% of criterion decisions escalated
CANDIDATE_ESCALATION_RATE = 44 / 61           # 72.1% of candidates have >= 1 escalated criterion
CANDIDATE_RESOLVED_LOCALLY = 17 / 61          # 27.9% of candidates need no reference call

# Provider-recorded tokens, single run (reference = gpt-oss-120b via Groq)
TOKENS_PER_CANDIDATE_REFERENCE = 1986         # Condition B: 121,174 tokens / 61 candidates (all 5 criteria)
TOKENS_PER_ESCALATED_CANDIDATE = 958          # A1: 42,170 tokens / 44 escalated candidates
MEASURED_TOKEN_REDUCTION = 65.2               # % (121,174 -> 42,170) at the measured routing mix

# Timing of the evaluation run (reference call for every candidate + routed call when escalated)
SECONDS_PER_CANDIDATE_UNTHROTTLED = 16.0      # mean of 58 inter-candidate intervals under 60 s
SECONDS_PER_CANDIDATE_OVERALL = 34.1          # 2,081 s / 61, including two rate-limit pauses (about 19 min)

st.set_page_config(page_title="CreateScore Planner", layout="wide")

import inspect

# Newer Streamlit versions use width="stretch"; older ones use use_container_width=True.
STRETCH = {"width": "stretch"} if "width" in inspect.signature(st.dataframe).parameters else {"use_container_width": True}

st.title("CreateScore Batch & Cost Planner")
st.caption(
    "Plans an evaluation run: split sizing, run schedule, wall-clock time and projected "
    "escalation volume and reference-model token cost, scaled from the corrected evaluation."
)
st.info(
    "Planning tool only. Defaults come from one run on 61 synthetic test candidates and one "
    "escalation-context policy (fixed window). Escalation rates depend on the corpus and the "
    "calibrated threshold (the study's own training fold resolved 71.7% of decisions locally, "
    "its test fold 77.7%). Treat every figure as a planning estimate, not a guarantee."
)

# --------------------------------------------------------------------------
# Sidebar: inputs
# --------------------------------------------------------------------------
with st.sidebar:
    st.header("1. Candidate pool")

    payload_file = st.file_uploader(
        "Optional: job-order payload (.json)",
        type=["json"],
        help="If it contains a candidate list and/or criteria list, N and the "
             "criteria count are inferred from it and pre-fill the fields below.",
    )

    inferred_n = None
    inferred_crit = None
    if payload_file is not None:
        try:
            payload = json.load(payload_file)

            def _find_list(obj, keys):
                if isinstance(obj, dict):
                    for k in keys:
                        if k in obj and isinstance(obj[k], list):
                            return obj[k]
                    for v in obj.values():
                        found = _find_list(v, keys)
                        if found is not None:
                            return found
                elif isinstance(obj, list):
                    return obj
                return None

            cand_list = _find_list(payload, ["candidates", "records", "profiles", "cvs"])
            crit_list = _find_list(payload, ["criteria", "criteria_evaluations", "requirements"])

            if cand_list is not None:
                inferred_n = len(cand_list)
            if crit_list is not None:
                inferred_crit = len(crit_list)
            elif isinstance(payload, dict) and "criteria" in payload and isinstance(payload["criteria"], dict):
                inferred_crit = len(payload["criteria"])

            st.success(
                f"Parsed payload: inferred N = {inferred_n if inferred_n is not None else 'n/a'}, "
                f"criteria = {inferred_crit if inferred_crit is not None else 'n/a'}"
            )
        except Exception as e:
            st.error(f"Could not parse payload as JSON: {e}")

    n_candidates = st.number_input(
        "Total candidates (N)",
        min_value=1, max_value=100000,
        value=int(inferred_n) if inferred_n else 200,
        step=1,
    )
    n_criteria = st.number_input(
        "Number of criteria per candidate",
        min_value=1, max_value=50,
        value=int(inferred_crit) if inferred_crit else DEFAULT_N_CRITERIA,
        step=1,
    )

    st.header("2. Train/test split")
    train_frac = st.slider("Train fraction", 0.50, 0.90, DEFAULT_TRAIN_FRAC, 0.05)
    st.caption("Study default: 0.70, stratified on the seniority criterion (MH_1), seed = 42.")

    st.header("3. Run schedule (over test set)")
    chunk_size = st.number_input(
        "Candidates per chunk", min_value=1, max_value=200, value=DEFAULT_CHUNK_SIZE, step=1,
        help="A scheduling and checkpointing unit only. The corrected harness caches every call, "
             "so a run can be stopped and resumed at any candidate.",
    )

    st.header("4. Timing assumption")
    timing_basis = st.radio(
        "Time basis",
        ["Unthrottled (about 16 s per candidate)", "Including rate-limit pauses (about 34 s per candidate)", "Custom"],
        index=0,
        help="The study's run was paused twice by provider rate limits (about 19 minutes in total). "
             "Whether you meet those pauses depends on your provider tier and request volume.",
    )
    if timing_basis.startswith("Unthrottled"):
        seconds_per_candidate = SECONDS_PER_CANDIDATE_UNTHROTTLED
    elif timing_basis.startswith("Including"):
        seconds_per_candidate = SECONDS_PER_CANDIDATE_OVERALL
    else:
        seconds_per_candidate = st.slider("Seconds per candidate", 1.0, 120.0, SECONDS_PER_CANDIDATE_UNTHROTTLED, 0.5)

    st.header("5. Escalation-rate assumption")
    crit_esc_rate = st.slider(
        "Criterion-level escalation rate", 0.0, 1.0, round(CRITERION_ESCALATION_RATE, 3), 0.01,
        help="Study observed: 22.3% (68 of 305 criterion decisions) on the test fold at the frozen threshold.",
    )
    cand_esc_rate = st.slider(
        "Candidate-level escalation rate (at least one criterion escalated)", 0.0, 1.0,
        round(CANDIDATE_ESCALATION_RATE, 3), 0.01,
        help="Study observed: 72.1% (44 of 61 candidates). The remaining 27.9% needed no reference call.",
    )

    st.header("6. Token assumptions")
    tok_ref = st.number_input(
        "Reference tokens per candidate (all criteria)", min_value=1, max_value=200000,
        value=TOKENS_PER_CANDIDATE_REFERENCE, step=50,
        help="Study observed: 1,986 (121,174 tokens / 61 candidates; five criteria, CV truncated at 8,000 characters). "
             "Scale it yourself for a different rubric size or CV length.",
    )
    tok_esc = st.number_input(
        "Reference tokens per escalated candidate (routed call)", min_value=1, max_value=200000,
        value=TOKENS_PER_ESCALATED_CANDIDATE, step=50,
        help="Study observed: 958 (42,170 tokens / 44 escalated candidates; about 1.5 criteria and a "
             "section-based excerpt of at most 2,500 characters).",
    )

# --------------------------------------------------------------------------
# Derived: split
# --------------------------------------------------------------------------
n_train = round(n_candidates * train_frac)
n_test = n_candidates - n_train

st.subheader("Split summary")
c1, c2, c3 = st.columns(3)
c1.metric("Total candidates (N)", f"{n_candidates}")
c2.metric("Train", f"{n_train}", f"{train_frac*100:.0f}%")
c3.metric("Test", f"{n_test}", f"{(1-train_frac)*100:.0f}%")
st.caption(
    "The study's own split (N = 200, train fraction 0.70) produced 139 train and 61 test candidates, "
    "stratified on MH_1 and fixed at seed = 42 so that it is reproducible. The planner uses simple rounding, so the "
    "same inputs show 140 and 60; the one-candidate difference comes from the stratified split."
)

# --------------------------------------------------------------------------
# Derived: run schedule
# --------------------------------------------------------------------------
st.subheader("Run schedule")
n_chunks = -(-n_test // chunk_size) if n_test > 0 else 0

rows = []
remaining = n_test
for i in range(n_chunks):
    size = chunk_size if remaining >= chunk_size else remaining
    remaining -= size
    rows.append({"Chunk": i, "Size": size, "Est. time (s)": round(size * seconds_per_candidate, 1)})
chunk_df = pd.DataFrame(rows)

bc1, bc2 = st.columns([1, 2])
with bc1:
    st.metric("Chunks over test set", f"{n_chunks}")
    st.metric("Last chunk size", f"{rows[-1]['Size'] if rows else 0}")
with bc2:
    if not chunk_df.empty:
        st.dataframe(chunk_df, **STRETCH, hide_index=True)

# --------------------------------------------------------------------------
# Derived: timing
# --------------------------------------------------------------------------
st.subheader("Wall-clock time estimate")
total_seconds = n_test * seconds_per_candidate
tc1, tc2, tc3 = st.columns(3)
tc1.metric("Per candidate", f"{seconds_per_candidate:.1f}s")
tc2.metric("Total (test set)", f"{total_seconds/60:.1f} min")
tc3.metric("Total (hh:mm:ss)", f"{int(total_seconds)//3600:02d}:{(int(total_seconds)%3600)//60:02d}:{int(total_seconds)%60:02d}")

if not chunk_df.empty:
    st.bar_chart(chunk_df.set_index("Chunk")["Est. time (s)"])
st.caption(
    "The timing covers the evaluation run (a reference call for every candidate plus the routed call "
    "when a candidate has escalated criteria), from one observed run. Latency depends on the provider, "
    "the model and its rate limits, so adjust the basis for a different setup rather than treating "
    "it as a fixed rate."
)

# --------------------------------------------------------------------------
# Derived: escalation & cost projection
# --------------------------------------------------------------------------
st.subheader("Escalation & cost projection")

n_criterion_decisions = n_test * n_criteria
n_escalated_criteria = round(n_criterion_decisions * crit_esc_rate)
n_escalated_candidates = round(n_test * cand_esc_rate)
n_local_candidates = n_test - n_escalated_candidates

ec1, ec2, ec3, ec4 = st.columns(4)
ec1.metric("Criterion decisions (test)", f"{n_criterion_decisions}")
ec2.metric("Escalated criteria", f"{n_escalated_criteria}", f"{crit_esc_rate*100:.1f}%")
ec3.metric("Candidates with at least one escalation", f"{n_escalated_candidates}", f"{cand_esc_rate*100:.1f}%")
ec4.metric("Candidates resolved entirely locally", f"{n_local_candidates}")

if n_escalated_candidates > 0:
    per_esc = n_escalated_criteria / n_escalated_candidates
    if per_esc < 1.0 or per_esc > n_criteria:
        st.warning(
            f"The two rates imply {per_esc:.2f} escalated criteria per escalated candidate, which is "
            f"impossible (it must lie between 1 and {n_criteria}). Adjust one of the sliders."
        )
    else:
        st.caption(
            f"Implied escalated criteria per escalated candidate: {per_esc:.2f} "
            f"(study observed: 68 / 44 = 1.55)."
        )
elif n_escalated_criteria > 0:
    st.warning("Escalated criteria are non-zero but no candidate is escalated; adjust the sliders.")

routing_df = pd.DataFrame({
    "Outcome": ["At least one criterion escalated to the reference model", "Resolved entirely locally"],
    "Candidates": [n_escalated_candidates, n_local_candidates],
    "Share": [f"{cand_esc_rate*100:.1f}%", f"{(1-cand_esc_rate)*100:.1f}%"],
})
st.dataframe(routing_df, **STRETCH, hide_index=True)
st.caption(
    "Savings are larger at criterion level than at candidate level: in the study 77.7% of criterion "
    "decisions were resolved locally, yet 72.1% of candidates still needed one call, because a single "
    "uncertain criterion draws a candidate into the reference tier."
)

est_tokens_reference = n_test * tok_ref
est_tokens_createscore = n_escalated_candidates * tok_esc
savings_pct = (1 - est_tokens_createscore / est_tokens_reference) * 100 if est_tokens_reference else 0.0

cc1, cc2, cc3 = st.columns(3)
cc1.metric("Reference tokens: all criteria", f"{est_tokens_reference:,.0f}")
cc2.metric("Reference tokens: routed (projected)", f"{est_tokens_createscore:,.0f}")
cc3.metric("Projected reduction", f"{savings_pct:.1f}%")
st.caption(
    f"At the study's own mix and token costs, the measured reduction was {MEASURED_TOKEN_REDUCTION}% "
    "(121,174 tokens to 42,170). The projection counts reference-model tokens only; the local 8B model "
    "and the Bayesian network run locally and are not costed here. Token counts come from a single run "
    "without replication, so the projection is descriptive. The routing signal itself did not identify the "
    "local model's errors in the study, so these savings should be read as cost reduction, not as "
    "targeted error correction."
)

st.divider()
st.caption(
    "Constants in this planner (train fraction, escalation rates, token costs and timing) come from "
    "the corrected evaluation reported in Section 4 of the accompanying paper, and the harness defect "
    "that invalidated the earlier evaluation is described in Section 7.2. All are adjustable above "
    "for a different corpus, hardware or reference model."
)

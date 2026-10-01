# CreateScore Batch & Cost Planner

A small Streamlit app that accompanies the CreateScore preprint (domain-theory-informed Bayesian routing for LLM-based CV screening). Given a candidate count (or an uploaded job-order JSON), it computes the train/test split, a run schedule, a wall-clock estimate, and a projected escalation and reference-token comparison against an escalate-everything baseline.

It is a planning tool only. It performs no scoring, routing or posterior computation and makes no claim about agreement. Every default comes from one corrected evaluation on 61 synthetic test candidates (one run, one escalation-context policy) and can be changed in the sidebar. Escalation rates in particular depend on the corpus and the calibrated threshold, so treat the projections as planning estimates.

## Run

    pip install -r requirements.txt
    streamlit run createscore_planner_app.py

It opens at http://localhost:8501.

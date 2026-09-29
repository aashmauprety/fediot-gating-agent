"""FedIoT-Gate experimentation code.

This package implements the deterministic, testable parts of the
FedIoT-Gate design described in the paper:

  - Phase 0: federated self-supervised traffic encoder pretraining
  - Phase I: federated dual-head (device-type / attack-type) classifier training
  - Gating Agent: digest computation, Bayesian long-term trust prior,
    a rule-based gating policy (LLM-pluggable), zero-shot onboarding
    retrieval, and explainable intrusion triage
  - Attack simulations: untargeted, targeted, and trust-building (delayed)

IMPORTANT HONESTY NOTE (read before citing any number this code produces):
No LLM API key was available in this environment when this code was
written, so the "Gating Agent" reasoning steps that the paper describes as
LLM-based are implemented here as an explicit, documented RULE-BASED
STAND-IN (see fedgate/gating_agent.py, class `RuleBasedGatingPolicy`).
The digest computation, Bayesian trust updating, FedAvg aggregation, and
attack simulations ARE the real thing and produce real numbers. The
rule-based stand-in is a reasonable baseline for testing the pipeline and
for an ablation ("Gating Agent with rule-based policy" vs "plain FedAvg"),
but it is NOT the LLM-based Gating Agent the paper's Section VI-B
describes, and results obtained with it must be labeled as such in the
paper, not reported as validating the LLM design.

To use a real LLM for the Gating Agent, implement `LLMGatingPolicy` in
gating_agent.py (a small wrapper is stubbed out) and set the appropriate
API key in your environment.
"""

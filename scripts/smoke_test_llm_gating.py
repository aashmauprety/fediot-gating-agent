"""Minimal check that LLMGatingPolicy can actually talk to a local Ollama
server and produce a valid decision. Does not run any federated training
-- just exercises the prompt/call/parse path on synthetic digests, one
clearly-normal and one clearly-anomalous, so you can sanity-check the
model's behavior before wiring it into a full experiment.

Prerequisites:
    brew install ollama
    ollama serve                 # in a separate terminal, or as a service
    ollama pull llama3.1:8b      # one-time download, ~5GB
"""
import sys

sys.path.insert(0, "..")

from fedgate.gating_agent import BayesianTrust, Digest, LLMGatingPolicy, LLMGatingPolicyError


def main():
    policy = LLMGatingPolicy(model="llama3.1:8b")
    trust = BayesianTrust()

    # A gateway with several rounds of normal-looking history already banked.
    for _ in range(5):
        trust.update("gw_1", "attack-head", 1.0)

    round_norms = [10.2, 9.8, 10.5, 11.0, 9.6, 10.1, 41.7, 10.3, 9.9]

    print("=== Case 1: normal-looking update from a trusted gateway ===")
    d1 = Digest(client_id="gw_1", component="attack-head",
                update_norm=10.4, cosine_sim_to_prev_update=0.95, calib_loss_delta=0.01)
    try:
        decision = policy.decide(d1, round_norms, trust)
        print(f"  action={decision.action} score={decision.gating_score:.2f}")
        print(f"  rationale: {decision.rationale}")
    except LLMGatingPolicyError as e:
        print(f"  FAILED: {e}")
        return 1

    print("\n=== Case 2: anomalous update, no trust history ===")
    d2 = Digest(client_id="gw_7", component="attack-head",
                update_norm=41.7, cosine_sim_to_prev_update=-0.4, calib_loss_delta=0.6)
    try:
        decision = policy.decide(d2, round_norms, trust)
        print(f"  action={decision.action} score={decision.gating_score:.2f}")
        print(f"  rationale: {decision.rationale}")
    except LLMGatingPolicyError as e:
        print(f"  FAILED: {e}")
        return 1

    print("\nBoth calls succeeded -- LLMGatingPolicy is reachable and returning valid decisions.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

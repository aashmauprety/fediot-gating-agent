"""Convert results/*.json into LaTeX table fragments for the paper.
Run after run_all.py and run_onboarding_and_triage.py complete."""
import json
import os

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


def load(name):
    path = os.path.join(RESULTS_DIR, name)
    if not os.path.exists(path):
        print(f"[skip] {name} not found yet")
        return None
    with open(path) as f:
        return json.load(f)


def backbone_table():
    d = load("backbone_nbaiot_attack_head.json")
    if not d:
        return
    print("\n% --- Backbone (attack-type head, N-BaIoT, no attack) ---")
    print(f"% final macro-F1 = {d['final_macro_f1']:.4f} over {d['num_rounds']} rounds, "
          f"classes={d['classes']}, wall-clock={d['wall_clock_seconds']:.1f}s")
    print(r"\begin{table}[hbt!]")
    print(r"\centering")
    print(r"\caption{Attack-type head validation on real N-BaIoT (9 devices as federated clients, no attack, rule-based Gating Agent not active)}")
    print(r"\label{tab:nbaiot_backbone}")
    print(r"\begin{tabular}{|l|r|}")
    print(r"\hline")
    print(r"\textbf{Round} & \textbf{Macro-F1} \\ \hline")
    for row in d["rounds"]:
        print(f"{row['round']} & {row['macro_f1']:.3f} \\\\ \\hline")
    print(r"\end{tabular}")
    print(r"\end{table}")


def robustness_table():
    d = load("robustness_nbaiot.json")
    if not d:
        return
    print("\n% --- Robustness under attack (N-BaIoT) ---")
    print(r"\begin{table*}[hbt!]")
    print(r"\centering")
    print(r"\caption{Attack-type head macro-F1 under untargeted and targeted attacks on real N-BaIoT, plain FedAvg vs. trimmed-mean vs. Krum vs. rule-based Gating Agent stand-in}")
    print(r"\label{tab:nbaiot_robustness}")
    print(r"\begin{tabular}{|l|c|l|c|c|}")
    print(r"\hline")
    print(r"\textbf{Attack} & \textbf{$\alpha$ (clients)} & \textbf{Strategy} & \textbf{Macro-F1} & \textbf{Targeted success rate} \\ \hline")
    for row in d:
        tsr = "--" if row["targeted_success_rate"] is None else f"{row['targeted_success_rate']:.2f}"
        print(
            f"{row['attack_type']} & {row['alpha_clients']} & {row['strategy']} & "
            f"{row['final_macro_f1']:.3f} & {tsr} \\\\ \\hline"
        )
    print(r"\end{tabular}")
    print(r"\end{table*}")


def trustbuild_table():
    d = load("trustbuild_nbaiot.json")
    if not d:
        return
    print("\n% --- Trust-building attack (N-BaIoT) ---")
    print(r"\begin{table}[hbt!]")
    print(r"\centering")
    print(r"\caption{Trust-building (delayed) attack: macro-F1 vs. delay $D$, plain FedAvg vs. Bayesian-informed rule-based Gating Agent stand-in}")
    print(r"\label{tab:nbaiot_trustbuild}")
    print(r"\begin{tabular}{|c|l|c|c|}")
    print(r"\hline")
    print(r"\textbf{Delay $D$} & \textbf{Strategy} & \textbf{Macro-F1} & \textbf{Targeted success rate} \\ \hline")
    for row in d:
        tsr = "--" if row["targeted_success_rate"] is None else f"{row['targeted_success_rate']:.2f}"
        print(f"{row['delay_rounds']} & {row['strategy']} & {row['final_macro_f1']:.3f} & {tsr} \\\\ \\hline")
    print(r"\end{tabular}")
    print(r"\end{table}")


def onboarding_summary():
    d = load("onboarding_nbaiot_proxy.json")
    if not d:
        return
    print("\n% --- Onboarding proxy validation (N-BaIoT devices) ---")
    print(f"% Held-out device: {d['held_out_device']}")
    print(f"% Unseen-device query -> action={d['unseen_device_query']['action']}, "
          f"correct_behavior={d['unseen_device_query']['correct_behavior']}")
    print(f"% Known-device query -> action={d['known_device_held_out_split_query']['action']}, "
          f"correct_behavior={d['known_device_held_out_split_query']['correct_behavior']}")


def triage_summary():
    d = load("triage_nbaiot.json")
    if not d:
        return
    print("\n% --- Triage validation summary (N-BaIoT) ---")
    for action, counts in d["summary"].items():
        total = counts["correct"] + counts["incorrect"]
        print(f"% {action}: {counts['correct']}/{total} correct ({counts['correct']/total:.0%})")


if __name__ == "__main__":
    backbone_table()
    robustness_table()
    trustbuild_table()
    onboarding_summary()
    triage_summary()

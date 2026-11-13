"""Diabetes risk check for any person: score, explanation and what to do next.

Uses two things the notebook saves in results/:
  points_card.json        - the 5-item paper card (works with only a few answers)
  screening_model.joblib  - the full logistic-regression model with per-sex thresholds

Ways to run it (from the repository root)
  python tool/predict.py                    ask questions one by one
  python tool/predict.py --demo 3           make up 3 random people and check them
  python tool/predict.py --age 52 --sex male --height 165 --weight 80 --family-history yes --high-bp yes
  python tool/predict.py --csv people.csv   check everyone in a CSV (see --template)
  python tool/predict.py --template         write tool/example_people.csv with the right columns

Any answer you don't know can be left out. This is a screening aid built on US data
(NHANES 2017-2020), not a diagnosis.
"""
import argparse
import csv
import json
import math
import random
import sys
import warnings
from pathlib import Path

HERE = Path(__file__).parent          # tool/
ROOT = HERE.parent                    # repository root
RESULTS = ROOT / "results"
CARD = json.loads((RESULTS / "points_card.json").read_text(encoding="utf-8"))

COLUMNS = ["name", "age", "sex", "height_cm", "weight_kg", "waist_cm", "systolic_bp", "diastolic_bp",
           "already_diagnosed", "pregnant", "family_history", "high_bp", "high_cholesterol", "smoked",
           "exercise", "hard_work", "diabetes_in_pregnancy", "symptoms"]
HELP = {
    "age": "age in years", "sex": "male / female", "height_cm": "height in cm", "weight_kg": "weight in kg",
    "waist_cm": "waist in cm (at the navel)", "systolic_bp": "blood pressure, top number",
    "diastolic_bp": "blood pressure, bottom number",
    "already_diagnosed": "ever told by a doctor you have diabetes? yes / no / borderline",
    "pregnant": "pregnant now? yes / no",
    "family_history": "parent, brother or sister with diabetes? yes / no",
    "high_bp": "ever told by a doctor you have high blood pressure? yes / no",
    "high_cholesterol": "ever told you have high cholesterol? yes / no",
    "smoked": "smoked 100+ cigarettes/bidis in your life? yes / no",
    "exercise": "regular exercise: none / moderate (brisk walk, yoga) / vigorous (running, sport)",
    "hard_work": "hard physical work most days (job or housework)? yes / no",
    "diabetes_in_pregnancy": "women: ever had diabetes during a pregnancy? yes / no",
    "symptoms": "unusual thirst, passing urine often, weight loss or tiredness? yes / no",
}
YES, NO = {"yes", "y", "1", "true", "haan", "ha"}, {"no", "n", "0", "false", "nahi"}


# ------------------------------------------------------------------ input cleaning
def num(v):
    try:
        return float(v) if v not in (None, "") else None
    except ValueError:
        return None


def yes_no(v):
    v = (str(v).strip().lower() if v is not None else "")
    return True if v in YES else False if v in NO else None


def clean(raw):
    """Raw strings -> typed person dict (None = unknown)."""
    p = {"name": (raw.get("name") or "").strip() or "Person"}
    for k in ["age", "height_cm", "weight_kg", "waist_cm", "systolic_bp", "diastolic_bp"]:
        p[k] = num(raw.get(k))
    sex = str(raw.get("sex") or "").strip().lower()
    p["sex"] = "male" if sex in {"m", "male", "man", "purush"} else "female" if sex in {"f", "female", "woman", "mahila"} else None
    diag = str(raw.get("already_diagnosed") or "").strip().lower()
    p["already_diagnosed"] = "borderline" if diag.startswith("border") else yes_no(diag)
    ex = str(raw.get("exercise") or "").strip().lower()
    p["exercise"] = ("vigorous" if ex.startswith("vig") else "moderate" if ex.startswith(("mod", "yes", "y"))
                     else "none" if ex.startswith(("no", "n")) else None)
    for k in ["pregnant", "family_history", "high_bp", "high_cholesterol", "smoked", "hard_work",
              "diabetes_in_pregnancy", "symptoms"]:
        p[k] = yes_no(raw.get(k))
    if p["sex"] == "male":
        p["diabetes_in_pregnancy"], p["pregnant"] = False, False
    return p


def bmi_of(p):
    if p["height_cm"] and p["weight_kg"]:
        return p["weight_kg"] / (p["height_cm"] / 100) ** 2
    return None


# ------------------------------------------------------------------ scoring
def card_score(p, bmi):
    """Points-card score with a line-by-line breakdown. Unknown yes/no answers count as 'no'."""
    parts, missing = [], []
    for var, value in [("age", p["age"]), ("bmi", bmi)]:
        spec = CARD["bands"][var]
        if value is None:
            missing.append(var)
            continue
        for i, pts in enumerate(spec["points"]):
            if spec["edges"][i] <= value < spec["edges"][i + 1]:
                parts.append((f"{var.upper() if var == 'bmi' else 'Age'} {spec['labels'][i]}", pts))
                break
    inactive = None if p["exercise"] is None and p["hard_work"] is None else \
        (p["exercise"] in (None, "none")) and not p["hard_work"]
    flags = {"family_history_diabetes": (p["family_history"], "Family history of diabetes"),
             "told_high_bp": (p["high_bp"], "Told high blood pressure"),
             "inactive": (inactive, "No regular exercise and no hard physical work")}
    for key, spec in CARD["yes_no"].items():
        answer, label = flags[key]
        if answer:
            parts.append((label, spec["points"]))
    return sum(pts for _, pts in parts), parts, missing


def load_model():
    path = RESULTS / "screening_model.joblib"
    if not path.exists():
        return None, "results/screening_model.joblib not found (run the notebook once to create it)"
    try:
        import joblib
        import sklearn
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            bundle = joblib.load(path)
        if bundle["versions"]["scikit-learn"] != sklearn.__version__:
            note = (f"model saved with scikit-learn {bundle['versions']['scikit-learn']}, you have "
                    f"{sklearn.__version__}; if results look odd, re-run the notebook")
            return bundle, note
        return bundle, None
    except Exception as e:  # missing packages or incompatible pickle: card still works
        return None, f"could not load the model ({e}); using the points card only"


def model_risk(bundle, p, bmi):
    import pandas as pd
    nan = float("nan")

    def b(v):
        return nan if v is None else float(v)

    ex = p["exercise"]
    row = {
        "age": b(p["age"]), "female": b(None if p["sex"] is None else p["sex"] == "female"), "bmi": b(bmi),
        "waist_cm": b(p["waist_cm"]),
        "waist_height_ratio": b(p["waist_cm"] / p["height_cm"] if p["waist_cm"] and p["height_cm"] else None),
        "weight_kg": b(p["weight_kg"]), "height_cm": b(p["height_cm"]),
        "systolic_bp": b(p["systolic_bp"]), "diastolic_bp": b(p["diastolic_bp"]),
        "family_history_diabetes": b(p["family_history"]), "told_high_bp": b(p["high_bp"]),
        "told_high_cholesterol": b(p["high_cholesterol"]), "ever_smoked_100_cig": b(p["smoked"]),
        "vigorous_activity": b(None if ex is None else ex == "vigorous"),
        "moderate_activity": b(None if ex is None else ex == "moderate"),
        "exercise": b(None if ex is None else ex != "none"),
        "strenuous_work": b(p["hard_work"]), "gestational_diabetes": b(p["diabetes_in_pregnancy"]),
    }
    X = pd.DataFrame([row])[bundle["features"]]

    def predict(model):
        # Fill blanks with the training medians ourselves. Left as NaN, the model's "was missing" flags would kick
        # in, and those learned survey quirks (who skipped which question), not anything about this person.
        medians = pd.Series(model.steps[0][1].statistics_[:len(bundle["features"])], index=bundle["features"])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return float(model.predict_proba(X.fillna(medians))[0, 1])

    risk, raised = predict(bundle["model_diabetes"]), predict(bundle["model_raised_sugar"])
    thr = bundle["threshold_by_sex"].get(p["sex"] or "", min(bundle["threshold_by_sex"].values()))
    return risk, raised, thr


# ------------------------------------------------------------------ advice
def assess(p, bundle):
    """Returns a dict with the verdict, numbers and a numbered action list."""
    bmi = bmi_of(p)
    out = {"name": p["name"], "bmi": bmi, "actions": [], "notes": []}
    bp_high = (p["systolic_bp"] or 0) >= 140 or (p["diastolic_bp"] or 0) >= 90
    bp_very_high = (p["systolic_bp"] or 0) >= 180 or (p["diastolic_bp"] or 0) >= 120

    if p["age"] is None:
        out["verdict"] = "CANNOT ASSESS"
        out["actions"].append("Age is needed for every risk score.")
        return out
    if p["age"] < 20:
        out["verdict"] = "OUT OF SCOPE"
        out["actions"].append("This check is for adults 20+. For children and teens with symptoms or obesity, see a doctor.")
        return out
    if p["pregnant"]:
        out["verdict"] = "OUT OF SCOPE (PREGNANT)"
        out["actions"].append("Pregnant women need the antenatal glucose test (OGTT, usually at 24-28 weeks). Ask the ANC clinic or ASHA.")
        return out

    if p["already_diagnosed"] is True:
        out["verdict"] = "ALREADY DIAGNOSED: FOLLOW THE CARE CHECKLIST"
        out["actions"] += [
            "HbA1c every 3-6 months. Ask your doctor for your personal target (often below 7%).",
            "Blood pressure at every visit; aim below 130/80 unless your doctor says otherwise.",
            "Once a year: urine albumin-creatinine ratio + blood creatinine (eGFR) to catch kidney damage early.",
            "Once a year: dilated eye check (retina) and a foot check for numbness or wounds.",
            "Cholesterol (lipid profile) once a year.",
        ]
        if bp_very_high:
            out["actions"].insert(0, "Blood pressure reading is dangerously high: see a doctor TODAY.")
        elif bp_high:
            out["actions"].insert(0, "Today's blood pressure is high (140/90 or more): get it re-checked within 2 weeks.")
        return out

    score, parts, missing = card_score(p, bmi)
    out.update({"card_score": score, "card_parts": parts, "card_max": CARD["max_score"], "card_cutoff": CARD["cutoff"]})
    card_flag = score >= CARD["cutoff"]
    if missing:
        out["notes"].append(f"Card is incomplete without: {', '.join(missing)} (height and weight are needed for BMI).")

    model_flag = None
    core_missing = [k for k, v in [("sex", p["sex"]), ("height", p["height_cm"]), ("weight", p["weight_kg"])] if v is None]
    if bundle is not None and core_missing:
        out["notes"].append(f"Full model skipped: it needs {', '.join(core_missing)} (with too many blanks it guesses badly).")
        bundle = None
    if bundle is not None:
        risk, raised, thr = model_risk(bundle, p, bmi)
        model_flag = risk >= thr
        out.update({"model_risk": risk, "model_raised_sugar": raised, "model_threshold": thr})

    # The full model with per-sex thresholds is the rule the notebook evaluated (equal sensitivity for men and
    # women). The paper card decides only when the model can't be loaded. Disagreements are shown, not hidden.
    screen_flag = card_flag if model_flag is None else model_flag
    out["decided_by"] = "points card" if model_flag is None else "full model"
    if model_flag is not None and model_flag != card_flag:
        out["notes"].append(f"The paper card says {'higher' if card_flag else 'lower'} risk; the full model, which uses more "
                            f"answers and a separate cut-off for {p['sex'] or 'each sex'}, decides here.")
    refer = bool(p["symptoms"]) or p["already_diagnosed"] == "borderline" or screen_flag
    if p["symptoms"]:
        out["verdict"] = "TEST NOW: SYMPTOMS"
        out["actions"].append("Symptoms of high sugar: get a blood glucose / HbA1c test within a few days, whatever the score.")
    elif p["already_diagnosed"] == "borderline":
        out["verdict"] = "TEST NOW: TOLD BORDERLINE BEFORE"
        out["actions"].append("You were told 'borderline' before: get an HbA1c test now and then every year.")
    elif refer:
        out["verdict"] = "HIGHER RISK: GET AN HbA1c TEST"
        out["actions"].append("Get an HbA1c blood test (no fasting needed) at a lab, PHC or health camp within the next month.")
    elif missing and score + sum(max(CARD["bands"][m]["points"]) for m in missing) >= CARD["cutoff"]:
        out["verdict"] = "NOT ENOUGH ANSWERS"
        out["actions"].append(f"Measure {' and '.join('height and weight' if m == 'bmi' else m for m in missing)} and run the "
                              f"check again: the missing item could push the score to {CARD['cutoff']} or more.")
    else:
        out["verdict"] = "LOWER RISK"
        out["actions"].append("No test needed now. Repeat this check every 3 years, or sooner if symptoms appear.")

    if refer:
        out["actions"].append("HbA1c result: below 5.7% normal (re-check in 3 years) · 5.7-6.4% prediabetes (lifestyle change, "
                              "re-test yearly) · 6.5% or more possible diabetes (see a doctor; a second test confirms it).")
    if bp_very_high:
        out["actions"].insert(0, "Blood pressure reading is dangerously high: see a doctor TODAY.")
    elif bp_high:
        out["actions"].append("Today's blood pressure is high (140/90 or more): get it re-checked within 2 weeks.")
    if bmi is not None and bmi >= 23:
        target = 23 * (p["height_cm"] / 100) ** 2
        out["actions"].append(f"BMI {bmi:.1f} is above the healthy range for South Asians (18.5-22.9). Losing even 5% of body weight "
                              f"(about {0.05 * p['weight_kg']:.0f} kg) lowers diabetes risk; a healthy weight for this height is up to {target:.0f} kg.")
    if p["waist_cm"] and p["height_cm"] and p["waist_cm"] / p["height_cm"] >= 0.5:
        out["actions"].append(f"Waist {p['waist_cm']:.0f} cm is more than half the height: belly fat raises risk. Aim for a waist "
                              f"under {p['height_cm'] / 2:.0f} cm.")
    if p["exercise"] == "none" and not p["hard_work"]:
        out["actions"].append("Start with a 30-minute brisk walk on 5 days a week (150 minutes a week).")
    elif p["exercise"] is None and p["hard_work"] is None:
        out["actions"].append("If you aren't already active: a 30-minute brisk walk on 5 days a week lowers risk.")
    if p["smoked"]:
        out["actions"].append("If you still smoke, quitting lowers diabetes and heart risk (free help: national tobacco quitline 1800-112-356).")
    return out


# ------------------------------------------------------------------ printing
def show(p, r):
    line = "=" * 64
    print(line)
    bits = [f"{p['age']:.0f} y" if p["age"] is not None else None, p["sex"],
            f"BMI {r['bmi']:.1f}" if r.get("bmi") else None]
    print(f"{r['name']}  ({', '.join(b for b in bits if b)})")
    print(line)
    print(f"RESULT: {r['verdict']}\n")
    if "card_score" in r:
        print(f"Points card: {r['card_score']} / {r['card_max']}  (test if {r['card_cutoff']} or more)")
        for label, pts in r["card_parts"]:
            print(f"   +{pts}  {label}")
    if "model_risk" in r:
        print(f"Full model: {r['model_risk']:.1%} chance of undiagnosed diabetes "
              f"(test if {r['model_threshold']:.1%} or more for {p['sex'] or 'unknown sex'})")
        print(f"            {r['model_raised_sugar']:.0%} chance of prediabetes or diabetes (HbA1c 5.7%+)")
    if "decided_by" in r:
        print(f"Screening decision made by: {r['decided_by']}")
    if "card_score" in r:
        print()
    print("WHAT TO DO")
    for i, a in enumerate(r["actions"], 1):
        print(f" {i}. {a}")
    for n in r["notes"]:
        print(f" note: {n}")
    print()


DISCLAIMER = ("Screening aid, not a diagnosis. Built on US survey data (NHANES 2017-2020) and not yet validated in India. "
              "The risk percentages are US rates and run high on newer data; use them to rank, not as exact chances.")


# ------------------------------------------------------------------ random people
def random_person(rng, i):
    sex = rng.choice(["male", "female"])
    age = rng.randint(20, 80)
    height = rng.gauss(168 if sex == "male" else 155, 7)
    bmi = min(max(rng.lognormvariate(math.log(26), 0.18), 16), 48)
    weight = bmi * (height / 100) ** 2
    waist = height * min(max(rng.gauss(0.40 + 0.0065 * bmi, 0.03), 0.38), 0.85)
    sbp = rng.gauss(105 + 0.5 * age + 0.4 * bmi, 14)
    ex = rng.choices(["none", "moderate", "vigorous"], [0.45, 0.35, 0.20])[0]
    return {
        "name": f"Random person {i}", "age": age, "sex": sex, "height_cm": round(height, 1), "weight_kg": round(weight, 1),
        "waist_cm": round(waist, 1), "systolic_bp": round(sbp), "diastolic_bp": round(sbp * 0.62 + rng.gauss(0, 6)),
        "already_diagnosed": rng.choices(["no", "yes", "borderline"], [0.86, 0.10, 0.04])[0],
        "pregnant": "no", "family_history": "yes" if rng.random() < 0.35 else "no",
        "high_bp": "yes" if rng.random() < 0.15 + 0.006 * (age - 20) else "no",
        "high_cholesterol": "yes" if rng.random() < 0.30 else "no", "smoked": "yes" if rng.random() < 0.35 else "no",
        "exercise": ex, "hard_work": "yes" if rng.random() < 0.3 else "no",
        "diabetes_in_pregnancy": ("yes" if rng.random() < 0.05 else "no") if sex == "female" else "no",
        "symptoms": "yes" if rng.random() < 0.05 else "no",
    }


# ------------------------------------------------------------------ main
def ask():
    print("Answer each question (press Enter to skip if you don't know).\n")
    raw = {}
    for k in COLUMNS:
        if k == "diabetes_in_pregnancy" and str(raw.get("sex", "")).lower().startswith("m"):
            continue
        if k == "pregnant" and str(raw.get("sex", "")).lower().startswith("m"):
            continue
        raw[k] = input(f"{HELP.get(k, 'name (optional)')}: ").strip()
    return raw


def main():
    ap = argparse.ArgumentParser(description="Diabetes risk check: score + what to do next.",
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__.split("Ways to run it")[1])
    ap.add_argument("--demo", type=int, metavar="N", help="check N made-up random people")
    ap.add_argument("--seed", type=int, default=None, help="random seed for --demo (same people each time)")
    ap.add_argument("--csv", metavar="FILE", help="check everyone in a CSV file")
    ap.add_argument("--out", metavar="FILE", help="with --csv or --demo: also save results to this CSV")
    ap.add_argument("--template", action="store_true", help="write example_people.csv and exit")
    for k in COLUMNS:
        flag = "--" + {"height_cm": "height", "weight_kg": "weight", "waist_cm": "waist", "systolic_bp": "sbp",
                       "diastolic_bp": "dbp", "high_bp": "high-bp"}.get(k, k.replace("_", "-"))
        ap.add_argument(flag, dest=k, help=HELP.get(k, "name shown in the output"))
    args = ap.parse_args()

    if args.template:
        rng = random.Random(7)
        with open(HERE / "example_people.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=COLUMNS)
            w.writeheader()
            for i in range(1, 6):
                w.writerow(random_person(rng, i))
        print("wrote tool/example_people.csv - edit it (blank cells are fine) and run: "
              "python tool/predict.py --csv tool/example_people.csv")
        return

    if args.demo:
        rng = random.Random(args.seed)
        people = [random_person(rng, i) for i in range(1, args.demo + 1)]
    elif args.csv:
        with open(args.csv, newline="", encoding="utf-8-sig") as f:
            people = list(csv.DictReader(f))
    elif any(getattr(args, k) is not None for k in COLUMNS):
        people = [{k: getattr(args, k) for k in COLUMNS}]
    else:
        people = [ask()]

    bundle, note = load_model()
    if note:
        print(f"[{note}]\n")
    rows = []
    for raw in people:
        p = clean(raw)
        r = assess(p, bundle)
        show(p, r)
        rows.append({"name": r["name"], "verdict": r["verdict"], "card_score": r.get("card_score", ""),
                     "model_risk": round(r["model_risk"], 4) if "model_risk" in r else "",
                     "what_to_do": " | ".join(r["actions"])})
    if len(rows) > 1:
        print("SUMMARY")
        for row in rows:
            print(f"  {row['name'][:28]:28s} {row['verdict']}")
        print()
    if args.out:
        with open(args.out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"saved results to {args.out}")
    print(DISCLAIMER)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()


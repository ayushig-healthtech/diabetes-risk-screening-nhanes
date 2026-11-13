# Finding hidden diabetes with five questions and a weighing scale

**About 1 in 6 US adults with diabetes don't know they have it. Among those hidden cases, 27% (2017–20) to 36% (2021–23) had not had a blood-sugar test in the previous three years.** This project asks whether a short questionnaire, with no blood test, can find them, and whether machine learning does that better than the paper risk scores doctors already use. It ends in [`tool/predict.py`](tool/predict.py), which takes any person's answers and says what to do next.

All data is public (CDC NHANES). Every number below is out-of-fold (each person scored by a model that never saw them) or comes from a later survey cycle, and every comparison with the existing scores is paired, with a 95% CI.

## Key findings

| | Result |
|---|---|
| US adults with undiagnosed diabetes (HbA1c ≥ 6.5, 2017–20, survey-weighted) | **2.4%** (95% CI 2.0–2.8%) |
| Share of all diabetes that is undiagnosed | **16.8%** (14.5–19.1%) |
| Undiagnosed cases with **no blood test in the past 3 years** | **26.6%** (20.7–32.5%) in 2017–20 → **36.3%** (28.3–44.4%) in 2021–23 |
| People in the prediabetes range who were never told | **77.0%** (73.5–80.6%) |

**Does ML beat the paper scores?** (target: undiagnosed diabetes, 6,504 adults, 227 cases)

| Score | AUROC (95% CI) | vs ADA test (paired) |
|---|---|---|
| IDRS (Indian Diabetes Risk Score, 4 items) | 0.697 (0.663–0.727) | −0.044 (−0.068 to −0.019) |
| ADA risk test (7 items) | 0.741 (0.714–0.768) | reference |
| Logistic regression (18 inputs) | 0.759 (0.729–0.786) | +0.018 (−0.005 to +0.039) |
| Gradient boosting (18 inputs) | 0.725 (0.692–0.758) | −0.016 (−0.042 to +0.010) |
| **Points card (this project, 5 items)** | **0.741** (0.709–0.771)* | −0.004 (−0.024 to +0.014) |

\*on the 6,217 people with height, weight and waist measured.

- **ML does not clearly beat the 7-question ADA test** for finding undiagnosed diabetes. At the ADA test's sensitivity (83%), logistic regression would send 47.6% of people for a test against ADA's 46.3%. That is an honest, useful finding: the simple test is already close to what these inputs allow.
- For the broader target (HbA1c ≥ 5.7, prediabetes or diabetes), ML is slightly but significantly better (+0.010 AUROC, CI +0.003 to +0.016).
- **The IDRS is significantly weaker than the ADA test on US data.** It flags 60% of people to catch 86% of cases.
- **The 5-item points card matches the ADA test with two fewer questions** (sex and diabetes in pregnancy added nothing once age, BMI, family history, blood pressure and activity were known).
- **BMI beat waist-to-height ratio** for the card (0.741 vs 0.714), against the usual advice for Asian body types. This is reported rather than hidden.
- **Module 2 (poor sugar control):** with treatment answers AUROC is 0.714; **without them it drops to 0.582**. Most of the apparent skill was the shortcut "on insulin means uncontrolled".
- **Temporal validation:** a model trained on 2017–20 ranks risk about as well on 2021–23 (logistic regression AUROC 0.750 out-of-fold → 0.741 on new data; gradient boosting 0.712 → 0.752).
  - **Clean threshold test:** a threshold frozen on 2017–20 to catch 83% of cases **caught 82.8% in 2021–23**, so the cut-off transfers.
  - **Over-prediction:** the model predicts too high in the new data (mean 3.4% predicted vs 2.5% observed), so risks should be multiplied by about 0.70. The ranking is fine, but the risk percentages need a local re-fit of the intercept.

### Fairness (logistic regression vs ADA, sensitivity at matched thresholds)
- **Young adults (20–39):** the model catches 71% of hidden cases, against 39% for the ADA test, which gives most points for age.
- **Men:** with one threshold for everyone, the model catches only 72% of cases in men (ADA: 83%). **With a separate threshold per sex, it catches 84% in both men and women, and tests 45.9% of people in total, slightly fewer than ADA's 46.3%.**
- **NH Asian adults (n = 776, 36 cases):** the ADA test with Asian BMI cut-offs catches 81% of cases (standard cut-offs: 72%). Asians with undiagnosed diabetes had a mean BMI of 28.8, against 34.9 for others.

## The points card

| Item | Points |
|---|---|
| Age 20–34 / 35–44 / 45–54 / 55–64 / 65+ | 0 / 4 / 4 / 5 / 5 |
| BMI < 25 / 25–29.9 / 30–34.9 / 35–39.9 / 40+ | 0 / 1 / 4 / 4 / 5 |
| Parent, brother or sister with diabetes | 1 |
| Ever told you have high blood pressure | 2 |
| No regular exercise **and** no physically hard work | 1 |
| **7 or more (out of 14) → get an HbA1c test** | catches 90% of cases; 52% of people tested |

Two points means roughly double the odds. The points come from a logistic model, and they never go down as age or BMI rises.

## What's in this repository

```
├── Diabetes_Care_Journey_NHANES.ipynb   the whole analysis (downloads the data itself, runs in Colab)
├── tool/                                use the result: check people, print the card
│   ├── predict.py
│   ├── make_card.py
│   └── example_people.csv
├── results/                             outputs saved by the notebook
├── docs/                                model card and next steps
├── scripts/                             optional raw-data downloader
├── requirements.txt                     everything, to re-run the notebook
└── requirements-tool.txt                just enough to use tool/
```

| File | What it is |
|---|---|
| [`Diabetes_Care_Journey_NHANES.ipynb`](Diabetes_Care_Journey_NHANES.ipynb) | The whole analysis, from top to bottom. It downloads the data itself and runs in Colab. |
| [`tool/predict.py`](tool/predict.py) | **Check anyone:** scores a person, a CSV of people, or random made-up people, and prints what to do next. |
| [`tool/make_card.py`](tool/make_card.py) | Builds `results/points_card_print.html`, a one-page printable card in English and Hindi with a BMI lookup table. Print it on A4 or save it as a PDF. |
| [`tool/example_people.csv`](tool/example_people.csv) | A sample input file for `predict.py --csv` (made-up people; blank cells are fine). |
| `results/screening_model.joblib` | The trained model with per-sex thresholds, saved by the notebook and used by `predict.py`. |
| [`results/points_card.json`](results/points_card.json) | The paper card, saved by the notebook and used by `predict.py` and `make_card.py`. |
| [`results/metrics.json`](results/metrics.json) | Every number in this README. |
| `results/*.png` | Calibration and SHAP plots. |
| [`docs/MODEL_CARD.md`](docs/MODEL_CARD.md) | Intended use, limits, subgroup performance. |

**Re-run the analysis:**

```bash
pip install -r requirements.txt
jupyter notebook Diabetes_Care_Journey_NHANES.ipynb   # or upload it to Colab and Run all
```

## Check a person

`tool/predict.py` works straight away, because the trained model and card are already in `results/`. Run these from the repository folder:

```bash
pip install -r requirements-tool.txt
python tool/predict.py                    # answers questions one by one
python tool/predict.py --demo 3           # makes up 3 random people and checks them
python tool/predict.py --age 52 --sex male --height 165 --weight 80 --waist 98 --sbp 150 --dbp 95 --family-history yes --high-bp yes --exercise none --hard-work no
python tool/predict.py --csv tool/example_people.csv --out results_people.csv
python tool/make_card.py                  # rebuilds the printable card
```

Any answer can be left out. Example output:

```
RESULT: HIGHER RISK: GET AN HbA1c TEST

Points card: 9 / 14  (test if 7 or more)
   +4  Age 45-54
   +1  BMI 25-29.9
   +1  Family history of diabetes
   +2  Told high blood pressure
   +1  No regular exercise and no hard physical work
Full model: 10.7% chance of undiagnosed diabetes (test if 1.8% or more for male)
            53% chance of prediabetes or diabetes (HbA1c 5.7%+)
Screening decision made by: full model

WHAT TO DO
 1. Get an HbA1c blood test (no fasting needed) at a lab, PHC or health camp within the next month.
 2. HbA1c result: below 5.7% normal (re-check in 3 years) · 5.7-6.4% prediabetes ... · 6.5% or more possible diabetes ...
 3. Today's blood pressure is high (140/90 or more): get it re-checked within 2 weeks.
 4. BMI 29.4 is above the healthy range for South Asians (18.5-22.9). Losing even 5% of body weight (about 4 kg) ...
 ...
```

**How it decides**
- **Who it screens:** non-pregnant adults aged 20+ not already diagnosed with diabetes. Each person gets the paper card score and the full model's risk; the **full model with its per-sex threshold** makes the call, because that is the rule validated in the notebook (equal sensitivity for men and women). If the two disagree, it says so. Without sex, height and weight it falls back to the card.
- **Test straight away:** symptoms (thirst, frequent urination, weight loss, tiredness) or an earlier "borderline" result mean "test now", whatever the score.
- **Already diagnosed:** a care checklist instead of a score: HbA1c every 3–6 months, yearly kidney, eye and foot checks, blood-pressure and cholesterol targets.
- **Pregnant or under 20:** pointed to the right test or service instead.
- **Extra advice:** high blood pressure readings, a BMI above the South Asian healthy range, a large waist, inactivity and smoking each add a specific action.

## Method notes
- **Data:** NHANES 2017–March 2020 (training) and Aug 2021–Aug 2023 (temporal test). Non-pregnant adults aged 20+.
- **Labels:** HbA1c < 5.7 normal, 5.7–6.4 prediabetes, ≥ 6.5 diabetes. People who answered "borderline" (n = 262) are excluded from the screening table. Lab values are used only for labels, never as inputs.
- **Population estimates** use exam weights with Taylor-linearised design-based CIs (strata + PSU).
- **Models** are not class-weighted, so their risks stay calibrated. The threshold is chosen to match the ADA test's sensitivity.
- **IDRS adaptation:** NHANES asks about "any close relative" with diabetes, so a yes scores 10 (IDRS gives 10 for one parent and 20 for both).

## Limitations
A single HbA1c test is a research label, not a diagnosis. The data is from the US, and "NH Asian" mixes East, South and Southeast Asians (n = 776). The section 5–7 thresholds were chosen on the same out-of-fold predictions they are reported on; the frozen-threshold test on 2021–23 is the clean estimate. 2021–23 lacks the family-history and activity questions, so the temporal test uses a reduced model. **The next step is validation on Indian data (NFHS-5, which has random glucose rather than HbA1c).**

## Credits
Built with help from an AI coding assistant (Claude Code). Study design, decisions and checking of the results are my own. Data: CDC National Health and Nutrition Examination Survey (NHANES), public domain.

## Licence
Code is released under the MIT licence. This is a screening aid for research and education, not a medical device or a diagnosis.

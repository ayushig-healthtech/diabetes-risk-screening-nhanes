"""Turn results/points_card.json into a printable one-page bilingual card (English + Hindi).

Usage (from the repository root):  python tool/make_card.py   -> results/points_card_print.html
Open it in a browser and print (Ctrl+P) on A4, or "Save as PDF".
"""
import html
import json
from pathlib import Path

ROOT = Path(__file__).parent.parent   # repository root (this file is in tool/)
card = json.loads((ROOT / "results" / "points_card.json").read_text(encoding="utf-8"))
assert card["body_measure"] == "bmi", "this layout expects the BMI version of the card"

HI = {
    "age": "उम्र (साल)",
    "bmi": "BMI (वज़न किलो ÷ लंबाई मीटर²)",
    "family_history_diabetes": "माता-पिता, भाई या बहन को डायबिटीज़",
    "told_high_bp": "डॉक्टर ने कभी हाई ब्लड प्रेशर बताया",
    "inactive": "न नियमित व्यायाम, न भारी शारीरिक मेहनत",
    "male": "पुरुष",
}
EN = {
    "age": "Age (years)",
    "bmi": "BMI (weight kg ÷ height m²)",
    "family_history_diabetes": "Parent, brother or sister has diabetes",
    "told_high_bp": "Ever told by a doctor: high blood pressure",
    "inactive": "No regular exercise AND no hard physical work",
    "male": "Male",
}


def esc(s):
    return html.escape(str(s))


rows = []
for var, spec in card["bands"].items():
    cells = "".join(f"<td><span class='lab'>{esc(l)}</span><b>{p}</b></td>"
                    for l, p in zip(spec["labels"], spec["points"]))
    rows.append(f"<tr><th>{esc(EN[var])}<br><span class='hi'>{esc(HI[var])}</span></th>{cells}<td class='box'></td></tr>")
n_band_cols = max(len(s["labels"]) for s in card["bands"].values())
for key, spec in card["yes_no"].items():
    rows.append(f"<tr><th>{esc(EN[key])}<br><span class='hi'>{esc(HI[key])}</span></th>"
                f"<td colspan='{n_band_cols}' class='yes'>Yes / हाँ = <b>{spec['points']}</b> &nbsp;·&nbsp; No / नहीं = <b>0</b></td>"
                f"<td class='box'></td></tr>")

# BMI lookup so nobody has to divide on paper: weight (kg) at which each BMI band starts, by height
bmi_edges = card["bands"]["bmi"]["edges"][1:-1]
bmi_labels = card["bands"]["bmi"]["labels"][1:]
lookup_head = "".join(f"<th>{esc(l)}<br><span class='hi'>from kg</span></th>" for l in bmi_labels)
lookup_rows = "".join(
    f"<tr><td><b>{h}</b></td>" + "".join(f"<td>{e * (h / 100) ** 2:.0f}</td>" for e in bmi_edges) + "</tr>"
    for h in range(145, 186, 5))
lookup = (f"<p><b>BMI without a calculator / बिना कैलकुलेटर BMI:</b> find the height row; the person is in the "
          f"highest column whose weight they reach. / लंबाई की पंक्ति देखें; जिस कॉलम का वज़न पूरा हो, वही BMI वर्ग।</p>"
          f"<table class='lookup'><tr><th>Height cm / लंबाई</th>{lookup_head}</tr>{lookup_rows}</table>")

page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>Diabetes risk card</title>
<style>
  @page {{ size: A4; margin: 14mm; }}
  body {{ font-family: "Noto Sans", "Nirmala UI", "Mangal", Arial, sans-serif; color: #111; background: #fff;
         max-width: 190mm; margin: 0 auto; padding: 8px; }}
  h1 {{ font-size: 20px; margin: 0 0 2px; }} h1 span {{ font-weight: normal; }}
  p {{ margin: 4px 0; font-size: 12.5px; }}
  table {{ width: 100%; border-collapse: collapse; margin: 10px 0; font-size: 12.5px; }}
  th, td {{ border: 1px solid #555; padding: 5px 4px; vertical-align: middle; }}
  th {{ text-align: left; width: 34%; font-weight: 600; }}
  td {{ text-align: center; }}
  .lab {{ display: block; font-size: 11px; color: #333; }}
  .hi {{ font-weight: normal; font-size: 11.5px; }}
  .yes {{ text-align: left; }}
  .box {{ width: 15mm; background: #f3f3f3; }}
  .total td {{ font-size: 15px; font-weight: bold; }}
  .rule {{ border: 2px solid #111; padding: 8px 10px; margin: 10px 0; font-size: 14px; }}
  .small {{ font-size: 10.5px; color: #333; }}
  .lookup {{ font-size: 11px; margin-top: 4px; }} .lookup th {{ width: auto; text-align: center; }}
  .lookup td, .lookup th {{ padding: 2px 4px; }}
</style></head><body>
<h1>Diabetes risk card <span>/ डायबिटीज़ जोखिम कार्ड</span></h1>
<p>For adults aged 20+, not pregnant, not already diagnosed. Circle one value in each row and write the points in the grey box.<br>
20+ उम्र के वयस्कों के लिए (गर्भवती नहीं, पहले से डायबिटीज़ नहीं)। हर पंक्ति में एक मान चुनें और अंक ग्रे बॉक्स में लिखें।</p>
<table>
{''.join(rows)}
<tr class="total"><td colspan="{n_band_cols + 1}" style="text-align:right">TOTAL / कुल (out of {card['max_score']})</td><td class="box"></td></tr>
</table>
<div class="rule"><b>{card['cutoff']} or more → refer for an HbA1c blood test.</b><br>
<b>{card['cutoff']} या ज़्यादा → HbA1c खून की जाँच के लिए भेजें।</b></div>
{lookup}
<p class="small">In the US survey this card was built on, a score of {card['cutoff']}+ found {card['sensitivity_at_cutoff']:.0%} of people with
hidden diabetes while referring {card['share_tested_at_cutoff']:.0%} of adults. Among people scoring {card['cutoff']}+, about
{round(100 * card['band_rates']['high']['undiagnosed_diabetes'])} in 100 had hidden diabetes and
{round(100 * card['band_rates']['high']['hba1c_5_7_or_more'])} in 100 had raised sugar.
Below {card['cutoff']}: under {max(1, round(100 * card['band_rates']['low']['undiagnosed_diabetes']))} in 100.</p>
<p class="small">Screening tool, not a diagnosis. Built from NHANES 2017–2020 (US); not yet validated in India.
Anyone with thirst, frequent urination, weight loss or tiredness should be tested whatever their score.<br>
यह स्क्रीनिंग है, निदान नहीं। ज़्यादा प्यास, बार-बार पेशाब, वज़न घटना या थकान हो तो स्कोर कुछ भी हो, जाँच करवाएँ।</p>
</body></html>
"""
out = ROOT / "results" / "points_card_print.html"
out.write_text(page, encoding="utf-8")
print("wrote", out)

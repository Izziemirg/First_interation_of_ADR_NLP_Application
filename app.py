import gradio as gr
import shap
import numpy as np
import scipy as sp
import torch
import transformers
from transformers import pipeline
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer, AutoModelForTokenClassification
import matplotlib.pyplot as plt
import sys
import csv
import os

device = "cuda" if torch.cuda.is_available() else "cpu"
pipeline_device = 0 if torch.cuda.is_available() else -1

HF_TOKEN = os.getenv("HF_TOKEN")

csv.field_size_limit(sys.maxsize)

# ── Model Loading ──────────────────────────────────────────────────────────────

model_name = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name, num_labels=2
).to(device)

pred = pipeline(
    "text-classification",
    model=model,
    tokenizer=tokenizer,
    device=pipeline_device
)

explainer = shap.Explainer(pred)

ner_tokenizer = AutoTokenizer.from_pretrained("d4data/biomedical-ner-all")
ner_model = AutoModelForTokenClassification.from_pretrained("d4data/biomedical-ner-all")
ner_pipe = pipeline(
    "ner",
    model=ner_model,
    tokenizer=ner_tokenizer,
    aggregation_strategy="simple"
)

# ── Core Logic ────────────────────────────────────────────────────────────────

def adr_predict(x):
    text_input = str(x).lower()
    encoded_input = tokenizer(text_input, return_tensors="pt").to(device)
    output = model(**encoded_input)
    scores = torch.softmax(output.logits, dim=-1)[0].detach().cpu().numpy()

    # SHAP
    try:
        shap_values = explainer([text_input])
        local_plot = shap.plots.text(shap_values[0], display=False)
        local_plot = f"""
        <div class="shap-wrapper">
            <div class="section-label">TOKEN ATTRIBUTION</div>
            {local_plot}
        </div>"""
    except Exception as e:
        print(f"SHAP explanation failed: {e}")
        local_plot = """
        <div class="shap-wrapper">
            <div class="section-label">TOKEN ATTRIBUTION</div>
            <div class="fallback-msg">⚠ Attribution analysis unavailable for this input.</div>
        </div>"""

    # NER
    try:
        res = ner_pipe(text_input)
        entity_styles = {
            "Severity":             ("var(--c-red)",    "var(--c-red-bg)"),
            "Sign_symptom":         ("var(--c-green)",  "var(--c-green-bg)"),
            "Medication":           ("var(--c-teal)",   "var(--c-teal-bg)"),
            "Age":                  ("var(--c-amber)",  "var(--c-amber-bg)"),
            "Sex":                  ("var(--c-amber)",  "var(--c-amber-bg)"),
            "Diagnostic_procedure": ("var(--c-slate)",  "var(--c-slate-bg)"),
            "Biological_structure": ("var(--c-purple)", "var(--c-purple-bg)"),
        }

        entity_labels = {
            "Severity":             "SEV",
            "Sign_symptom":         "SYM",
            "Medication":           "MED",
            "Age":                  "AGE",
            "Sex":                  "SEX",
            "Diagnostic_procedure": "DX",
            "Biological_structure": "BIO",
        }

        htext = ""
        prev_end = 0
        res = sorted(res, key=lambda x: x["start"])

        for entity in res:
            start, end = entity["start"], entity["end"]
            word = text_input[start:end]
            etype = entity["entity_group"]
            color, bg = entity_styles.get(etype, ("var(--c-slate)", "var(--c-slate-bg)"))
            label = entity_labels.get(etype, etype[:3].upper())
            htext += text_input[prev_end:start]
            htext += (
                f"<mark class='ner-tag' style='color:{color};background:{bg};border-color:{color};'>"
                f"{word}<sup class='ner-label'>{label}</sup>"
                f"</mark>"
            )
            prev_end = end
        htext += text_input[prev_end:]

        legend_items = "".join([
            f"<span class='legend-chip' style='color:{v[0]};background:{v[1]};border-color:{v[0]};'>{entity_labels[k]} {k.replace('_',' ')}</span>"
            for k, v in entity_styles.items()
        ])

        htext = f"""
        <div class="ner-wrapper">
            <div class="section-label">ENTITY RECOGNITION</div>
            <div class="ner-body">{htext}</div>
            <div class="legend">{legend_items}</div>
        </div>"""
    except Exception as e:
        print(f"NER processing failed: {e}")
        htext = """
        <div class="ner-wrapper">
            <div class="section-label">ENTITY RECOGNITION</div>
            <div class="fallback-msg">⚠ Entity recognition unavailable for this input.</div>
        </div>"""

    label_output = {
        "Severe Reaction":     float(scores[1]),
        "Non-severe Reaction": float(scores[0]),
    }
    return label_output, local_plot, htext


def main(prob1):
    return adr_predict(prob1)


# ── Styles ────────────────────────────────────────────────────────────────────

CSS = """
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@300;400;500&family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

/* ── Design Tokens ── */
:root {
    --bg:          #080d18;
    --bg-panel:    #0d1525;
    --bg-card:     #111c2e;
    --border:      #1e2f4a;
    --border-glow: #1e6abf40;
    --text-primary: #e8eef8;
    --text-secondary: #7a91b4;
    --text-muted:  #3d5373;

    --accent:      #00c8d4;
    --accent-dim:  #00c8d420;
    --accent-glow: #00c8d440;

    --c-red:       #ff6b8a; --c-red-bg:    #ff6b8a15;
    --c-green:     #4ade80; --c-green-bg:  #4ade8015;
    --c-teal:      #00c8d4; --c-teal-bg:   #00c8d415;
    --c-amber:     #fbbf24; --c-amber-bg:  #fbbf2415;
    --c-slate:     #94a3b8; --c-slate-bg:  #94a3b815;
    --c-purple:    #c084fc; --c-purple-bg: #c084fc15;

    --font-display: 'Syne', sans-serif;
    --font-mono:    'DM Mono', monospace;
    --font-body:    'DM Sans', sans-serif;
    --radius:       8px;
}

/* ── Global Reset ── */
*, *::before, *::after { box-sizing: border-box; }

.gradio-container {
    background: var(--bg) !important;
    font-family: var(--font-body) !important;
    color: var(--text-primary) !important;
    min-height: 100vh;
}

/* hide default gradio header chrome */
.gr-prose h1, footer { display: none !important; }

/* ── App Header ── */
.app-header {
    background: linear-gradient(135deg, var(--bg-panel) 0%, #0a1628 100%);
    border: 2px solid var(--accent);
    border-radius: 10px;
    margin: 20px 20px 0;
    padding: 36px 48px 32px;
    position: relative;
    overflow: hidden;
    box-shadow: 0 0 28px var(--accent-glow), inset 0 0 40px #00c8d408;
}

.app-header::before {
    content: '';
    position: absolute;
    top: -80px; right: -80px;
    width: 320px; height: 320px;
    background: radial-gradient(circle, var(--accent-glow) 0%, transparent 70%);
    pointer-events: none;
}

.app-header::after {
    content: '';
    position: absolute;
    bottom: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--accent), transparent);
    opacity: 0.5;
}

.header-eyebrow {
    font-family: var(--font-mono);
    font-size: 11px;
    letter-spacing: 0.2em;
    color: var(--accent);
    text-transform: uppercase;
    margin-bottom: 10px;
    display: flex;
    align-items: center;
    gap: 8px;
}

.header-eyebrow::before {
    content: '';
    display: inline-block;
    width: 20px; height: 1px;
    background: var(--accent);
}

.header-title {
    font-family: var(--font-display);
    font-size: 38px;
    font-weight: 800;
    color: var(--text-primary);
    letter-spacing: -0.02em;
    line-height: 1.1;
    margin: 0 0 6px;
}

.header-title span {
    color: var(--accent);
}

.header-subtitle {
    font-family: var(--font-body);
    font-size: 15px;
    color: var(--text-secondary);
    font-weight: 300;
    max-width: 560px;
    line-height: 1.6;
    margin: 0;
}

.header-badges {
    display: flex;
    gap: 8px;
    margin-top: 20px;
    flex-wrap: wrap;
}

.badge {
    font-family: var(--font-mono);
    font-size: 10px;
    letter-spacing: 0.1em;
    padding: 4px 10px;
    border-radius: 4px;
    border: 1px solid;
    text-transform: uppercase;
}

.badge-teal  { color: var(--accent);   border-color: var(--accent);   background: var(--accent-dim); }
.badge-warn  { color: #fbbf24;         border-color: #fbbf2440;       background: #fbbf2410; }
.badge-model { color: var(--text-secondary); border-color: var(--border); background: var(--bg-card); }

/* ── Disclaimer ── */
.disclaimer {
    margin: 20px 20px 0;
    padding: 16px 20px;
    background: #ff6b8a08;
    border: 1px solid #ff6b8a30;
    border-left: 4px solid var(--c-red);
    border-radius: var(--radius);
    font-family: var(--font-mono);
    font-size: 13px;
    color: #ff6b8acc;
    letter-spacing: 0.05em;
    display: flex;
    align-items: center;
    gap: 14px;
}

.disclaimer-icon {
    font-size: 26px;
    flex-shrink: 0;
    line-height: 1;
}

/* ── Main Content ── */
.main-content {
    padding: 32px 48px;
    display: flex;
    flex-direction: column;
    gap: 24px;
}

/* ── Panels ── */
.panel {
    background: var(--bg-panel);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    overflow: hidden;
    transition: border-color 0.2s;
}

.panel:hover { border-color: var(--border-glow); }

.panel-header {
    padding: 14px 20px;
    border-bottom: 1px solid var(--accent);
    background: #00c8d4;
    display: flex;
    align-items: center;
    gap: 10px;
}

.panel-icon {
    width: 28px; height: 28px;
    display: flex; align-items: center; justify-content: center;
    background: rgba(0,0,0,0.15);
    border-radius: 6px;
    font-size: 14px;
}

.panel-title {
    font-family: var(--font-display);
    font-size: 13px;
    font-weight: 800;
    color: #080d18 !important;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

.panel-body { padding: 20px; }

/* ── Nuclear: kill ALL white backgrounds and force dark everywhere ── */

/* Every Gradio block / wrapper / form surface */
.gradio-container *,
.block, .form, .wrap, .gap,
div[class*="svelte-"], span[class*="svelte-"],
.label-wrap, .prose,
input, textarea, select,
.gr-box, .gr-panel, .gr-input, .gr-textbox,
.gr-label, .gr-html,
[data-testid], [class*="block"],
.output-class, .output-html, .output-label {
    background-color: var(--bg-panel) !important;
    color: #ffffff !important;
}

/* Ensure the top-level container doesn't bleed white */
.gradio-container { background: var(--bg) !important; }

/* ── Textbox ── */
textarea {
    background: var(--bg) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    color: #ffffff !important;
    font-family: var(--font-body) !important;
    font-size: 15px !important;
    line-height: 1.7 !important;
    padding: 14px 16px !important;
    resize: vertical !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
}

textarea:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px var(--accent-dim) !important;
    outline: none !important;
}

textarea::placeholder { color: #3d5373 !important; }

/* ── All labels / spans / p tags → white ── */
label, label span, p, span, h1, h2, h3, h4, li,
.label-wrap span, [class*="label"] span {
    color: #ffffff !important;
    font-family: var(--font-body) !important;
}

/* Textbox field label — monospace treatment */
label > span:first-child,
.gr-textbox label > span,
[class*="label"] > span {
    font-family: var(--font-mono) !important;
    font-size: 11px !important;
    letter-spacing: 0.15em !important;
    text-transform: uppercase !important;
    color: var(--accent) !important;
}

/* ── Analyze button ── */
button.lg.primary, .gr-button-primary, button[variant="primary"] {
    background: linear-gradient(135deg, #007a82, var(--accent)) !important;
    border: none !important;
    border-radius: var(--radius) !important;
    color: #080d18 !important;
    font-family: var(--font-display) !important;
    font-weight: 700 !important;
    font-size: 14px !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    padding: 14px 32px !important;
    width: 100% !important;
    cursor: pointer !important;
    transition: opacity 0.2s, transform 0.1s !important;
    box-shadow: 0 4px 20px var(--accent-glow) !important;
}

button.lg.primary:hover { opacity: 0.9 !important; transform: translateY(-1px) !important; }
button.lg.primary:active { transform: translateY(0) !important; }

/* ── Label / confidence bars ── */
.gr-label, [data-testid="label"] {
    background: var(--bg-panel) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    overflow: hidden !important;
}

/* confidence bar track */
.gr-label .bg-gray-200, [class*="progress-bar"] {
    background: var(--border) !important;
}

/* confidence bar fill */
.gr-label .bg-green-500, [class*="progress"] {
    background: var(--accent) !important;
}

/* label text inside confidence output */
.gr-label span, .gr-label p, .gr-label div {
    color: #ffffff !important;
    font-family: var(--font-mono) !important;
    font-size: 12px !important;
}

/* ── HTML output areas ── */
.gr-html, [data-testid="html"] {
    background: var(--bg-panel) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    min-height: 80px !important;
    color: #ffffff !important;
}

/* ── Examples table ── */
.gr-examples table { border-collapse: collapse !important; width: 100% !important; }

.gr-examples td {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    color: #ffffff !important;
    font-family: var(--font-body) !important;
    font-size: 13px !important;
    padding: 10px 14px !important;
    cursor: pointer !important;
    transition: background 0.15s, color 0.15s !important;
}

.gr-examples td:hover {
    background: var(--accent-dim) !important;
    color: var(--accent) !important;
}

.gr-examples th {
    background: var(--bg) !important;
    border-bottom: 1px solid var(--accent) !important;
    color: var(--accent) !important;
    font-family: var(--font-mono) !important;
    font-size: 10px !important;
    letter-spacing: 0.2em !important;
    text-transform: uppercase !important;
    padding: 10px 14px !important;
}

/* Section dividers */
.divider {
    border: none;
    border-top: 1px solid var(--border);
    margin: 0;
}

/* ── NER & SHAP Output Styles ── */
.section-label {
    font-family: var(--font-mono);
    font-size: 10px;
    letter-spacing: 0.2em;
    color: var(--accent);
    text-transform: uppercase;
    margin-bottom: 14px;
    padding-bottom: 8px;
    border-bottom: 1px solid var(--border);
}

.ner-wrapper, .shap-wrapper {
    padding: 16px;
    font-family: var(--font-body);
}

.ner-body {
    font-size: 15px;
    line-height: 2.2;
    color: var(--text-primary);
    margin-bottom: 16px;
}

.ner-tag {
    display: inline;
    padding: 2px 7px 2px 6px;
    border-radius: 4px;
    border: 1px solid;
    font-weight: 500;
    position: relative;
    margin: 0 2px;
}

.ner-label {
    font-family: var(--font-mono);
    font-size: 8px;
    letter-spacing: 0.1em;
    vertical-align: super;
    margin-left: 3px;
    opacity: 0.8;
}

.legend {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    margin-top: 12px;
    padding-top: 12px;
    border-top: 1px solid var(--border);
}

.legend-chip {
    font-family: var(--font-mono);
    font-size: 9px;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    padding: 3px 8px;
    border-radius: 4px;
    border: 1px solid;
}

.fallback-msg {
    color: var(--text-muted);
    font-family: var(--font-mono);
    font-size: 12px;
    padding: 16px 0;
}

/* ── Examples header override ── */
.examples-header {
    font-family: var(--font-mono) !important;
    font-size: 11px !important;
    letter-spacing: 0.15em !important;
    text-transform: uppercase !important;
    color: var(--text-secondary) !important;
    margin: 8px 0 !important;
}

/* scrollbar */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--accent); }
"""

HEADER_HTML = """
<div class="app-header">
    <div class="header-eyebrow">Biomedical NLP · Adverse Drug Reaction Analysis</div>
    <h1 class="header-title">ADR <span>Detector</span></h1>
    <p class="header-subtitle">
        Clinical-grade adverse drug reaction severity classification powered by
        PubMedBERT & biomedical named entity recognition (NER), including SHAP token attribution for model decision explainability.
    </p>
    <div class="header-badges">
        <span class="badge badge-teal">PubMedBERT</span>
        <span class="badge badge-teal">Bio NER</span>
        <span class="badge badge-teal">Team 9</span>
        <span class="badge badge-model">UVA MSBA Program</span>
    </div>
</div>
"""

DISCLAIMER_HTML = """
<div class="disclaimer">
    <span class="disclaimer-icon">⚠️</span>
    <span>RESEARCH USE ONLY, NOT FOR CLINICAL DIAGNOSIS. This tool does not constitute medical advice.</span>
</div>
"""

INPUT_PANEL_HTML = """
<div class="panel-header">
    <div class="panel-icon">🔬</div>
    <span class="panel-title">ADR Symptom Description Input</span>
</div>
"""

RESULTS_PANEL_HTML = """
<div class="panel-header">
    <div class="panel-icon">📊</div>
    <span class="panel-title">Analysis Results</span>
</div>
"""

EXAMPLES_HEADER_HTML = """
<p class="examples-header">▸ Quick-load example cases</p>
"""

# ── Interface ──────────────────────────────────────────────────────────────────

with gr.Blocks(css=CSS, title="ADR Detector") as demo:

    gr.HTML(HEADER_HTML)
    gr.HTML(DISCLAIMER_HTML)

    with gr.Row():
        with gr.Column(scale=1):
            gr.HTML(INPUT_PANEL_HTML)
            prob1 = gr.Textbox(
                label="CLINICAL NARRATIVE",
                lines=5,
                placeholder="Describe the patient case, medication administered, and observed symptoms...",
            )
            submit_btn = gr.Button("⬡  Run Analysis", variant="primary")

        with gr.Column(scale=1):
            gr.HTML(RESULTS_PANEL_HTML)
            label = gr.Label(
                label="SEVERITY CLASSIFICATION",
                num_top_classes=2,
            )

    gr.HTML("<hr class='divider' style='margin:8px 48px;'>")

    with gr.Row():
        local_plot = gr.HTML(label="SHAP Token Attribution")

    with gr.Row():
        htext = gr.HTML(label="Named Entity Recognition")

    gr.HTML("<hr class='divider' style='margin:8px 48px;'>")

    gr.HTML(EXAMPLES_HEADER_HTML)
    gr.Examples(
        examples=[
            ["A 35 year-old male had severe headache after taking Aspirin. The lab results were normal."],
            ["A 15 year-old female had minor pain in upper abdomen after taking Acetaminophen."],
            ["Patient reported anaphylactic shock within minutes of penicillin administration. Immediate epinephrine administered."],
            ["Mild nausea and slight dizziness observed two hours post ibuprofen ingestion. Symptoms self-resolved."],
        ],
        inputs=[prob1],
        outputs=[label, local_plot, htext],
        fn=main,
        cache_examples=False,
        run_on_click=True,
    )

    submit_btn.click(
        fn=main,
        inputs=[prob1],
        outputs=[label, local_plot, htext],
        api_name="adr",
    )

demo.launch()

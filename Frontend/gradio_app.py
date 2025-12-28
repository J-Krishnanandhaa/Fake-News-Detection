import gradio as gr
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# ---------- Load Model ----------
MODEL_PATH = "models"

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)


# ---------- Heuristic ----------
def is_unfamiliar(text, tokenizer, threshold=1.5):
    tokens = tokenizer.tokenize(text)
    word_count = len(text.split())
    token_count = len(tokens)
    if word_count == 0:
        return True
    return (token_count / word_count) > threshold


# ---------- Prediction ----------
def predict(text):
    text = text.strip()

    if text == "":
        return """
        <div style='padding:12px; border:1px solid #aa3333; border-radius:10px;'>
        ❌ <b>No text provided.</b> Please enter some news text.
        </div>
        """

    if is_unfamiliar(text, tokenizer):
        return f"""
        <div style='padding:12px; border:1px solid #ffcc00; border-radius:10px;'>
        ⚠️ <b>This text looks unfamiliar or noisy.</b><br>
        Prediction skipped because it may be unreliable.<br><br>
        <i>{text}</i>
        </div>
        """

    inputs = tokenizer([text], padding=True, truncation=True, max_length=256, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.softmax(outputs.logits, dim=1)[0]

    fake_prob = float(probs[0])
    real_prob = float(probs[1])
    label = "FAKE" if fake_prob > real_prob else "REAL"
    conf = max(fake_prob, real_prob) * 100

    return f"""
    <div style='padding:12px; border:1px solid #333; border-radius:10px;'>
        <b>Prediction:</b> {label} ({conf:.1f}% confidence)<br><br>
        <b>Fake:</b>
        <div style='width:100%; background:#333; height:12px; border-radius:6px;'>
            <div style='width:{fake_prob*100:.1f}%; background:#d9534f; height:12px; border-radius:6px;'></div>
        </div>
        <b>Real:</b>
        <div style='width:100%; background:#333; height:12px; border-radius:6px; margin-bottom:6px;'>
            <div style='width:{real_prob*100:.1f}%; background:#5cb85c; height:12px; border-radius:6px;'></div>
        </div>
        <small><i>{text}</i></small>
    </div>
    """


# ---------- Samples ----------
REAL_1 = """Trumps hopes Syria safe zones may force decision Assad WASHINGTON by was stated by as reports President Donald Trumps push create safe zones Syria could force make risky decisions far go protect refugees, including shooting Syrian Russian aircraft committing thousands U.S. troops, experts reported. Wednesday was reported by Trump absolutely safe zones Syria refugees fleeing violence. According document seen by was stated by as reports, expected coming days order Pentagon State Department draft plan create zones Syria nearby nations. document spell would make safe zone safe whether would protect refugees threats ground jihadist fighters whether Trump envisions nofly zone policed America allies. nofly zone, without negotiating agreement Russia Trump would decide whether give U.S. military authority shoot Syrian Russian aircraft posed threat people zone, predecessor, former President Barack Obama."""

REAL_2 = """German politicians accuse Trump trivializing Nazi violence BERLIN by was reported by as reports Senior German politicians Wednesday accused U.S. President Donald Trump trivializing violence white supremacists Virginia called clear rejection ideology. Governments could win fight hatred, racism antiSemitism rejecting ideology willingness use violence, reported Martin Schulz, centerleft candidate chancellor, adding applies Germany United States. trivialization Nazi violence confused utterances Donald Trump highly dangerous, reported Schulz, leader Social Democrats SPD. tolerate monstrosities coming presidents mouth, told RND newspaper group interview. Republican leaders criticized Trump saying leftist counterprotesters also blame violence last Saturday Charlottesville left one person dead several injured. comments praise white farright groups. Schulz main challenger Chancellor Angela Merkel Sept. 24 election. SPD, junior partner Merkels grand coalition…"""

FAKE_1 = """Republican Senators Lindsey Graham and John McCain have joined calls for a bipartisan investigation into Russia’s potential involvement in the 2016 U.S. presidential election. Graham, a senior member of the Senate, stated his intention to pursue the matter through two subcommittees he chairs, while also planning a visit to Eastern Europe with McCain to examine Russia’s alleged interference in other countries’ elections. Both senators were previously critical of Donald Trump during his campaign—Graham supported independent candidate Evan McMullin in the general election, and McCain withdrew his endorsement of Trump following controversy over leaked private comments. Although the Obama administration initially stated in late November 2016 that there was no evidence of election-day tampering, a later report from U.S. intelligence agencies concluded with “high confidence” that Russia had attempted to influence the election outcome in Trump’s favor. The agencies cited cyberattacks on both Democratic and Republican organizations but noted that only Democratic documents were made public."""

FAKE_2 = """On a recent episode of MSNBC’s Morning Joe, host and former Republican congressman Joe Scarborough reflected critically on the economic messaging of the Republican Party over the last several decades. During a discussion about the appeal of Donald Trump to conservative voters, Scarborough acknowledged that some traditional Republican economic policies may have failed to address the needs of working-class Americans. “The problem with the Republican Party over the past 30 years,” Scarborough said, “is that we haven’t developed an economic message that truly connects with working-class Americans the way Donald Trump’s message does.” He pointed to specific policies often promoted by the party, such as cuts to capital gains taxes and the repeal of the estate tax, as largely irrelevant to the economic realities faced by many of Trump’s supporters. According to Scarborough, the benefits of free trade agreements and tax reductions for high earners have not translated into broader prosperity for everyday workers. “We talk about how great free trade deals are,” he remarked, “but those benefits don’t reach the people attending these rallies. It never trickles down.”"""

def fill_and_predict(sample_text):
    return sample_text, predict(sample_text)


# ---------- UI ----------
with gr.Blocks(css="""
    .prediction-card {
        border: 1px solid #444;
        border-radius: 10px;
        padding: 14px;
        min-height: 280px;
        background: #1c1c1c;
    }
    .predict-btn {
        background-color: #4169E1 !important;
        color: white !important;
    }
    .sample-btn {
        background-color: #6A89A7 !important;
        color: white !important;
    }
""") as demo:

    gr.Markdown(
        """
        <h1 style="text-align:center;">📰 Fake News Detection (DistilBERT)</h1>
        <p style="text-align:center;">
        Paste headline(s) or article text. Use <b>;</b> to separate multiple inputs.
        </p>
        """
    )

    with gr.Row():
        # --- LEFT: INPUT ---
        with gr.Column(scale=1):
            input_box = gr.Textbox(
                lines=14,
                label="Input Text",
                placeholder="Paste news text here..."
            )

            with gr.Row():
                clear_btn = gr.Button("🧹 Clear")
                submit_btn = gr.Button("🔍 Predict", elem_classes="predict-btn")

        # --- RIGHT: OUTPUT ---
        with gr.Column(scale=1):
            
            output_box = gr.HTML(
                label="Results",
                value="<i>No prediction yet.</i>",
                elem_classes="prediction-card"
            )

    gr.Markdown("### 📄 Sample Texts (Click predict buttons below)")

    with gr.Row():
        with gr.Column():
            sample1 = gr.Textbox(value=REAL_1, label="Real Sample 1", lines=10, interactive=False)
            s1_btn = gr.Button("Predict using Sample 1 (Real)", elem_classes="sample-btn")

        with gr.Column():
            sample2 = gr.Textbox(value=REAL_2, label="Real Sample 2", lines=10, interactive=False)
            s2_btn = gr.Button("Predict using Sample 2 (Real)", elem_classes="sample-btn")

    with gr.Row():
        with gr.Column():
            sample3 = gr.Textbox(value=FAKE_1, label="Fake Sample 1", lines=10, interactive=False)
            s3_btn = gr.Button("Predict using Sample 1 (Fake)", elem_classes="sample-btn")

        with gr.Column():
            sample4 = gr.Textbox(value=FAKE_2, label="Fake Sample 2", lines=10, interactive=False)
            s4_btn = gr.Button("Predict using Sample 2 (Fake)", elem_classes="sample-btn")

    # wiring
    submit_btn.click(fn=predict, inputs=input_box, outputs=output_box)
    clear_btn.click(lambda: "", None, input_box)

    s1_btn.click(lambda: fill_and_predict(REAL_1), None, [input_box, output_box])
    s2_btn.click(lambda: fill_and_predict(REAL_2), None, [input_box, output_box])
    s3_btn.click(lambda: fill_and_predict(FAKE_1), None, [input_box, output_box])
    s4_btn.click(lambda: fill_and_predict(FAKE_2), None, [input_box, output_box])

demo.launch()


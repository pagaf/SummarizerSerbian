#!/usr/bin/env python3
# pip install gradio feedparser cyrtranslit beautifulsoup4 torch transformers peft bitsandbytes accelerate

import re
import datetime
from datetime import timedelta
import feedparser
import gradio as gr
import cyrtranslit
from bs4 import BeautifulSoup
import os
import zipfile
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, PeftModel


LORA_ZIP = "lora_adapter_qwen3_14b.zip"
LORA_PATH = "./qwen3_14b_serbian_sum_lora"

if not os.path.exists(LORA_PATH):
    with zipfile.ZipFile(LORA_ZIP, "r") as z:
        z.extractall(LORA_PATH)

tokenizer = AutoTokenizer.from_pretrained(LORA_PATH)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

base_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-14B",
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)

model = PeftModel.from_pretrained(base_model, LORA_PATH)
model.eval()

SYSTEM_PROMPT = (
    "You are an AI assistant that summarizes news in Serbian/Croatian. "
    "Be brief, precise, and factual."
)

SOURCES = {
    "N1 Serbia":            "https://rs.n1info.com/feed/",
    "24sata.hr (Hrvatska)": "https://www.24sata.hr/feeds/najnovije.xml"
}

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)

TIME_CHOICES = [
    ("Last hour",   "1h"),
    ("Last day",    "1d"),
    ("Last 3 days", "3d"),
    ("Last week",   "week"),
]

ARTICLES_PER_PAGE = 8
MAX_VISIBLE_ARTICLES = ARTICLES_PER_PAGE
articles = []
current_page = 0


def clean(text):
    return re.sub(r"\s+", " ", BeautifulSoup(text or "", "html.parser").get_text(" ")).strip()


def to_latin(text):
    if not text:
        return ""
    cyr = sum(1 for ch in text if "\u0400" <= ch <= "\u04FF")
    return cyrtranslit.to_latin(text, "sr") if cyr / max(len(text), 1) > 0.1 else text


def trunc(text, n=240):
    return text[:n] + "…" if len(text) > n else text


def ask_model(prompt, max_t=200):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    
    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_t,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    return response.strip()


def summarize_article(idx):
    if not articles:
        return "Load news first."
    if idx < 0 or idx >= len(articles):
        return "Invalid article number."

    a = articles[idx]
    text = to_latin(f"{a['title']}\n\n{a['body']}")
    return ask_model(f"Article:\n{text}\n\nSummarize in 3-5 sentences:", 180)


def get_pub_dt(entry):
    p = getattr(entry, "published_parsed", None) or getattr(entry, "updated_parsed", None)
    if not p:
        return None
    return datetime.datetime(*p[:6])


def fetch_news(source_name, time_filter):
    global articles, current_page
    current_page = 0

    feed_url = SOURCES[source_name]
    feed = feedparser.parse(feed_url, agent=USER_AGENT)

    if not feed.get("entries"):
        articles = []
        return [f"No items found in {source_name}."] + [
            gr.update(value="", visible=False) for _ in range(MAX_VISIBLE_ARTICLES * 6)
        ]

    now = datetime.datetime.now()
    articles = []

    for entry in feed.entries[:20]:
        pub_dt = get_pub_dt(entry)
        if pub_dt is None:
            continue

        delta = now - pub_dt
        if time_filter == "1h"   and delta > timedelta(hours=1):   continue
        if time_filter == "1d"   and delta > timedelta(days=1):    continue
        if time_filter == "3d"   and delta > timedelta(days=3):    continue
        if time_filter == "week" and delta > timedelta(weeks=1):   continue

        title = clean(entry.get("title", "Untitled"))
        body  = clean(entry.get("summary", "") or entry.get("description", ""))
        link  = entry.get("link", "#")

        articles.append({
            "title":      to_latin(title),
            "body":       to_latin(body),
            "link":       link,
            "date":       pub_dt.strftime("%H:%M %d.%m"),
            "summary_ai": "",
        })

    status = f"Loaded {len(articles)} article(s) from {source_name} ({time_filter})."
    return render_page(status)


def render_page(status_override=""):
    global current_page, articles

    if not articles:
        updates = [status_override or "Load news first."]
        for i in range(MAX_VISIBLE_ARTICLES):
            updates.extend([
                gr.update(value="", visible=False),
                gr.update(value="", visible=False),
                gr.update(value="", visible=False),
                gr.update(value="", visible=False),
                gr.update(visible=False),
                gr.update(value="", visible=False),
            ])
        return updates

    total_pages = (len(articles) + ARTICLES_PER_PAGE - 1) // ARTICLES_PER_PAGE
    start_idx = current_page * ARTICLES_PER_PAGE
    end_idx   = min(start_idx + ARTICLES_PER_PAGE, len(articles))

    status = f"Page {current_page + 1}/{max(1, total_pages)} | {len(articles)} total."
    if status_override:
        status = status_override

    updates = [status]

    for i in range(MAX_VISIBLE_ARTICLES):
        real_idx = start_idx + i
        if real_idx < end_idx:
            a = articles[real_idx]

            text = f"{trunc(a['body'], 240)}"

            updates.extend([
                gr.update(value=f"### {real_idx + 1}. {a['title']}", visible=True),
                gr.update(value=text, visible=True),
                gr.update(
                    value=f"*{a['date']} · [Open original]({a['link']})*",
                    visible=True,
                ),
                gr.update(value="Summarize this article", visible=True),
                gr.update(visible=True),
                gr.update(
                    value="" if not a.get("summary_ai") else f"**Summary:**\n\n{a['summary_ai']}",
                    visible=True,
                ),
            ])
        else:
            updates.extend([
                gr.update(value="", visible=False),
                gr.update(value="", visible=False),
                gr.update(value="", visible=False),
                gr.update(value="", visible=False),
                gr.update(visible=False),
                gr.update(value="", visible=False),
            ])

    return updates


def page_prev():
    global current_page, articles
    if not articles:
        return ["Load news first."] + [
            gr.update(value="", visible=False)
            for _ in range(MAX_VISIBLE_ARTICLES * 6)
        ]
    if current_page > 0:
        current_page -= 1
    return render_page()


def page_next():
    global current_page, articles
    if not articles:
        return ["Load news first."] + [
            gr.update(value="", visible=False)
            for _ in range(MAX_VISIBLE_ARTICLES * 6)
        ]
    total_pages = (len(articles) + ARTICLES_PER_PAGE - 1) // ARTICLES_PER_PAGE
    if current_page < total_pages - 1:
        current_page += 1
    return render_page()


def summarize_one_article(idx_page_offset):
    global articles, current_page
    if not articles:
        return "Load news first."

    start_idx = current_page * ARTICLES_PER_PAGE
    real_idx = start_idx + idx_page_offset

    if real_idx >= len(articles):
        return "Index out of bounds."

    a = articles[real_idx]
    text = to_latin(f"{a['title']}\n\n{a['body']}")
    summary = ask_model(f"Article:\n{text}\n\nSummarize in 2–3 sentences:", 180)
    articles[real_idx]["summary_ai"] = summary

    return f"**Summary:**\n\n{summary}"


css = """
* { font-family: Inter, system-ui, sans-serif; }
.gradio-container { max-width: 1200px !important; margin: 0 auto; padding: 20px; }

.header {
  background: linear-gradient(135deg, #667eea, #764ba2);
  border-radius: 20px; padding: 24px; color: white;
  text-align: center; margin-bottom: 20px;
  box-shadow: 0 20px 40px rgba(102,126,234,.25);
}
.header h1 { margin: 0; font-size: 1.8rem; }
.header p  { margin: 6px 0 0; opacity: .95; }

.news-card {
  border: 1px solid #e5e7eb; border-radius: 18px;
  padding: 14px 16px; margin-bottom: 12px; background: #fff;
  transition: transform .15s ease, box-shadow .15s ease;
}
.news-card:hover {
  transform: translateY(-1px);
  box-shadow: 0 8px 24px rgba(0,0,0,.08);
}

.hover-summarize {
  opacity: 0;
  pointer-events: none;
  transition: opacity .15s ease;
}
.news-card:hover .hover-summarize {
  opacity: 1;
  pointer-events: auto;
}
"""

with gr.Blocks(css=css, title="Serbian News AI") as demo:
    gr.HTML("""
        <div class="header">
          <h1>Balkan News Summarizer </h1>
        </div>
    """)

    with gr.Row():
        time_filter = gr.Radio(choices=TIME_CHOICES, value="1d",
                               label="Time window", container=True)
        source = gr.Dropdown(choices=list(SOURCES.keys()),
                             value=list(SOURCES.keys())[0],
                             label="Source", scale=2)

    fetch_btn = gr.Button("Load news", variant="primary")
    status    = gr.Textbox(label="Status", interactive=False, max_lines=1)

    with gr.Row(equal_height=False):
        prev_btn = gr.Button("◀ Prev page", variant="secondary", scale=1)
        next_btn = gr.Button("Next page ▶", variant="secondary", scale=1)

    gr.Markdown("### News")

    card_titles, card_bodies, card_meta = [], [], []
    card_buttons_text, card_buttons_vis, card_summaries = [], [], []

    for i in range(MAX_VISIBLE_ARTICLES):
        with gr.Group(elem_classes=["news-card"]):
            title = gr.Markdown(visible=False)
            body  = gr.Markdown(visible=False)
            meta  = gr.Markdown(visible=False)
            button_text = gr.Button("Summarize this article", variant="secondary",
                                    visible=False)
            btn_vis = gr.Markdown("", visible=False)
            summary = gr.Markdown(visible=False)

        card_titles.append(title)
        card_bodies.append(body)
        card_meta.append(meta)
        card_buttons_text.append(button_text)
        card_buttons_vis.append(btn_vis)
        card_summaries.append(summary)

    fetch_outputs = [status]
    for i in range(MAX_VISIBLE_ARTICLES):
        fetch_outputs.extend([
            card_titles[i],
            card_bodies[i],
            card_meta[i],
            card_buttons_text[i],
            card_buttons_vis[i],
            card_summaries[i],
        ])

    fetch_btn.click(
        fetch_news,
        inputs=[source, time_filter],
        outputs=fetch_outputs,
    )

    prev_btn.click(
        page_prev,
        outputs=fetch_outputs,
    )

    next_btn.click(
        page_next,
        outputs=fetch_outputs,
    )

    for i in range(MAX_VISIBLE_ARTICLES):
        card_buttons_text[i].click(
            lambda idx=i: summarize_one_article(idx),
            inputs=[],
            outputs=card_summaries[i],
        )

if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=8080,
        share=False,
        theme=gr.themes.Soft(),
        show_error=True,
    )
    print("http://127.0.0.1:8080")

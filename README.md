# 💄 Nykaa Beauty Advisor — RAG Chatbot

A Streamlit + LangChain chatbot that acts as a safe, grounded customer-journey companion for beauty shopping. It discloses that it's an AI, assesses the shopper's needs, recommends products **only** from a verified local catalogue, tracks orders, answers policy questions, and hands off to a human whenever a query is sensitive, high-value or uncertain.

> **Demo data notice:** all products, brands, prices, policies and orders in the CSV files are fictional sample data for this prototype. They are not Nykaa's real catalogue or policies.

## Quick start

```bash
git init                      # optional: start a repo
python -m venv .venv
source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

The app opens at http://localhost:8501. **No API key is needed** — without one it runs in offline mode, where every answer is a grounded template built directly from the CSV data.

### Optional: turn on an LLM

The LLM only writes the explanatory prose; prices, stock and order status are always rendered from the data. Set one provider via environment variables or `.streamlit/secrets.toml` (see `.streamlit/secrets.toml.example`):

```bash
export OPENAI_API_KEY=sk-...            # uses gpt-4o-mini by default (OPENAI_MODEL to change)
# or
export ANTHROPIC_API_KEY=sk-ant-...     # uses claude-haiku-5-5 by default (ANTHROPIC_MODEL to change)
```

The sidebar's **Engine → Mode** line shows which mode is active.

## Try these

| Say | What happens |
|---|---|
| `I have oily skin` | Asks product type, then budget, then shows verified matches with checkout links |
| `Compare these` / `is the first one fragrance free?` | Comparison table / grounded product Q&A |
| `NYK10234, 4821` | Verified order tracking (order ID + last 4 phone digits) |
| `NYK10239, 7710` | High-value delayed order → automatic specialist handoff |
| `What's your return policy?` | Policy answer with source IDs |
| `I have a rash, what is it?` | Refuses to diagnose; redirects to a dermatologist |
| `I was charged twice` | Sensitive intent → handoff with chat summary |
| `are you a real person?` | Honest AI disclosure |

## Project structure

```
app.py              Streamlit UI, RAG engine, guardrails, fact-checker (heavily commented)
catalog.csv         18 sample products: name, brand, category, skin types, price, ingredients, stock…
policies.csv        13 shipping / payment / returns / cancellation FAQs
orders.csv          6 demo orders for tracking
test_app.py         25 automated tests (pytest -q) — no API key required
.streamlit/         Theme + secrets example
```

## How it works

**Journey (6 steps):** Disclose → Ask intent → Needs assessment (skin type, product type, budget) → Retrieve → Explain → Checkout link or human handoff. The sidebar tracks progress.

**Retrieval:** pandas hard-filters the catalogue on skin type, category, budget and stock, then TF-IDF ranks the survivors (and answers free-text policy/product questions) over LangChain `Document`s. For a large catalogue, swap `KnowledgeBase.search()` for an embeddings vector store.

**Zero-hallucination design**
- Product cards (price, size, stock, ingredients) are rendered straight from `catalog.csv`, never from LLM text.
- The LLM sees only retrieved documents and must cite source IDs (e.g. `[NBA-SE-008]`, `[POL-05]`), or reply `INSUFFICIENT_CONTEXT`.
- Every LLM draft is fact-checked before display: unknown numbers or prices, discount/guarantee/medical/stock claims not in the data, un-retrieved product names, invalid citations and human-mimicry phrases all cause rejection, and a grounded template is shown instead. Citations are stripped from the visible text and listed under **🔎 Verified sources & fact-check**.

**Escalation gate (runs before any LLM):** explicit agent requests, medical queries tied to a purchase, payment / fraud / damaged-item / legal / data-privacy topics, bulk orders, budgets ≥ ₹5,000, high-value orders with delivery problems, repeated failed order verification, and two low-confidence turns in a row. Each handoff creates a ticket with a summary (reason, preferences, products viewed, order, recent transcript) that's downloadable as JSON.

Thresholds (`HIGH_VALUE_THRESHOLD_INR`, `CONFIDENCE_THRESHOLD`, etc.) live at the top of `app.py`.

## Customising

- **Data:** edit the CSVs (keep the column names). The app re-indexes automatically when a file changes.
- **Checkout links:** set `CHECKOUT_BASE_URL`; the SKU is appended to it. The default points to a placeholder `example.com` URL.
- **Human handoff:** `AdvisorEngine._handoff()` currently builds the ticket in-app. Connect it to your helpdesk (Zendesk, Freshdesk, Salesforce) by posting the summary dict to its API there.

## Run the tests

```bash
pytest -q
```

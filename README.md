# **PreMortem AI → SEC Filings + BigQuery**

This NLP pipeline quantifies historical risk from SEC filings, creating a blueprint for a predictive, real-time market analysis engine all using Google's Big Query.

<p>
  <img width="300" height="500" alt="BQM" src="https://github.com/user-attachments/assets/99e7cbf2-6aa0-4707-ae5a-08e244c078fa" />
  <img width="215" height="215" alt="image" align="right" src="https://github.com/user-attachments/assets/e81946fa-31dc-4af0-b125-37d8f88c12d8" />
</p>

# **PreMortem AI → SEC Filings + BigQuery — Focus: 2008 Financial Crisis (JPM, MS, GS, BAC)**
This project targets the roots of the 2008 financial crisis. I focused on the 4 major U.S. banks that played central roles:

* JP Morgan Chase(JPM)
* Morgan Stanley(MS)
* Goldman Sachs(GS)
* Bank of America(BAC)

  <p>
    <img width="250" height="250" alt="image" src="https://github.com/user-attachments/assets/97629b8b-4a86-48d9-84f2-c4a2fb414d74" />
  </p>

My mission was to extract the language and signals inside their SEC 10-K and 10-Q filings from the pre-crisis and crisis years, measure tone and risk, and test whether those signals could have flagged trouble early. The goal is practical: if a similar micro- or macro-crisis were brewing today, could the same pipeline surface early warnings?

## Scope and Era
I collected 10-Ks and 10-Qs for JPM, MS, GS, and BAC covering roughly 2006–2010. That window captures the pre-crisis buildup, the crisis itself, and immediate aftermath. I focused on sections that historically show risk signals: Management Discussion & Analysis (MD&A), liquidity/disclosures, risk factors, and other sections related to financial disclosures.

## My Mission
* Extract the text passages where management discusses liquidity, credit risk, counterparty exposure, and valuation uncertainty.

* Score tone and highlight negative, cautious, or evasive language using FinBERT sentiment.

* Create semantic embeddings so I can cluster similar passages across banks and dates and find recurring warning phrases.

* Link filing signals to stock market moves and test simple predictive checks — could spikes in negative language precede large drawdowns?

* Produce an executive narrative that ties specific filing passages to market events and lists the earliest signals that, in hindsight, correlated with trouble.

## Why These Banks and Why This Era
These four banks are the pillars of modern financial plumbing they underwrite markets, provide short‑term funding, clear trades, and serve as major counterparties for governments, corporations, and households. The 2005–2010 window captures the arc from stability to collapse quiet increases in leverage and complex valuations in 2005–2006, growing liquidity caveats in 2007, and frank admissions of stress in 2008. Reading their filings side‑by‑side exposes a reproducible pattern recurring language about liquidity reliance, valuation uncertainty, and counterparty concentration that often precedes measurable market dislocations. That combination of institutions and years gives PreMortem AI a compact, high‑stakes testbed for separating routine corporate caution from signals of systemic fragility.

<img width="500" height="900" alt="image" src="https://github.com/user-attachments/assets/9f33d721-ce63-4e45-8389-a4370da5cf44" />


## What I Did (Step by Step)
* Collected the Filings: I gathered the SEC 10-K and 10-Q PDF filings from the SEC Edgar databases for the four target banks JPM, MS, GS, and BAC from the era of 2005-2010

* Extracted and Processed Text: I used the filter-10k.py and filter-10q.py scripts to automate text extraction and sentiment scoring. For each PDF the scripts performed the following actions:

* Identified and extracted text: From high signal sections such as Management's Discussion & Analysis (MD&A), Financial Statements, Risk Factors and other financial sections.

* Calculated sentiment scores: (positive, negative, and neutral) for the text using the ProsusAI/finbert model.

* Structured the output as JSONL rows: With each row containing the text chunk, sentiment scores, and provenance fields (e.g., bank ticker, filing date, and section etc etc).

* Ingested Data and Generated Embeddings in BigQuery: The generated JSONL files were uploaded into BigQuery tables. The BigQuery-injector.py script then performed the following:

* Created tables named 10q and 10k with the processed text and sentiment features.

* Called BigQuery ML.GENERATE_EMBEDDING to generate semantic vector embeddings for the text.

* This data was then linked with historical stock return data for subsequent event analysis.

* Analyzed and Synthesized Insights: I ran the analyser.py script to connect the filing data to market events. This script:

* Queried BigQuery to analyze sentiment trends and correlations between spikes in negative language and subsequent stock drawdowns.

* Generated visualizations of sentiment trends and stock performance using matplotlib.

* Used BigQuery's ML.GENERATE_TEXT function to produce an executive-level narrative that synthesized the key findings, tying specific passages from filings to market events and identifying the earliest warning signals found in the data.

## What I Tested and Why It Matters

* **Alpha Generation from Language:** I tested whether spikes in negative sentiment consistently preceded significant stock drawdowns.

* **Why it Matters:** This transforms corporate language from a qualitative descriptor into a quantitative, leading indicator of risk. Finding a consistent lead/lag effect means the signal has potential alpha, offering an exploitable information edge before that risk is reflected in the market price. In finance, capital follows confidence. An auditable, transparent signal is fundamentally more valuable than a blackbox prediction. This traceability builds the conviction required for portfolio managers to act decisively on the data, turning AI insights into defensible, capital backed market positions.

* **Systemic Contagion Signals:** Using embeddings, I tested whether the same semantic warnings appeared across multiple banks simultaneously before market wide distress.

* **Why it Matters:** This shifts the philosophy from single stock analysis to systemic risk hedging. When the same red flag appears across pillars of the financial system, it's no longer an isolated issue but a signal of potential contagion. This provides a rare lens on market wide fragility, informing macro and portfolio-level decisions.

* **Signal Verifiability:** I ensured that every AI flagged insight is traceable back to a specific sentence and page in the original filing.

## Why This Matters Today
Every market era tells itself a story. In the late 90s, it was about dot-coms and the internet hype. Before 2008, it was about the infallibility of housing. Today, the story is unequivocally about AI. This narrative directs the flow of capital: VC funds pour billions into startups based on a pitch deck, and public companies pivot their entire strategy around it, fearing obsolescence.

But beneath the main story, there's a subtext. It’s written not in press releases, but in the cautious, legally vetted language of SEC filings. This is where the real sentiment the corporate subconscious of the market lives. This pipeline was built to read that subtext.

Its purpose isn't just to spot AI buzzwords. It's to act as a barometer for the entire innovation ecosystem. When a wave of VC funded hype promises to revolutionize an industry, this tool can track if those private market promises are translating into sustainable revenue and realistic risk assessments within public companies. It’s designed to detect the subtle shift from bold ambition to cautious realism the first sign that a hype cycle is maturing or souring.

Ultimately, this is about understanding the narrative engine of our economy. Before the numbers turn, the language does. This pipeline is designed to listen to that language, providing a ground truth signal on whether the story we're being told is one of sustainable innovation or just another bubble waiting for a pin.

## Techinical insights


<img width="4200" height="1170" alt="image" src="https://github.com/user-attachments/assets/85389fe2-ed91-4cd8-8576-068f0c6edc3c" />


I have architected a Multi-Stage ETL & RAG (Retrieval-Augmented Generation) Hybrid Pipeline. This system moves unstructured financial data (PDFs) through local deep learning inference (FinBERT) for semantic enrichment, into a cloud data warehouse (BigQuery), and finally utilizes Cloud-native Generative AI (Gemini via BQML) for executive reporting.

1. **Ingestion & Preprocessing Layer:**

**Files: filter-10k-v2.py, filter-10q-v2.py**

This layer acts as my "Extract and Transform" phase. I bifurcated the logic for 10-K (Annual) and 10-Q (Quarterly) reports to handle their differing structural complexities effectively.

* **High-Fidelity Text Extraction:** I utilize pdfplumber for text extraction rather than basic OCR or simple wrappers like PyPDF2. This provides superior handling of layout analysis, which is crucial for the multi-column format often found in SEC filings.

* **Regex-Based Sectioning:**

* **10-K Logic:** I target specific Items (1A, 7, 7A) using ITEM\s+(\d+[A-Z]?)\.? patterns to isolate risk factors and management discussions.

* **10-Q Logic (Advanced):** My 10-Q script is significantly more robust. I implemented a Table of Contents (TOC) detection strategy first, scanning the head of the document for Item mappings. If that fails, I fall back to a "brute-force" regex search for "Management's Discussion and Analysis" (MD&A), utilizing "bounding box" logic (finding the start of Item 2 and the start of Item 3 or Part II to define the end).

* **Semantic Enrichment (FinBERT):** Instead of generic sentiment analysis, I implemented ProsusAI/finbert, a BERT model fine-tuned specifically on financial text.

* **Tokenization & Chunking:** Since BERT models have a hard limit of 512 tokens, I implemented a sliding window mechanism. In my 10-K script, I chunk tokens with a max length of 510 to allow for special tokens ([CLS], [SEP]). I then aggregate these chunks to derive a document-level sentiment score.

* **Compute Optimization:** I included logic to dynamically check for CUDA availability, optimizing for GPU acceleration whenever the environment supports it.

2. **Injection & Vectorization Layer**

**File: bg-injector.py**

This script handles the "Load" phase and the initial "Vectorization" for the data warehouse.

* **Resilient ETL:** I used the tenacity library to implement exponential backoff strategies. This ensures my pipeline is production-grade and can handle transient network failures gracefully when communicating with Google Cloud APIs.

* **In-Warehouse Embedding Generation:** Rather than generating embeddings locally—which is compute-heavy and hard to scale—I offloaded this task to BigQuery ML. My SQL query CREATE OR REPLACE TABLE ... AS SELECT ... FROM ML.GENERATE_EMBEDDING generates vector embeddings for text chunks directly inside the warehouse using the hackathon.embedder model. This design drastically reduces data movement overhead.

3. Analytics & Generation Layer
**File: analyser.py**

This is the consumption layer where I merge traditional analytics with Generative AI.

* **SQL-Based Financial Analytics:**

My analyze_stock_trends method utilizes SQL Window Functions (LAG(close, 30) OVER ...) to calculate month-over-month percentage changes directly in the database engine. This avoids pulling raw daily data into Python memory, ensuring high performance for large datasets.

* **Hybrid Insight Generation (RAG-Lite):**

I construct a "context window" for the LLM by aggregating statistical descriptors of stock performance and aggregated sentiment metrics (e.g., peak negative sentiment).

* **BQML Generative Inference:** I utilize ML.GENERATE_TEXT with the crisis_analysis.anal model to invoke Gemini Pro directly via SQL. By passing the prompt as a ScalarQueryParameter, I ensure strict separation between code and data to prevent SQL injection.

Fallback Mechanism: I implemented a deterministic fallback method, _generate_fallback_insights. If the ML model call fails, my system degrades gracefully, providing a template-based report instead of crashing.

## Future Vision

This project was conceived as a forensic tool. We looked back at the wreckage of 2008 and proved we could find the hairline fractures in the banks own disclosures before the structure collapsed. We successfully turned the slow, deliberate language of SEC filings into a quantifiable time series of risk. But the true promise of this architecture isnt in analyzing the past it's in interpreting the present.

The core pipeline: extract, embed, analyze is data agnostic by design. The logical evolution is to point this engine away from the quiet archives of the SEC and towards the chaotic, high-frequency data streams that shape today's market narrative.

The next gen pipeline would ingest a torrent of unstructured and numerical data:

* News Feeds & APIs: Real-time data from sources like NewsAPI, Bloomberg, or Reuters, parsed for sentiment shifts and the emergence of specific risk factors.

* Social & Forum Data: Scraping and analyzing sentiment from niche, high-signal communities like Reddit's r/wallstreetbets or specialized financial forums to capture retail momentum and emerging thematic bets.

* Earnings Call Analysis (Text & Audio): Augmenting transcript analysis with vocal sentiment models. By processing the call audio, we can extract acoustic signals like vocal stress or hesitation that provide a layer of context impossible to capture from text alone. This offers a more holistic view of executive confidence and conviction.

* Stochastic Volatility & Path Modeling: Integrate GARCH (Generalized Autoregressive Conditional Heteroskedasticity) models to capture volatility clustering and forecast market variance. The outputs from these models would then seed large scale Monte Carlo simulations, generating a robust synthetic numerical dataset of potential future market paths. Fusing this probabilistic numerical data with real-time qualitative sentiment from other streams would create a powerful, hybrid predictive model capable of simulating a narrative's potential impact on market structure.

This is the blueprint for a true market prediction engine. Deployed on large scale servers with massive GPU clusters, the pipeline would process and correlate this hybrid data in real time. We would train a proprietary foundation model on this fused dataset combining linguistic sentiment, vocal stress, social momentum, and probabilistic market simulations. The goal is to achieve signal precision at an unprecedented level.

While most quantitative models predict entry and exit "zones," this system is designed to pinpoint the inflection points— the exact moments of market capitulation or euphoric ascent. It’s an engine built not just to forecast a range, but to identify the catalyst before it's visible in the price action. The ultimate objective is to create a source of alpha so profound it operates with a clarity that aims to be even better than insider trading, identifying the spark that ignites a rally "to the moon" or the pin that pops the bubble.

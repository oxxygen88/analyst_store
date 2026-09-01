# INDOKIDS Branch Command Center V2.8
## Advanced AI Presentation Studio

V2.8 upgrades the AI Presentation menu from a fixed 8-slide deck into a configurable presentation studio.

### Workflow
1. Choose AI provider (Gemini/OpenAI), audience, and language.
2. Write the presentation objective.
3. Choose presentation depth: Executive, Standard, or Deep Dive.
4. Select focus areas.
5. Add mandatory discussion points in free text.
6. Generate an AI Slide Plan or use the recommended local plan.
7. Review/edit the plan: order, title, objective, slide type, include/exclude, emphasis, and optional new rows.
8. Generate final AI insights and the dynamic PPTX.

### Slide library
- Executive Performance
- Target & Forecast
- Sales Growth
- Gap Diagnosis
- Pareto Product
- Pareto Migration
- Product Opportunity
- Stockout / Revenue Recovery
- Inventory Health
- Inventory Capital at Risk
- Profitability
- Supplier Performance
- Category Performance
- Mutasi / Transfer Analysis
- Purchase & Replenishment
- Anomaly & Data Quality
- 30-Day Action Plan

### Data governance
AI receives an aggregated fact-pack, not the complete raw transaction table. Numerical facts, tables, KPI cards, and charts are created by the application analytics engine. If causality is not proven, the AI is instructed to present it as a hypothesis or recommended investigation.

### Deployment
No new Python dependency is required compared with V2.7. For Streamlit Cloud, replace `app.py` and `src/ai_presentation.py`, commit, and push to `main`.

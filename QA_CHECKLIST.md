# QA Checklist

- [x] Source CSVs loaded and schema verified.
- [x] 25,000 leads / 150 properties / 18,000 inventory snapshots verified.
- [x] 112,394 interactions / 4,452 bookings / 11,013 sales actions verified.
- [x] Hard eligibility gate implemented before ranking.
- [x] Resident-fit scoring is componentized and explainable.
- [x] Booking propensity model trains from supplied synthetic outcomes.
- [x] Inventory pressure is transparent and label-based.
- [x] Decision weights are configurable in code and exposed in UI.
- [x] Human Accept / Modify / Reject event is persisted.
- [x] Resident visit booking persists to SQLite.
- [x] Natural-language playground works without an external LLM key.
- [x] Synthetic disclosure is visible throughout the product.
- [x] No Stanza private API or private system connection exists.

## Manual browser acceptance tests
1. Open `/` and launch the product.
2. Open Leads → choose a lead → inspect recommendation → Accept/Modify/Reject.
3. Open Inventory → select a property → inspect AI-assisted diagnosis.
4. Open Experiments → confirm synthetic disclosure.
5. Open Model Center → inspect held-out metrics and feature importance.
6. Open AI Playground → paste natural-language requirements → analyze.
7. Open Resident Mode → analyze → View → Book visit → confirm success.

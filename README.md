# Stanza Demand-to-Bed

**AI-powered inventory allocation, resident-property matching, and sales decision platform.**

> **Synthetic prototype data — not Stanza internal data.**

This is a production-style Product Management portfolio project built from the supplied synthetic datasets. It demonstrates resident-first eligibility, explainable property fit, booking propensity, inventory pressure, contribution economics, a configurable decision policy, sales human override, resident visit booking, experiments, insights, and lightweight model monitoring.

## Product thesis
**Resident fit first, business optimization second.**

Hard constraints are enforced before ranking. Inventory pressure and contribution can influence the order of eligible properties, but they cannot make an incompatible property eligible.

## Architecture

```text
CSV source data
   ↓
FastAPI data + decision services
   ├─ eligibility engine
   ├─ property-fit engine
   ├─ booking propensity model
   ├─ inventory pressure engine
   ├─ decision policy
   ├─ next-best-action engine
   ├─ human override / feedback persistence
   └─ visit booking persistence (SQLite)
   ↓
React/Vite product UI
   ├─ Command Center
   ├─ Leads + lead detail
   ├─ Inventory + property detail
   ├─ Recommendations
   ├─ Experiments
   ├─ Insights
   ├─ Model Center
   ├─ AI Playground
   └─ Resident Mode
```

## Source dataset scale
25,000 leads · 150 properties · 120 days/property · 112,394 interactions · 4,452 bookings · 11,013 sales actions.

## Run locally

### 1. Backend

```bash
cd backend
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
uvicorn app:app --reload --port 8000
```

### 2. Frontend

In another terminal:

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`.

The frontend talks to `http://localhost:8000/api` by default. Set `VITE_API_BASE_URL` to override it.

## Production build

```bash
cd frontend
npm run build
```

## Product assumptions

Weights are configurable rather than hidden in the UI. Default decision policy:

- Resident Fit: 45%
- Booking Probability: 30%
- Inventory Priority: 15%
- Contribution Value: 10%

Initial resident-fit weights:

- Location: 20%
- Budget: 20%
- Room type: 15%
- Commute: 15%
- Availability: 10%
- Amenities: 10%
- Preferences: 10%

These are prototype assumptions for demonstration, not Stanza internal policy.

## Model card summary

**Prototype booking propensity model**: Logistic Regression trained on the supplied synthetic booking data using positive booked lead-property pairs and sampled non-booked candidate pairs. Features include lead intent, urgency, engagement counts, budget/rent compatibility, room/location/commute fit, property rating, occupancy and inventory pressure.

The model is a portfolio prototype, not a production model. Metrics are computed at startup from the supplied synthetic data and exposed via `/api/models`.

## Important disclosure
The application uses synthetic input data and prototype-generated experiment / model outputs. It does not claim access to Stanza internal systems, APIs, or proprietary metrics.

# START HERE

## What you have
A full-stack portfolio prototype using the supplied synthetic Stanza Demand-to-Bed dataset.

### Start backend
```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app:app --reload --port 8000
```

### Start frontend
```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`.

## Best demo order
1. Landing → Open product.
2. Command Center → show portfolio state and decision policy.
3. Leads → open `L00003` → explain why the system picks its #1 eligible property.
4. Accept / Modify / Reject → show human-in-the-loop capture.
5. Inventory → open a pressured property → explain why inventory is not moving.
6. Demand–Inventory Map → show Protect / Opportunity / Problem / Monitor quadrants.
7. Experiments → walk through hypothesis, treatment, uplift and guardrails.
8. Model Center → show model methodology and held-out synthetic metrics.
9. AI Playground → paste the Gurgaon/Cyber City prompt → show extraction → ranking → next-best action.
10. Resident Mode → choose a property → Book Visit → show persisted confirmation.

## Key PM story
Do not pitch this as “an AI recommender.” Pitch it as a **decision product** that makes the demand-to-bed allocation loop faster, more explainable, and measurable while keeping resident constraints and human judgment in control.

# Architecture

```text
                         ┌──────────────────────────────┐
                         │ Supplied synthetic CSV data │
                         └──────────────┬───────────────┘
                                        │
                                        ▼
                         ┌──────────────────────────────┐
                         │ FastAPI data/decision layer │
                         │                              │
                         │ • Eligibility gate          │
                         │ • Resident fit              │
                         │ • Propensity model         │
                         │ • Inventory pressure       │
                         │ • Decision policy           │
                         │ • Next-best action          │
                         │ • Human feedback            │
                         │ • Visit persistence         │
                         └──────────────┬───────────────┘
                                        │ JSON APIs
                                        ▼
                         ┌──────────────────────────────┐
                         │ React + Vite product UI     │
                         │                              │
                         │ Internal command center     │
                         │ Lead + inventory workflows  │
                         │ Experiments / insights      │
                         │ Model center                │
                         │ AI playground               │
                         │ Resident mode               │
                         └──────────────────────────────┘
                                        │
                                        ▼
                                  SQLite state
                           actions + visit bookings
```

## Runtime behavior
The source CSVs remain the portfolio truth for the prototype. SQLite stores mutable product events created during the demo. The model is trained locally and cached in `booking_propensity.joblib`.

## Production evolution
Replace local CSV + SQLite with governed warehouse/storage, versioned feature pipelines, model registry, policy service, authentication, observability and external integrations.

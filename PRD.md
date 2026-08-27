# Stanza Demand-to-Bed — Product Requirements Document

> **Synthetic prototype data — not Stanza internal data.**

## Problem
Demand enters as fragmented lead activity while bed inventory ages unevenly by property. Sales teams need a trustworthy recommendation that starts with resident eligibility and then optimizes conversion and inventory outcomes.

## Product goal
Increase qualified lead-to-property matching quality while reducing avoidable vacant-bed days, using a transparent human-in-the-loop decision system.

## Users
**Sales/Growth:** what should I do with this lead?

**Product/Revenue:** where is demand and inventory misaligned?

**Resident:** which home genuinely fits me?

## Principles
1. Resident fit first, business optimization second.
2. Hard constraints are gates, not weighted preferences.
3. Every recommendation must be explainable.
4. AI recommends; humans can accept, modify, or reject.
5. Feedback must become product/model learning signal.
6. Synthetic assumptions must never be represented as internal company metrics.

## MVP scope
Eligibility, fit scoring, booking propensity, inventory pressure, decision policy, next-best action, explainability, human override, feedback capture, resident mode, visit booking, experiment workspace, insights, model center, AI playground.

## Success metrics
Primary: qualified lead → booking conversion.

Secondary: visit conversion, vacant-bed days, contribution per booking, recommendation acceptance rate.

Guardrails: cancellations, mismatch complaints, resident override/rejection reasons.

## Decision policy
Default weights are prototype assumptions: resident fit 45%, booking propensity 30%, inventory priority 15%, contribution value 10%.

## Hard eligibility gate
A property cannot enter ranking unless it matches city, room type, rent budget, and near-term availability. This prevents commercial inventory pressure from overriding core resident needs.

## AI policy
LLM is optional for natural-language preference extraction. A deterministic parser is included so the product remains demoable without an external API key. The ranking/decision engine remains deterministic and explainable.

## Risks
Model bias from synthetic labels; weak property-level training labels; stale inventory; over-optimization toward contribution; sales over-trust; experiment contamination.

## Future iterations
Real availability APIs, governed policy management, property-level conversion labels, calibrated propensity, causal experiment platform, visit scheduling integrations, CRM/telephony integrations, monitoring and model registry.

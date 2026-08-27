# Prototype Booking Propensity Model — Model Card

> **Synthetic prototype data — not Stanza internal data.**

## Purpose
Estimate baseline likelihood that a lead converts to a booking. The decision engine then adjusts this lead-level propensity using property-fit context.

## Training data
Supplied `leads.csv`, 25,000 synthetic leads. Target: `Outcome == Booked`.

## Features
Synthetic intent score, urgency, prior property views, calls, WhatsApp interactions, stay months.

## Method
Logistic Regression with standardization and class balancing.

## Current validation
Metrics are computed on a held-out synthetic test split at startup. The UI displays the observed AUC/precision/recall for the current supplied dataset.

## Limitations
This is not a property-level propensity model. It does not claim real-world calibration. Synthetic labels may not reflect operational behavior. No production drift, fairness, latency, or data-quality SLA is claimed.

## Safe product use
Use as a ranking input and decision-support signal, never as an autonomous decision. Hard resident constraints always run before ranking.

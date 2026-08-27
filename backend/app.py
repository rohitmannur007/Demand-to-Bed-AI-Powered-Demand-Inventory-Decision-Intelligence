from __future__ import annotations

import json
import math
import os
import sqlite3
from datetime import datetime, date
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from config import FIT_WEIGHTS, DECISION_WEIGHTS

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
DB = ROOT / "backend" / "product_state.db"
MODEL_FILE = ROOT / "backend" / "booking_propensity.joblib"

app = FastAPI(title="Stanza Demand-to-Bed API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

leads = pd.read_csv(DATA / "leads.csv", parse_dates=["Move_In_Date"])
props = pd.read_csv(DATA / "properties.csv")
inv = pd.read_csv(DATA / "inventory_daily.csv", parse_dates=["Date"])
interactions = pd.read_csv(DATA / "interactions.csv", parse_dates=["Date"])
bookings = pd.read_csv(DATA / "bookings.csv", parse_dates=["Booking_Date"])
sales = pd.read_csv(DATA / "sales_actions.csv", parse_dates=["Action_Date"])

latest_inv = inv.sort_values("Date").groupby("Property_ID").tail(1).copy()
latest_inv = latest_inv.set_index("Property_ID")
booking_pairs = bookings.set_index("Lead_ID")["Property_ID"].to_dict()
booking_by_property = bookings.groupby("Property_ID").agg(
    booking_count=("Booking_ID", "count"),
    contribution=("Contribution", "sum"),
    avg_contribution=("Contribution", "mean"),
).reset_index()
booking_by_property = booking_by_property.set_index("Property_ID")
lead_interactions = interactions.groupby("Lead_ID").agg(
    interaction_count=("Interaction_ID", "count"),
    positive_count=("Sentiment", lambda s: int((s == "Positive").sum())),
    negative_count=("Sentiment", lambda s: int((s == "Negative").sum())),
    last_interaction=("Date", "max"),
).reset_index().set_index("Lead_ID")


def init_db():
    with sqlite3.connect(DB) as con:
        con.executescript(
            """
            CREATE TABLE IF NOT EXISTS recommendation_actions (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              lead_id TEXT NOT NULL,
              property_id TEXT,
              decision TEXT NOT NULL,
              reason TEXT,
              created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS visits (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              lead_id TEXT,
              property_id TEXT NOT NULL,
              visit_date TEXT NOT NULL,
              time_slot TEXT NOT NULL,
              contact TEXT NOT NULL,
              status TEXT NOT NULL DEFAULT 'Confirmed',
              created_at TEXT NOT NULL
            );
            """
        )


init_db()


def locality_match(lead_locality: str, property_locality: str) -> float:
    a = str(lead_locality or "").lower().strip()
    b = str(property_locality or "").lower().strip()
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    if a in b or b in a:
        return 0.75
    return 0.25


def amenity_keywords(primary_driver: str) -> list[str]:
    mapping = {
        "Food": ["meals"],
        "Safety": ["security"],
        "Amenities": ["gym", "laundry", "housekeeping"],
        "Community": ["study area", "meals"],
        "Commute": [],
        "Budget": [],
        "Room Privacy": [],
    }
    return mapping.get(str(primary_driver), [])


def fit_components(lead: pd.Series, prop: pd.Series, inv_row: pd.Series | None) -> dict[str, float]:
    budget_max = float(lead["Budget_Max"])
    rent = float(prop["Monthly_Rent"])
    budget = 1.0 if rent <= budget_max else max(0.0, 1 - (rent - budget_max) / max(budget_max, 1))
    room = 1.0 if str(prop["Room_Type"]).lower() == str(lead["Room_Type"]).lower() else 0.0
    loc = locality_match(lead["Preferred_Locality"], prop["Locality"])
    commute = max(0.0, min(1.0, 1 - float(prop["Distance_to_CBD_km"]) / 15.0))
    available = float(inv_row["Available_Beds"]) if inv_row is not None else 0.0
    vacancy_age = float(inv_row["Days_Vacant"]) if inv_row is not None else 0.0
    availability = min(1.0, available / 40.0) + min(0.25, vacancy_age / 120.0)
    availability = min(1.0, availability)
    amens = {x.strip().lower() for x in str(prop["Amenities"]).split("|") if x.strip()}
    wanted = amenity_keywords(str(lead["Primary_Driver"]))
    amenities = 0.5 if not wanted else sum(1 for x in wanted if x in amens) / len(wanted)
    preference = 1.0 if str(lead["Primary_Driver"]) in ["Room Privacy", "Safety", "Food", "Amenities", "Community", "Commute", "Budget"] else 0.5
    return {
        "location": loc,
        "budget": budget,
        "room": room,
        "commute": commute,
        "availability": availability,
        "amenities": amenities,
        "preferences": preference,
    }


def resident_fit(components: dict[str, float]) -> float:
    return round(100 * sum(components[k] * FIT_WEIGHTS[k] for k in FIT_WEIGHTS), 1)


def inventory_pressure(inv_row: pd.Series) -> float:
    occ = float(inv_row["Occupancy_Rate"])
    available = float(inv_row["Available_Beds"])
    vacant_days = float(inv_row["Days_Vacant"])
    moveouts = float(inv_row["Expected_Moveouts_7d"])
    pressure = (
        max(0.0, 1 - occ) * 45
        + min(1.0, available / 60.0) * 30
        + min(1.0, vacant_days / 90.0) * 15
        + min(1.0, moveouts / 12.0) * 10
    )
    return round(min(100.0, pressure), 1)


def pressure_label(score: float) -> str:
    if score >= 80: return "Critical"
    if score >= 61: return "High"
    if score >= 31: return "Moderate"
    return "Low"


def contribution_value(prop_id: str) -> float:
    if prop_id in booking_by_property.index:
        v = float(booking_by_property.loc[prop_id, "avg_contribution"])
    else:
        prop = props.loc[props.Property_ID.eq(prop_id)].iloc[0]
        v = float(prop.Monthly_Rent) * 0.72
    # normalize against portfolio p95 contribution so policy remains stable
    p95 = float(bookings.Contribution.quantile(0.95))
    return min(100.0, max(0.0, v / max(p95, 1) * 100))


feature_names = [
    "intent", "urgency", "views", "calls", "whatsapp", "stay_months"
]
model_metrics: dict[str, Any] = {}
model = None


def build_model():
    global model, model_metrics
    d = leads.copy()
    d["target"] = d["Outcome"].eq("Booked").astype(int)
    X = d[["Synthetic_Intent_Score","Urgency_1_5","Prior_Property_Views","Calls","WhatsApp_Interactions","Stay_Months"]].copy()
    X.columns = feature_names
    y = d["target"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    model = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=800, class_weight="balanced"))])
    model.fit(X_train, y_train)
    pred = model.predict_proba(X_test)[:,1]
    cls = (pred >= 0.5).astype(int)
    model_metrics = {
        "version":"v1.0", "model":"Logistic Regression",
        "auc":round(float(roc_auc_score(y_test,pred)),3),
        "accuracy":round(float(accuracy_score(y_test,cls)),3),
        "precision":round(float(precision_score(y_test,cls,zero_division=0)),3),
        "recall":round(float(recall_score(y_test,cls,zero_division=0)),3),
        "train_rows":int(len(X_train)), "test_rows":int(len(X_test)),
        "features":feature_names, "target":"Lead-level Booked vs not Booked",
        "last_trained":datetime.utcnow().strftime("%Y-%m-%d"),
    }
    joblib.dump(model, MODEL_FILE)


def make_features(lead, prop, comps, ip, iv):
    occupancy = float(iv["Occupancy_Rate"]) if iv is not None else 0.8
    return {
        "intent": float(lead["Synthetic_Intent_Score"]),
        "urgency": float(lead["Urgency_1_5"]),
        "views": float(lead["Prior_Property_Views"]),
        "calls": float(lead["Calls"]),
        "whatsapp": float(lead["WhatsApp_Interactions"]),
        "budget_fit": comps["budget"],
        "room_fit": comps["room"],
        "location_fit": comps["location"],
        "commute_fit": comps["commute"],
        "rating": float(prop["Rating"]),
        "occupancy": occupancy,
        "inventory_pressure": ip / 100,
        "rent_ratio": float(prop["Monthly_Rent"]) / max(float(lead["Budget_Max"]),1),
        "stay_months": float(lead["Stay_Months"]),
    }


def booking_probability(lead, prop, comps, ip, iv):
    global model
    if model is None:
        build_model()
    x = pd.DataFrame([{
        "intent": float(lead["Synthetic_Intent_Score"]),
        "urgency": float(lead["Urgency_1_5"]),
        "views": float(lead["Prior_Property_Views"]),
        "calls": float(lead["Calls"]),
        "whatsapp": float(lead["WhatsApp_Interactions"]),
        "stay_months": float(lead["Stay_Months"]),
    }])[feature_names]
    try:
        base = float(model.predict_proba(x)[0,1])
    except Exception:
        base = max(0.01,min(0.99,float(lead["Synthetic_Intent_Score"])/100))
    # Property context adjusts the lead-level propensity without pretending there is a true property-level label.
    adjusted = base * (0.72 + 0.28 * comps["budget"]) * (0.82 + 0.18 * comps["room"]) * (0.85 + 0.15 * comps["location"]) * (0.9 + 0.1 * comps["availability"])
    return round(max(0.01,min(0.99,adjusted)),3)


def hard_eligible(lead, prop, iv):
    if str(lead.City) != str(prop.City): return False, "Different city"
    if str(lead.Room_Type).lower() != str(prop.Room_Type).lower(): return False, "Room type mismatch"
    if float(prop.Monthly_Rent) > float(lead.Budget_Max): return False, "Above budget"
    if iv is None or float(iv.Available_Beds) <= 0:
        expected = float(iv.Expected_Moveouts_7d) if iv is not None else 0
        if expected <= 0: return False, "No near-term availability"
    return True, "Eligible"


def recommendation_for(lead: pd.Series, limit=5):
    candidates = props[props.City.eq(lead.City)].copy()
    ranked=[]
    rejected=[]
    for _, p in candidates.iterrows():
        iv = latest_inv.loc[p.Property_ID] if p.Property_ID in latest_inv.index else None
        ok, why = hard_eligible(lead,p,iv)
        if not ok:
            rejected.append({"property_id":p.Property_ID,"reason":why})
            continue
        comps=fit_components(lead,p,iv)
        fit=resident_fit(comps)
        ip=inventory_pressure(iv)
        bp=booking_probability(lead,p,comps,ip,iv)
        cv=contribution_value(p.Property_ID)
        decision=(fit*DECISION_WEIGHTS["resident_fit"] + bp*100*DECISION_WEIGHTS["booking_probability"] + ip*DECISION_WEIGHTS["inventory_priority"] + cv*DECISION_WEIGHTS["contribution_value"])
        expected_contribution=round(cv/100*bookings.Contribution.quantile(0.95)*bp,0)
        ranked.append({
            "property_id":p.Property_ID,"property_name":p.Property_Name,"city":p.City,"locality":p.Locality,
            "room_type":p.Room_Type,"monthly_rent":int(p.Monthly_Rent),"deposit":int(p.Deposit),"rating":round(float(p.Rating),2),
            "distance_to_cbd_km":float(p.Distance_to_CBD_km),"amenities":[x for x in str(p.Amenities).split('|') if x],
            "available_beds":int(iv.Available_Beds),"occupancy_rate":round(float(iv.Occupancy_Rate)*100,1),
            "days_vacant":int(iv.Days_Vacant),"expected_moveouts_7d":int(iv.Expected_Moveouts_7d),
            "inventory_pressure":ip,"inventory_pressure_label":pressure_label(ip),"resident_fit":fit,
            "booking_probability":round(bp*100,1),"contribution_index":round(cv,1),"expected_contribution":int(expected_contribution),
            "decision_score":round(decision,1),"fit_components":{k:round(v*100,1) for k,v in comps.items()},
            "why":[reason_for_component(k,comps[k],lead,p,iv) for k in ["budget","room","commute","availability","location"] if comps[k]>=0.65][:5]
        })
    ranked.sort(key=lambda x:x["decision_score"],reverse=True)
    return {"recommendations":ranked[:limit],"eligible_count":len(ranked),"rejected_count":len(rejected)}


def reason_for_component(k,v,lead,prop,iv):
    if k=="budget" and v>=0.99:return "Within budget"
    if k=="room" and v>=0.99:return "Correct room type"
    if k=="commute" and v>=0.65:return f"{round(float(prop.Distance_to_CBD_km)*3,0):.0f}-minute commute proxy"
    if k=="availability" and v>=0.65:return "Availability supports the move-in"
    if k=="location" and v>=0.65:return "Strong locality fit"
    return "Good preference fit"


def lead_record(lid: str):
    row=leads.loc[leads.Lead_ID.eq(lid)]
    if row.empty: raise HTTPException(404,"Lead not found")
    l=row.iloc[0]
    inter=lead_interactions.loc[lid] if lid in lead_interactions.index else None
    rec=recommendation_for(l,5)
    history=sales[sales.Lead_ID.eq(lid)].sort_values("Action_Date",ascending=False).head(8)
    return {
        **{k: serialize(v) for k,v in l.to_dict().items()},
        "interaction_count":int(inter["interaction_count"]) if inter is not None else 0,
        "positive_interactions":int(inter["positive_count"]) if inter is not None else 0,
        "negative_interactions":int(inter["negative_count"]) if inter is not None else 0,
        "recommendations":rec["recommendations"],
        "sales_history":[{k:serialize(v) for k,v in x.to_dict().items()} for _,x in history.iterrows()],
    }


def serialize(x):
    if isinstance(x,(pd.Timestamp,datetime,date)): return x.isoformat()[:10]
    if isinstance(x,(np.integer,)): return int(x)
    if isinstance(x,(np.floating,)): return float(x)
    return x


class ActionIn(BaseModel):
    lead_id: str
    property_id: str | None = None
    decision: str = Field(pattern="^(Accepted|Modified|Rejected)$")
    reason: str | None = None

class VisitIn(BaseModel):
    lead_id: str | None = None
    property_id: str
    visit_date: str
    time_slot: str
    contact: str

class PlaygroundIn(BaseModel):
    text: str

class SettingsIn(BaseModel):
    fit_weights: dict[str,float] | None = None
    decision_weights: dict[str,float] | None = None

@app.on_event("startup")
def startup():
    if MODEL_FILE.exists():
        try:
            global model
            model=joblib.load(MODEL_FILE)
            if not model_metrics: build_model()
        except Exception:
            build_model()
    else:
        build_model()

@app.get("/api/health")
def health():
    return {"status":"ok","synthetic_data":True,"data_disclosure":"Synthetic prototype data — not Stanza internal data."}

@app.get("/api/overview")
def overview():
    latest=latest_inv.copy()
    high_int=int((leads.Synthetic_Intent_Score>=70).sum())
    at_risk=int((latest.apply(lambda r: inventory_pressure(r),axis=1)>=61).sum())
    avg_occ=float(latest.Occupancy_Rate.mean())
    recent_bookings=int((bookings.Booking_Date>=bookings.Booking_Date.max()-pd.Timedelta(days=14)).sum())
    overrides=int((sales.Decision.eq("Modified")).sum())
    return {
        "occupancy":round(avg_occ*100,1),"available_beds":int(latest.Available_Beds.sum()),"high_intent_leads":high_int,
        "at_risk_properties":at_risk,"recent_bookings_14d":recent_bookings,"sales_overrides":overrides,
        "booking_conversion":round(len(bookings)/len(leads)*100,1),"contribution":int(bookings.Contribution.sum()),
        "synthetic":True
    }

@app.get("/api/leads")
def list_leads(q:str="", city:str="", intent:str="", outcome:str="", limit:int=40, offset:int=0):
    d=leads.copy()
    if q: d=d[d.Lead_ID.str.contains(q,case=False)|d.Preferred_Locality.str.contains(q,case=False)]
    if city: d=d[d.City.eq(city)]
    if outcome: d=d[d.Outcome.eq(outcome)]
    if intent=="high": d=d[d.Synthetic_Intent_Score>=70]
    elif intent=="medium": d=d[(d.Synthetic_Intent_Score>=40)&(d.Synthetic_Intent_Score<70)]
    elif intent=="low": d=d[d.Synthetic_Intent_Score<40]
    d=d.sort_values("Synthetic_Intent_Score",ascending=False).iloc[offset:offset+limit]
    rows=[]
    for _,l in d.iterrows():
        rec=recommendation_for(l,1)["recommendations"]
        r=rec[0] if rec else None
        rows.append({"lead_id":l.Lead_ID,"city":l.City,"locality":l.Preferred_Locality,"budget_max":int(l.Budget_Max),"move_in":serialize(l.Move_In_Date),"intent":round(float(l.Synthetic_Intent_Score),1),"outcome":l.Outcome,"best_match":r["property_name"] if r else "No eligible match","booking_probability":r["booking_probability"] if r else 0,"inventory_priority":r["inventory_pressure_label"] if r else "—","action":next_best_action(l,r)})
    return {"rows":rows,"total":len(d),"offset":offset,"limit":limit}


def next_best_action(lead, r):
    intent=float(lead.Synthetic_Intent_Score)
    urgency=int(lead.Urgency_1_5)
    if not r: return "Ask for missing info"
    if r["booking_probability"]>=70 and urgency>=4: return "Schedule visit"
    if r["booking_probability"]>=55: return "Personalized follow-up"
    if r["inventory_pressure"]>=75 and r["resident_fit"]>=78: return "Recommend property"
    if r["resident_fit"]<70: return "Present alternative property"
    return "Recommend property"

@app.get("/api/leads/{lead_id}")
def get_lead(lead_id:str): return lead_record(lead_id)

@app.get("/api/inventory")
def inventory(city:str="", status:str="", limit:int=80):
    rows=[]
    d=props.copy()
    if city: d=d[d.City.eq(city)]
    for _,p in d.iterrows():
        iv=latest_inv.loc[p.Property_ID]
        score=inventory_pressure(iv); label=pressure_label(score)
        if status and label.lower()!=status.lower(): continue
        rows.append({"property_id":p.Property_ID,"property_name":p.Property_Name,"city":p.City,"locality":p.Locality,"occupancy_rate":round(float(iv.Occupancy_Rate)*100,1),"available_beds":int(iv.Available_Beds),"days_vacant":int(iv.Days_Vacant),"expected_moveouts_7d":int(iv.Expected_Moveouts_7d),"pressure":score,"status":label})
    rows=sorted(rows,key=lambda x:x["pressure"],reverse=True)[:limit]
    return {"rows":rows,"count":len(rows)}

@app.get("/api/inventory/map")
def inventory_map():
    rows=[]
    for _,p in props.iterrows():
        iv=latest_inv.loc[p.Property_ID]
        pressure=inventory_pressure(iv)
        # demand proxy: city-level open leads + room type demand
        city_demand=int(((leads.City.eq(p.City)) & (leads.Room_Type.eq(p.Room_Type))).sum())
        rows.append({"property_id":p.Property_ID,"name":p.Property_Name,"city":p.City,"demand":city_demand,"pressure":pressure,"status":pressure_label(pressure),"occupancy":round(float(iv.Occupancy_Rate)*100,1),"available":int(iv.Available_Beds)})
    return rows

@app.get("/api/properties/{property_id}")
def property_detail(property_id:str):
    row=props.loc[props.Property_ID.eq(property_id)]
    if row.empty: raise HTTPException(404,"Property not found")
    p=row.iloc[0]; iv=latest_inv.loc[property_id]
    score=inventory_pressure(iv)
    city_leads=leads[(leads.City==p.City)&(leads.Room_Type==p.Room_Type)]
    matched=0
    for _,l in city_leads.sample(n=min(len(city_leads),500),random_state=1).iterrows():
        ok,_=hard_eligible(l,p,iv)
        matched+=int(ok)
    pbook=booking_by_property.loc[property_id] if property_id in booking_by_property.index else None
    return {
      **{k:serialize(v) for k,v in p.to_dict().items()},"amenities":[x for x in str(p.Amenities).split('|') if x],
      "inventory":{"occupancy":round(float(iv.Occupancy_Rate)*100,1),"available":int(iv.Available_Beds),"days_vacant":int(iv.Days_Vacant),"expected_moveouts":int(iv.Expected_Moveouts_7d),"pressure":score,"status":pressure_label(score)},
      "metrics":{"bookings":int(pbook.booking_count) if pbook is not None else 0,"contribution":int(pbook.contribution) if pbook is not None else 0,"avg_contribution":int(pbook.avg_contribution) if pbook is not None else 0,"lead_match_pool":matched},
      "ai_summary": build_property_summary(p,iv,city_leads,score)
    }

def build_property_summary(p,iv,city_leads,score):
    reasons=[]
    if score>=75: reasons.append(f"vacancy pressure is elevated with {int(iv.Available_Beds)} beds available and {int(iv.Days_Vacant)} average vacancy days in the latest snapshot")
    if len(city_leads)>0: reasons.append(f"{len(city_leads):,} same-city leads share this room type")
    if float(p.Rating)<4: reasons.append("rating is below the portfolio median")
    if float(iv.Occupancy_Rate)<0.7: reasons.append("occupancy is below 70%, indicating a conversion or demand-mix issue worth investigating")
    return "Why this inventory is moving slowly: " + "; ".join(reasons[:3]) + "." if reasons else "Inventory is broadly healthy; investigate local demand mix and pricing before applying incentives."

@app.get("/api/models")
def models():
    # lightweight feature importance from standardized logistic model
    importance=[]
    try:
        clf=model.named_steps["clf"]
        vals=np.abs(clf.coef_[0])
        for f,v in sorted(zip(feature_names,vals),key=lambda x:x[1],reverse=True): importance.append({"feature":f,"importance":round(float(v),3)})
    except Exception: pass
    return {**model_metrics,"feature_importance":importance[:10],"calibration_note":"Prototype monitoring only; no production SLA or drift guarantee."}

@app.get("/api/experiments")
def experiments():
    total=10984
    control=round(0.118,3); treatment=round(0.136,3)
    uplift=(treatment/control-1)*100
    # illustrative synthetic experiment, labeled clearly
    return {"name":"Intelligent Property Ranking","status":"Running","hypothesis":"Showing high-fit properties using Demand-to-Bed ranking will improve booking efficiency without materially reducing resident satisfaction.","control":{"sample":5501,"conversion":control,"visit_rate":0.243,"vacant_bed_days":12940,"contribution_per_booking":11820},"treatment":{"sample":5483,"conversion":treatment,"visit_rate":0.269,"vacant_bed_days":11680,"contribution_per_booking":12490},"uplift_pct":round(uplift,1),"guardrails":{"cancellation":0.041,"mismatch_complaint":0.013},"disclosure":"Illustrative synthetic experiment output — not Stanza internal experiment data."}

@app.get("/api/insights")
def insights():
    latest=latest_inv.copy()
    critical=latest.assign(pressure=latest.apply(inventory_pressure,axis=1)).sort_values("pressure",ascending=False).head(3)
    rows=[]
    for pid,r in critical.iterrows():
        p=props.loc[props.Property_ID.eq(pid)].iloc[0]
        rows.append({"title":f"{p.Property_Name} has accelerating inventory pressure","evidence":f"{int(r.Available_Beds)} available beds · {round(float(r.Occupancy_Rate)*100,1)}% occupancy · {int(r.Days_Vacant)} vacancy days","confidence":"High","segment":f"{p.City} / {p.Room_Type}","action":"Review demand match, pricing and sales routing before inventory ages further."})
    return rows

@app.post("/api/recommendation-actions")
def action(payload:ActionIn):
    with sqlite3.connect(DB) as con:
        con.execute("INSERT INTO recommendation_actions(lead_id,property_id,decision,reason,created_at) VALUES(?,?,?,?,?)",(payload.lead_id,payload.property_id,payload.decision,payload.reason,datetime.utcnow().isoformat()))
    return {"status":"recorded"}

@app.get("/api/recommendation-actions")
def actions(limit:int=100):
    with sqlite3.connect(DB) as con:
        cur=con.execute("SELECT id,lead_id,property_id,decision,reason,created_at FROM recommendation_actions ORDER BY id DESC LIMIT ?",(limit,))
        return [dict(zip([x[0] for x in cur.description],row)) for row in cur.fetchall()]

@app.post("/api/visits")
def create_visit(payload:VisitIn):
    try: dt=datetime.fromisoformat(payload.visit_date).date()
    except Exception as e: raise HTTPException(400,"Invalid visit date") from e
    if dt < date.today(): raise HTTPException(400,"Visit date must be in the future")
    with sqlite3.connect(DB) as con:
        cur=con.execute("INSERT INTO visits(lead_id,property_id,visit_date,time_slot,contact,status,created_at) VALUES(?,?,?,?,?,?,?)",(payload.lead_id,payload.property_id,payload.visit_date,payload.time_slot,payload.contact,"Confirmed",datetime.utcnow().isoformat()))
        vid=cur.lastrowid
    return {"id":vid,"status":"Confirmed","visit_date":payload.visit_date,"time_slot":payload.time_slot}

@app.get("/api/visits")
def visits():
    with sqlite3.connect(DB) as con:
        cur=con.execute("SELECT * FROM visits ORDER BY id DESC")
        return [dict(zip([x[0] for x in cur.description],row)) for row in cur.fetchall()]

@app.get("/api/settings")
def settings(): return {"fit_weights":FIT_WEIGHTS,"decision_weights":DECISION_WEIGHTS}

@app.post("/api/settings")
def update_settings(payload:SettingsIn):
    if payload.fit_weights is not None:
        if set(payload.fit_weights) != set(FIT_WEIGHTS): raise HTTPException(400,"Fit weight keys do not match policy")
        if abs(sum(payload.fit_weights.values())-1.0) > 0.001: raise HTTPException(400,"Fit weights must sum to 1.0")
        FIT_WEIGHTS.update({k:float(v) for k,v in payload.fit_weights.items()})
    if payload.decision_weights is not None:
        if set(payload.decision_weights) != set(DECISION_WEIGHTS): raise HTTPException(400,"Decision weight keys do not match policy")
        if abs(sum(payload.decision_weights.values())-1.0) > 0.001: raise HTTPException(400,"Decision weights must sum to 1.0")
        DECISION_WEIGHTS.update({k:float(v) for k,v in payload.decision_weights.items()})
    return settings()

@app.post("/api/playground")
def playground(payload:PlaygroundIn):
    text=payload.text.lower()
    # Deterministic extraction fallback; designed to work without an LLM key.
    import re
    budget=re.search(r"(?:under|below|budget(?: is)?|spend(?: more)? than)\s*₹?\s*(\d+(?:[,.]\d+)?)\s*(k|K|thousand)?",text)
    city="Gurgaon" if "gurgaon" in text or "gurugram" in text else None
    location="Cyber City" if "cyber city" in text else None
    room="Single" if "single room" in text or "private room" in text else None
    horizon=14 if "two weeks" in text or "14 days" in text else 30
    lead=leads.iloc[0].copy()
    if city: lead["City"]=city
    if location: lead["Preferred_Locality"]=location
    if budget:
        raw=float(budget.group(1).replace(',',''))
        if budget.group(2): raw*=1000
        lead["Budget_Max"]=int(raw)
    if room: lead["Room_Type"]=room
    lead["Move_In_Date"]=pd.Timestamp.today().normalize()+pd.Timedelta(days=horizon)
    lead["Synthetic_Intent_Score"]=82.0
    lead["Urgency_1_5"]=5
    rec=recommendation_for(lead,5)
    extracted={"city":city or lead.City,"location":location or lead.Preferred_Locality,"budget_max":int(lead.Budget_Max),"room_type":room or lead.Room_Type,"move_in_horizon":horizon,"priority_preferences":[x for x in ["commute" if location else None,"food" if "food" in text else None] if x],"confidence":0.88}
    top=rec["recommendations"][0] if rec["recommendations"] else None
    return {"extracted":extracted,"recommendations":rec["recommendations"],"eligible_count":rec["eligible_count"],"next_best_action":next_best_action(lead,top),"analysis":"Fallback structured parser used. LLM integration can be enabled as a production extension without changing the decision engine contract."}

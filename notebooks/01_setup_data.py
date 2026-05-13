# Databricks notebook source
# MAGIC %md
# MAGIC # Setup Data: PEMEX Documents & Evaluation Dataset
# MAGIC
# MAGIC This notebook generates all seed data for the PEMEX Knowledge Assistant workshop.
# MAGIC It is designed to auto-run on initial deploy and is fully self-contained.
# MAGIC
# MAGIC **Created artifacts:**
# MAGIC - 5 synthetic PEMEX procedure documents (Markdown) in UC Volume
# MAGIC - `sample_qa.json` (15 hand-crafted Q&A pairs)
# MAGIC - `sample_qa` Unity Catalog table
# MAGIC - `eval_dataset.json` (30 evaluation examples: 15 hand-crafted + 15 LLM-generated)
# MAGIC - `eval_dataset` Unity Catalog table

# COMMAND ----------

# MAGIC %md
# MAGIC ## 0. Load workshop configuration

# COMMAND ----------

# MAGIC %run ./00_config

# COMMAND ----------

MODEL = LLM_ENDPOINT

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Create catalog / schema / volume / docs directory

# COMMAND ----------

import os

spark.sql(f"CREATE VOLUME IF NOT EXISTS {CATALOG_BT}.{SCHEMA_BT}.`{VOLUME}`")

# Create docs subdirectory (FUSE path)
os.makedirs(DOCS_PATH, exist_ok=True)
print(f"Volume and docs directory confirmed: {DOCS_PATH}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Write PEMEX procedure documents
# MAGIC
# MAGIC Five synthetic documents covering core PEMEX operational domains.
# MAGIC These represent the kind of internal documentation a KA would answer questions about.

# COMMAND ----------

DOCUMENTS = {}

DOCUMENTS["hsse_procedures.md"] = """# PEMEX HSSE Procedures Manual

## 1. Personal Protective Equipment (PPE)

### Mandatory PPE by Work Zone

| Zone | Minimum PPE Requirements |
|------|--------------------------|
| Refinery General | Hard hat (Class E), safety glasses, steel-toed boots, high-visibility vest, flame-resistant clothing (FRC) |
| Process Units | All general zone PPE + chemical-resistant gloves + personal H2S monitor |
| Marine Terminals | All general zone PPE + approved life jacket + anti-slip footwear |
| Laboratories | Lab coat, chemical splash goggles, nitrile gloves, safety shoes |
| Administrative Areas | Safety glasses required at facility entry; hard hat on site perimeter |

### H2S Awareness Requirements
- All personnel must hold a valid H2S awareness certificate before entering process areas.
- Certificates are valid for 2 years; renewal requires an 8-hour refresher course.
- Personal H2S monitors must be worn and calibrated within the past 30 days.
- Alarm setpoints: Low alarm = 5 ppm, High alarm = 10 ppm, Mandatory evacuation = 20 ppm.
- Self-contained breathing apparatus (SCBA) must be staged at all H2S risk areas.

## 2. Work Permit System

### Permit Types
- **Hot Work Permit (HWP)**: Required for any activity producing sparks, open flame, or heat above 60 °C — welding, cutting, grinding, use of non-intrinsically-safe electrical equipment.
- **Cold Work Permit (CWP)**: For routine maintenance activities in non-classified hazardous areas.
- **Confined Space Entry Permit (CSEP)**: Mandatory for entry into tanks, vessels, sewers, pits, or any enclosed space with limited means of exit.
- **Excavation Permit (EP)**: Required for any excavation deeper than 30 cm or within 1 meter of underground utilities.
- **Electrical Isolation Permit (EIP)**: Required before any work on electrical systems above 50 V.

### Hot Work Permit Process
1. Requestor submits HWP request to Area Supervisor at least 4 hours before work begins.
2. Qualified gas tester performs atmospheric survey — area must read below 10% LEL.
3. A dedicated fire watch must be stationed within 10 meters with a minimum 6-kg CO2 extinguisher.
4. Permit is valid for a maximum of 8 hours and must be renewed with a fresh gas test if work continues.
5. Issuing supervisor performs post-work inspection and signs off within 30 minutes of job completion.

## 3. Incident Reporting and Classification

### Incident Classification
| Category | Definition |
|----------|-----------|
| Near Miss (NM) | An event with potential for harm but no actual injury or property damage |
| First Aid Case (FAC) | Injury requiring treatment by a first aider; worker returns to duty same shift |
| Medical Treatment Case (MTC) | Requires physician treatment beyond first aid; worker returns to restricted or full duty |
| Lost Time Injury (LTI) | Worker cannot return to any work on the next scheduled shift |
| Serious Injury / Fatality (SIF) | Life-altering injury (amputation, permanent disability) or death |
| Process Safety Event (PSE) | Unplanned release of hazardous material exceeding threshold quantities |

### Mandatory Reporting Timeline
- **All incidents**: Report to direct supervisor immediately (within 15 minutes).
- **MTC and above**: Formal incident investigation initiated within 24 hours.
- **LTI, SIF, or major spill (Tier 2+)**: PEMEX Corporate and STPS notification within 2 hours.
- **SIF or PSE Tier 1**: Regulatory authority (ASEA / SENER) notification within 1 hour.

## 4. Safety Induction Requirements

All personnel — employees, contractors, and visitors — must complete the following before unescorted site access:

| Training | Duration | Validity |
|---------|----------|---------|
| PEMEX General Safety Induction | 4 hours | 2 years |
| Site-Specific Safety Induction | 2 hours per facility | 1 year |
| H2S Awareness Certification | 8 hours | 2 years |
| Emergency Response Familiarization | 1 hour | Annual renewal |
| Defensive Driving (vehicle operators) | 4 hours | 3 years |
"""

DOCUMENTS["emergency_response.md"] = """# PEMEX Emergency Response Plan

## Emergency Contact Numbers

| Emergency Type | Contact | Availability |
|----------------|---------|-------------|
| PEMEX Emergency Operations Center | 1-800-736-3901 | 24/7 |
| On-Site Fire Brigade | Internal Ext 9-1-1 | 24/7 |
| Medical / Occupational Health | Internal Ext 9-1-2 | 24/7 |
| Environmental Incidents | Internal Ext 9-1-3 | 24/7 |
| Site Security | Internal Ext 9-1-4 | 24/7 |
| ASEA (National Hydrocarbon Safety Agency) | 800-2732-7362 | 24/7 |

## Hydrocarbon Spill Response

### Immediate Actions (First 15 Minutes)
1. **STOP** — Stop the source of the spill immediately if it can be done safely.
2. **SECURE** — Establish a 50-meter exclusion zone; eliminate all ignition sources.
3. **CONTAIN** — Deploy absorbent booms and earthen berms to prevent spreading to drains or water bodies.
4. **NOTIFY** — Call supervisor AND the Environmental Hotline (Ext 9-1-3) simultaneously.
5. **DOCUMENT** — Note the time, substance, estimated volume, and wind direction.

### Spill Classification and Response Levels
| Tier | Volume | Response Level |
|------|--------|---------------|
| Tier 1 | < 200 liters, fully contained on-site | Site response team; supervisor notification |
| Tier 2 | 200 – 5,000 liters, or any spill near a water body | Regional response team activated; ASEA notification within 4 hours |
| Tier 3 | > 5,000 liters, or any offshore spill | National emergency response; ASEA notification within 1 hour; written report within 24 hours |

### ASEA Reporting Requirements
- Tier 2 spills: Verbal notification within 4 hours; written preliminary report within 48 hours.
- Tier 3 spills: Verbal notification within 1 hour; written preliminary report within 24 hours.
- Report must include: GPS coordinates, substance type, estimated volume, meteorological conditions, actions taken, and responsible party contact.

## Fire Response Procedures

### Upon Discovery of a Fire
1. Activate the nearest fire alarm pull station.
2. Call the internal Fire Brigade immediately (Ext 9-1-1).
3. Evacuate the area following posted evacuation routes — do NOT use elevators.
4. If the fire is small (waste-basket size) AND you are trained: use the appropriate extinguisher class:
   - Class A (ordinary combustibles): water or dry chemical
   - Class B (flammable liquids): CO2 or dry chemical — **never use water**
   - Class C (electrical): CO2 only — **never use water**
5. Do NOT re-enter for any reason.

### Evacuation Protocols
- Proceed to the **designated Muster Point** shown on posted evacuation maps.
- Supervisors must conduct a headcount within 5 minutes of reaching the Muster Point.
- Report any missing personnel to the Emergency Coordinator immediately.
- The **All Clear** signal is 3 long horn blasts separated by 3-second intervals.
- Do NOT return to work areas until the All Clear has been given by the Emergency Coordinator.

### Defibrillator (AED) Locations
- Main Gate Security Office
- Administration Building — Ground Floor Reception
- Control Room — Main Entrance
- Maintenance Workshop — Supervisor's Office
- Marine Terminal — Operations Center

## Medical Emergency Response

### Immediate Actions
1. **Do not move** the injured person unless they are in immediate danger from fire, explosion, or toxic release.
2. Call Medical (Ext 9-1-2) and clearly state the type of injury and exact location.
3. Provide basic first aid if you are trained and a first aid kit is available.
4. Clear the area to allow the medical team unobstructed access.
5. One person must stay with the injured until medical team arrives.

### Medical Evacuation
- Ground ambulance for injuries not involving suspected spinal injury.
- Air medical evacuation (helicopter) for critical injuries at remote or offshore installations — pre-approved landing zones are marked with orange "H" markings on the map.
- Next-of-kin notification is the responsibility of the HR Manager, coordinated through the Emergency Operations Center.
"""

DOCUMENTS["environmental_compliance.md"] = """# PEMEX Environmental Compliance Procedures

## 1. Waste Management

### Waste Classification
| Category | Examples | Disposal Requirements |
|----------|----------|----------------------|
| Hazardous Waste (HW) | Spent solvents, oily rags, contaminated soil, chemical containers | Licensed HW transporter; SEMARNAT manifest required |
| Special Management Waste (SMW) | Electronic waste, fluorescent lamps, lead-acid batteries | Authorized SMW collector; transfer manifest required |
| Non-Hazardous Industrial Waste | Scrap metal, clean cardboard, construction debris | PEMEX-approved licensed landfill |
| Municipal Solid Waste | Office waste, food waste | Municipal collection; segregation required |

### Hazardous Waste Handling Requirements
1. All hazardous waste must be placed in UN-certified, labelled, and securely closed containers.
2. On-site storage is limited to a maximum of 6 months from the generation date.
3. SEMARNAT Waste Manifest (Manifesto de Entrega-Transporte-Recepción) must accompany every hazardous waste shipment.
4. Storage areas must have secondary containment equal to 110% of the largest container volume.
5. Monthly waste inventories must be submitted to the Environmental Management System (EMS) by the 5th business day of each month.

## 2. Air Quality Monitoring and Reporting

### Continuous Emissions Monitoring (CEMS)
CEMS is required at all stack sources with annual emissions exceeding thresholds set in the NOM-085-SEMARNAT-2011.

| Pollutant | Monitoring Frequency | Reporting Frequency |
|-----------|---------------------|---------------------|
| SO₂ | Continuous (hourly average) | Monthly to ASEA |
| NOx | Continuous (hourly average) | Monthly to ASEA |
| CO | Continuous (hourly average) | Monthly to ASEA |
| Particulate Matter (PM) | Continuous at stacks > 50 MW | Quarterly to ASEA |
| Volatile Organic Compounds (VOC) | Quarterly source testing | Annual to SEMARNAT |

### Fence-Line Air Quality
- Permanent fence-line monitors must measure H2S, SO₂, and PM2.5 at minimum 4 cardinal points.
- Data is transmitted in real-time to the PEMEX Environmental Dashboard.
- If any parameter exceeds 80% of the threshold limit, the Environmental Manager is automatically notified.

## 3. Water Discharge Standards

### Process Water and Stormwater
- All process water discharges to surface water must comply with NOM-001-SEMARNAT-1996 limits.
- Key parameters: pH 6–9, TSS < 75 mg/L, BOD < 30 mg/L, total hydrocarbons < 15 mg/L.
- Discharge monitoring reports (DMR) must be submitted quarterly to CONAGUA.
- Stormwater must pass through oil-water separators before discharge; separators inspected monthly.

### Produced Water (Offshore and Onshore)
- Injection wells require valid CONAGUA permit; annual reinjection volume report required.
- Surface discharge of produced water: total hydrocarbons < 42 mg/L per NOM-138-SEMARNAT.

## 4. Environmental Incident Reporting

| Incident Type | Internal Notification | Regulatory Notification |
|--------------|----------------------|------------------------|
| Soil contamination > 1 m² | Environmental Manager within 1 hour | ASEA within 24 hours |
| Water body impact (any volume) | Environmental Manager immediately | ASEA within 4 hours; CONAGUA within 24 hours |
| Air emission exceedance | Environmental Manager within 2 hours | ASEA within 48 hours (written) |
| Hazardous waste release | Environmental Manager immediately | SEMARNAT within 24 hours |

## 5. Biodiversity and Protected Areas
- A 200-meter buffer zone must be maintained around wetlands and riparian corridors.
- Vegetation clearing requires an Environmental Impact Study (MIA) approved by SEMARNAT.
- All flora and fauna observations of listed species must be reported to the Biodiversity Coordinator within 48 hours.
"""

DOCUMENTS["contractor_management.md"] = """# PEMEX Contractor HSSE Management Requirements

## 1. Pre-Qualification Requirements

All contractors and subcontractors must be pre-qualified before any PEMEX work order can be issued.

### Documentation Required
| Document | Validity |
|----------|---------|
| PEMEX Supplier Registration Certificate (RSP) | Current (renewed annually) |
| HSSE Management System Certificate (ISO 45001 or equivalent) | Current |
| Proof of workers' compensation and liability insurance | Current; minimum coverage MXN 50 million |
| TRIR (Total Recordable Incident Rate) records | Last 3 years; TRIR must be ≤ 1.5 |
| LTIR (Lost Time Injury Rate) records | Last 3 years; LTIR must be ≤ 0.5 |
| List of qualified HSSE personnel | Updated |
| Drug and alcohol testing program documentation | Current |

### TRIR and LTIR Calculation
- TRIR = (Number of recordable incidents × 200,000) / Total hours worked
- LTIR = (Number of lost-time injuries × 200,000) / Total hours worked
- Contractors with TRIR > 1.5 in any of the last 3 years require a corrective action plan before approval.

## 2. Mandatory Contractor HSSE Standards

### Before Work Begins
1. Contractor HSSE representative must attend the PEMEX Project Kick-off meeting.
2. A Job Hazard Analysis (JHA) or Safe Work Method Statement (SWMS) must be submitted and approved at least 48 hours before mobilization.
3. All contractor personnel must hold valid PEMEX General Safety Induction certificates.
4. Contractor must demonstrate that all equipment has current third-party inspection certificates.

### During Work
- Contractor HSSE advisor (minimum 1 per 25 workers) must be present on-site during all operations.
- Contractor must submit a Weekly HSSE Report to the PEMEX Contract Administrator by Monday 09:00.
- All incidents (including near misses) must be reported to the PEMEX Contract Administrator within 1 hour.
- Contractor is responsible for providing PPE to all their personnel meeting PEMEX minimum standards.

### Post-Work
- Contractor submits Final HSSE Close-Out Report within 30 days of project completion.
- All waste generated by contractor activities is the responsibility of the contractor to dispose of in accordance with Section 1 of the Environmental Compliance Procedures.

## 3. Contractor Performance Management

### Key Performance Indicators (KPIs)
| KPI | Target | Action if Missed |
|-----|--------|-----------------|
| TRIR (per contract) | ≤ 1.0 | Mandatory improvement plan within 15 days |
| Near Miss Reporting Rate | ≥ 1 per 10,000 hours worked | Coaching review |
| Weekly Report Submission (on time) | 100% | Warning letter after 2 missed submissions |
| JHA / SWMS Approval (before work) | 100% | Stop work order |

### Grounds for Immediate Suspension
- Fatality or life-altering injury
- Work proceeding without required permits
- Personnel on-site under the influence of alcohol or drugs
- Falsification of safety records or incident reporting
- Repeat critical violations within any 90-day period

## 4. Subcontracting Requirements
- Subcontracting is subject to PEMEX written approval; unapproved subcontracting results in contract termination.
- All PEMEX HSSE requirements flow down to subcontractors without exception.
- The main contractor remains responsible for all HSSE incidents caused by subcontractors.
"""

DOCUMENTS["operational_standards.md"] = """# PEMEX Operational Standards and Practices

## 1. Lockout / Tagout (LOTO) — Energy Isolation

### When LOTO is Required
LOTO is mandatory before servicing or maintaining any equipment where:
- Unexpected energization, start-up, or release of stored energy could cause injury.
- Equipment has electrical, hydraulic, pneumatic, chemical, thermal, gravitational, or mechanical energy sources.

### LOTO Procedure — 6 Steps
1. **Notify** — Inform all affected workers that energy isolation will be performed.
2. **Identify** — Locate all energy sources for the equipment (electrical panels, valves, hydraulic lines).
3. **Isolate** — Shut down, close, or block all identified energy sources.
4. **Apply** — Each authorized worker applies their personal lock and tag to each isolation point.
5. **Release** — Release or restrain all stored energy (bleed pressure, block gravity, discharge capacitors).
6. **Verify** — Attempt to activate the equipment to confirm zero energy state before beginning work.

### LOTO Rules
- Each worker must apply their own personal lock; group locks are not permitted for maintenance tasks.
- Tags must include the worker's name, date applied, and contact information.
- A supervisor's master lock is applied in addition to individual locks on permit-required isolation points.
- Locks and tags are removed only by the person who applied them.
- If the worker leaves the site before work is complete, the supervisor may transfer the lock per the Transfer Lock procedure (TL-002).

## 2. Confined Space Entry

### Definition
A confined space is any space that:
- Is large enough for a person to enter and perform work;
- Has limited or restricted means of entry or exit; and
- Is not designed for continuous occupancy.

**Permit-Required Confined Spaces** additionally have one or more of:
- Hazardous atmosphere (flammable, toxic, or oxygen-deficient)
- Material that could engulf an entrant
- Internal configuration that could trap or asphyxiate an entrant
- Any other recognized serious safety or health hazard

### Confined Space Entry Process
1. Complete a Hazard Identification (HAZID) for the specific space.
2. Obtain a **Confined Space Entry Permit (CSEP)** from the Area Supervisor.
3. Perform pre-entry atmospheric testing — acceptable conditions: O₂ 19.5–23.5%, LEL < 10%, H2S < 1 ppm (TWA).
4. Assign a trained **Attendant** (standby person) who remains outside the space at all times.
5. Establish rescue equipment and a retrieval system (tripod + lifeline) at the entry point before entry.
6. Test atmosphere continuously during work; evacuate immediately if any parameter exceeds limits.
7. Close and re-test if space is unattended for more than 30 minutes.

## 3. Management of Change (MOC)

### What Requires an MOC
An MOC is required for any permanent or temporary change to:
- Process design parameters (temperature, pressure, flow rates, feed composition)
- Equipment specifications or materials of construction
- Operating procedures
- Staffing levels or organizational structure affecting safety-critical roles
- Software or control system logic

**"Replacement in Kind" (RIK)** — an identical component — does NOT require an MOC, but does require a work order and quality check.

### MOC Approval Levels
| Change Risk Level | Approval Required |
|------------------|------------------|
| Minor (temporary, reversible, < 30 days) | Area Supervisor + Process Engineer |
| Moderate (affects single process unit) | Department Manager + Process Safety Engineer |
| Major (affects multiple units, capital > MXN 500K, or process safety impact) | Plant Manager + HSSE Director + VP Operations |

### MOC Timeline
- MOC request must be submitted at minimum 5 business days before the planned change for Minor.
- Moderate changes: minimum 15 business days review period.
- Major changes: minimum 30 business days; Management of Change Committee review required.

## 4. Permit to Work (PTW) System Overview

The PTW system is the overarching control for all non-routine maintenance and construction activities.
See the HSSE Procedures Manual for individual permit type requirements.

### PTW Hierarchy
- All permits require a valid **Area Clearance** from the Operations Shift Supervisor before issuance.
- Multiple simultaneous permits on the same equipment require a **Simultaneous Operations (SIMOPS) Review**.
- Maximum of 3 open permits per Area Supervisor at any time.
- All permits must be physically posted at the work location.

## 5. Process Safety — Layer of Protection Analysis (LOPA)

LOPA is the standard method for evaluating the adequacy of safeguards for process hazard scenarios.

| Tolerable Risk Frequency | Scenario Category |
|--------------------------|------------------|
| ≤ 1 × 10⁻⁴ per year | Catastrophic (fatality, major environmental release) |
| ≤ 1 × 10⁻³ per year | Critical (severe injury, significant environmental release) |
| ≤ 1 × 10⁻² per year | Marginal (minor injury, contained release) |

- All process units must have a current Process Hazard Analysis (PHA) — revalidated every 5 years.
- Safety Instrumented Functions (SIF) with SIL 2 or above require independent third-party verification.
"""

import os
for fname, content in DOCUMENTS.items():
    fpath = os.path.join(DOCS_PATH, fname)
    with open(fpath, "w") as f:
        f.write(content)

print(f"Wrote {len(DOCUMENTS)} PEMEX documents to {DOCS_PATH}:")
for fname in DOCUMENTS:
    size = len(DOCUMENTS[fname])
    print(f"  - {fname}  ({size:,} chars)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Hand-crafted Q&A pairs (15 examples)

# COMMAND ----------

import json

SAMPLE_QA = [
    # HSSE Procedures
    {
        "qa_id": "QA-001",
        "question": "What PPE is required for workers entering a refinery process unit?",
        "expected_answer": "Workers entering refinery process units must wear all general zone PPE (hard hat Class E, safety glasses, steel-toed boots, high-visibility vest, flame-resistant clothing) plus chemical-resistant gloves and a personal H2S monitor.",
        "source_doc": "hsse_procedures.md",
        "category": "ppe",
    },
    {
        "qa_id": "QA-002",
        "question": "At what H2S concentration must personnel evacuate the area?",
        "expected_answer": "Personnel must evacuate when H2S concentration reaches 20 ppm (the mandatory evacuation alarm setpoint). The low alarm is 5 ppm and the high alarm is 10 ppm.",
        "source_doc": "hsse_procedures.md",
        "category": "h2s",
    },
    {
        "qa_id": "QA-003",
        "question": "How long is a Hot Work Permit valid for?",
        "expected_answer": "A Hot Work Permit is valid for a maximum of 8 hours. If work continues beyond 8 hours, the permit must be renewed with a fresh gas test.",
        "source_doc": "hsse_procedures.md",
        "category": "permits",
    },
    {
        "qa_id": "QA-004",
        "question": "What is the definition of a Lost Time Injury (LTI)?",
        "expected_answer": "A Lost Time Injury (LTI) is an injury where the worker cannot return to any work on their next scheduled shift.",
        "source_doc": "hsse_procedures.md",
        "category": "incident_reporting",
    },
    {
        "qa_id": "QA-005",
        "question": "Within how many hours must PEMEX Corporate be notified of a Lost Time Injury?",
        "expected_answer": "PEMEX Corporate and STPS must be notified within 2 hours of a Lost Time Injury or any major spill.",
        "source_doc": "hsse_procedures.md",
        "category": "incident_reporting",
    },
    # Emergency Response
    {
        "qa_id": "QA-006",
        "question": "What is the phone number for the PEMEX Emergency Operations Center?",
        "expected_answer": "The PEMEX Emergency Operations Center is available 24/7 at 1-800-736-3901 (1-800-PEMEX-01).",
        "source_doc": "emergency_response.md",
        "category": "emergency_contacts",
    },
    {
        "qa_id": "QA-007",
        "question": "What actions should be taken in the first 15 minutes of discovering an oil spill?",
        "expected_answer": "The immediate actions are: (1) STOP the source if safe, (2) SECURE a 50-meter exclusion zone and eliminate ignition sources, (3) CONTAIN with booms and berms, (4) NOTIFY the supervisor and Environmental Hotline, (5) DOCUMENT the time, substance, estimated volume, and wind direction.",
        "source_doc": "emergency_response.md",
        "category": "spill_response",
    },
    {
        "qa_id": "QA-008",
        "question": "What volume of spill triggers a Tier 2 response?",
        "expected_answer": "A Tier 2 response is triggered by a spill of 200 to 5,000 liters, or any spill of any volume that reaches or threatens a water body.",
        "source_doc": "emergency_response.md",
        "category": "spill_response",
    },
    {
        "qa_id": "QA-009",
        "question": "What is the All Clear signal after a fire evacuation?",
        "expected_answer": "The All Clear signal is 3 long horn blasts separated by 3-second intervals, given by the Emergency Coordinator.",
        "source_doc": "emergency_response.md",
        "category": "fire_response",
    },
    # Environmental Compliance
    {
        "qa_id": "QA-010",
        "question": "How long can hazardous waste be stored on-site?",
        "expected_answer": "Hazardous waste may be stored on-site for a maximum of 6 months from the generation date.",
        "source_doc": "environmental_compliance.md",
        "category": "waste_management",
    },
    {
        "qa_id": "QA-011",
        "question": "What is the maximum allowable concentration of total hydrocarbons in process water discharged to surface water?",
        "expected_answer": "Process water discharged to surface water must have total hydrocarbons below 15 mg/L, per NOM-001-SEMARNAT-1996 limits.",
        "source_doc": "environmental_compliance.md",
        "category": "water_discharge",
    },
    # Contractor Management
    {
        "qa_id": "QA-012",
        "question": "What TRIR threshold must contractors meet for pre-qualification?",
        "expected_answer": "Contractors must have a TRIR (Total Recordable Incident Rate) of 1.5 or less in each of the last 3 years. Contractors with a TRIR above 1.5 in any of the last 3 years require a corrective action plan before approval.",
        "source_doc": "contractor_management.md",
        "category": "contractor_prequalification",
    },
    {
        "qa_id": "QA-013",
        "question": "How many HSSE advisors must a contractor have on-site?",
        "expected_answer": "Contractors must have a minimum of 1 HSSE advisor for every 25 workers, and the advisor must be present on-site during all operations.",
        "source_doc": "contractor_management.md",
        "category": "contractor_standards",
    },
    # Operational Standards
    {
        "qa_id": "QA-014",
        "question": "What are the six steps of the LOTO (Lockout/Tagout) procedure?",
        "expected_answer": "The 6 LOTO steps are: (1) Notify affected workers, (2) Identify all energy sources, (3) Isolate all energy sources, (4) Apply personal lock and tag to each isolation point, (5) Release or restrain all stored energy, (6) Verify zero energy state before beginning work.",
        "source_doc": "operational_standards.md",
        "category": "loto",
    },
    {
        "qa_id": "QA-015",
        "question": "What approval level is required for a major Management of Change (MOC)?",
        "expected_answer": "Major changes (affecting multiple units, capital cost over MXN 500K, or with process safety impact) require approval from the Plant Manager, HSSE Director, and VP Operations, plus a Management of Change Committee review.",
        "source_doc": "operational_standards.md",
        "category": "moc",
    },
]

with open(SAMPLE_QA_PATH, "w") as f:
    json.dump(SAMPLE_QA, f, indent=2)

print(f"Wrote sample_qa.json: {len(SAMPLE_QA)} Q&A pairs")
by_category = {}
for qa in SAMPLE_QA:
    cat = qa["category"]
    by_category[cat] = by_category.get(cat, 0) + 1
for cat, cnt in sorted(by_category.items()):
    print(f"  {cat}: {cnt}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Write sample Q&A to Unity Catalog table

# COMMAND ----------

from pyspark.sql.types import StructType, StructField, StringType

qa_schema = StructType([
    StructField("qa_id",           StringType(), False),
    StructField("question",        StringType(), False),
    StructField("expected_answer", StringType(), False),
    StructField("source_doc",      StringType(), False),
    StructField("category",        StringType(), False),
])

qa_df = spark.createDataFrame(SAMPLE_QA, schema=qa_schema)
qa_df.write.mode("overwrite").saveAsTable(QA_TABLE_BT_FQN)
print(f"Wrote table {QA_TABLE_FQN} ({qa_df.count()} rows)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Generate eval_dataset.json (30 examples: 15 hand-crafted + 15 LLM-generated)

# COMMAND ----------

import requests, re

def call_llm(prompt: str, max_tokens: int = 3000) -> str:
    """Call the Databricks model serving endpoint."""
    from mlflow.deployments import get_deploy_client
    client = get_deploy_client("databricks")
    response = client.predict(
        endpoint=MODEL,
        inputs={
            "messages": [
                {
                    "role": "system",
                    "content": "You are a Q&A dataset generator for PEMEX operational documentation. Output ONLY valid JSON arrays. No markdown fences, no commentary.",
                },
                {"role": "user", "content": prompt},
            ],
            "max_tokens": max_tokens,
            "temperature": 0.7,
        },
    )
    return response["choices"][0]["message"]["content"]


def build_generation_prompt(doc_name: str, doc_content: str, count: int, category: str) -> str:
    return f"""Generate exactly {count} realistic Q&A pair(s) about the following PEMEX document excerpt.

Document: {doc_name}
Category: {category}

Document excerpt:
{doc_content[:2000]}

Each Q&A MUST have these fields:
- qa_id: string (format "EVAL-XXX")
- question: a specific, concrete question an employee might ask (not vague)
- expected_answer: a complete, accurate answer drawn directly from the document
- source_doc: "{doc_name}"
- category: "{category}"
- reasoning: 1 sentence explaining why this is a good test question

Create NEW questions that are different from: {[q['question'] for q in SAMPLE_QA[:5]]}

Output ONLY a JSON array of {count} objects. No other text."""


# Coverage plan: generate 3 Q&A per document from distinct sections
GENERATION_PLAN = [
    ("hsse_procedures.md",       "confined_space",      3),
    ("emergency_response.md",    "medical_emergency",   3),
    ("environmental_compliance.md", "air_quality",      3),
    ("contractor_management.md", "contractor_kpis",     3),
    ("operational_standards.md", "confined_space_entry",3),
]

generated_qas = []
gen_id_counter = 1

for doc_name, category, count in GENERATION_PLAN:
    doc_content = DOCUMENTS[doc_name]
    print(f"Generating {count} Q&A for {doc_name} ({category})...")
    prompt = build_generation_prompt(doc_name, doc_content, count, category)

    try:
        raw = call_llm(prompt)
        raw = re.sub(r"^```(?:json)?\s*", "", raw.strip())
        raw = re.sub(r"\s*```$", "", raw.strip())
        items = json.loads(raw)
        if not isinstance(items, list):
            items = [items]

        for item in items[:count]:
            item["qa_id"] = f"EVAL-{gen_id_counter:03d}"
            item.setdefault("category", category)
            item.setdefault("source_doc", doc_name)
            generated_qas.append(item)
            gen_id_counter += 1

        print(f"  Generated {len(items[:count])} Q&A pairs.")

    except Exception as e:
        print(f"  WARN: Generation failed for {doc_name}: {e}")
        # Fallback: adapt a seed QA
        for i in range(count):
            seed = SAMPLE_QA[i % len(SAMPLE_QA)]
            fallback = {
                "qa_id": f"EVAL-{gen_id_counter:03d}",
                "question": seed["question"] + f" (variant {gen_id_counter})",
                "expected_answer": seed["expected_answer"],
                "source_doc": doc_name,
                "category": category,
            }
            generated_qas.append(fallback)
            gen_id_counter += 1

print(f"\nGenerated {len(generated_qas)} LLM Q&A pairs.")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5b. Validate and combine into eval_dataset

# COMMAND ----------

REQUIRED_FIELDS = {"qa_id", "question", "expected_answer", "source_doc", "category"}

def validate_qa(qa: dict) -> bool:
    if not REQUIRED_FIELDS.issubset(set(qa.keys())):
        return False
    if not isinstance(qa.get("question"), str) or len(qa["question"]) < 10:
        return False
    if not isinstance(qa.get("expected_answer"), str) or len(qa["expected_answer"]) < 10:
        return False
    return True

valid_generated = [qa for qa in generated_qas if validate_qa(qa)]
print(f"Valid generated Q&A: {len(valid_generated)} / {len(generated_qas)}")

# Combine: 15 hand-crafted + up to 15 generated = 30
eval_dataset_raw = SAMPLE_QA + valid_generated[:15]
print(f"Eval dataset: {len(eval_dataset_raw)} total examples")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5c. Write eval dataset to volume and table

# COMMAND ----------

with open(EVAL_DATASET_PATH, "w") as f:
    json.dump(eval_dataset_raw, f, indent=2)
print(f"Wrote {EVAL_DATASET_PATH}")

eval_df = spark.createDataFrame(eval_dataset_raw)
eval_df.write.mode("overwrite").saveAsTable(EVAL_TABLE_BT_FQN)
print(f"Wrote table {EVAL_TABLE_FQN} ({eval_df.count()} rows)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Summary

# COMMAND ----------

print("=" * 60)
print("  SETUP COMPLETE")
print("=" * 60)
print()
print(f"  Catalog: {CATALOG}")
print(f"  Schema:  {CATALOG}.{SCHEMA}")
print(f"  Volume:  {VOLUME_PATH}")
print()
print("  Documents written:")
for fname in DOCUMENTS:
    print(f"    - {DOCS_PATH}/{fname}")
print()
print("  Files written:")
print(f"    - {SAMPLE_QA_PATH}  ({len(SAMPLE_QA)} Q&A pairs)")
print(f"    - {EVAL_DATASET_PATH}  ({len(eval_dataset_raw)} eval examples)")
print()
print("  Tables written:")
print(f"    - {QA_TABLE_FQN} ({len(SAMPLE_QA)} rows)")
print(f"    - {EVAL_TABLE_FQN} ({eval_df.count()} rows)")
print()
print("  Next steps:")
print("    1. Run 02a_setup_and_agent to create the Knowledge Assistant")
print("       and register V1 instructions in the MLflow Prompt Registry.")
print("=" * 60)

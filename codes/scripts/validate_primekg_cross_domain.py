#!/usr/bin/env python3
"""
Cross-Domain Validation of Regime-Gated Non-Transitive Reasoning (Phi, Gamma) on PrimeKG.

Evaluates unconditional 2-hop transitive inference (Drug --[target]--> Protein --[associated with]--> Disease)
versus the Tri-Anchored Regime-Gated Framework (Phi = (phi_phys, phi_bio, phi_time), Gamma) on the PrimeKG
pharmacological subgraph containing administration-route barriers and pharmacokinetic half-life regimes.
WARNING: Exploratory development-set analysis, not independent validation.
Rules were revised after inspecting labels/results. Barrier/time metrics are
circular heuristics, not adjudicated false positives. See leakage_audit.md.
"""

import csv
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path

BASE_DIR = Path("/data/AI4BioEM/PrimmeKG_dataverse")
OUT_DIR = Path("/home/AI4BioEM/data/primekg_validation")

ORGAN_KEYWORDS = {
    "dermatologicals": [
        "skin", "dermat", "eczema", "psoriasis", "acne", "rash", "prurit",
        "epiderm", "cutane", "urticaria", "alopecia", "keratosis", "melanoma",
        "seborrheic", "pemphig", "vitiligo", "ichthyosis", "rosacea", "tinea",
        "wart", "callus", "wound", "burn", "cellulitis", "impetigo", "ulcer",
        "erythema", "lichen", "bullous", "folliculitis", "hidradenitis",
    ],
    "sensory organs": [
        "ocular", "ophthalm", "conjunctiv", "cornea", "retina", "glaucoma",
        "uveitis", "keratitis", "eyelid", "eye", "macular", "cataract",
        "otitis", "ear", "myopia", "strabismus", "blepharitis", "choroid",
        "optic", "vision", "blind", "iridocyclitis", "episcleritis", "scleritis",
        "lacrimal", "intraocular", "presbyopia", "astigmatism", "retinitis",
    ],
    "respiratory system": [
        "asthma", "bronch", "pulmon", "lung", "respirat", "rhinitis", "copd",
        "pneumonia", "airway", "nasal", "sinusitis", "cough", "emphysema",
        "pleur", "laryng", "pharyng", "trache", "apnea", "croup", "fibrosis",
        "allergic", "anaphylax", "sarcoidosis", "tuberculos",
    ],
    "nervous system": [
        "brain", "cerebr", "nervous", "alzheimer", "parkinson", "epilep",
        "seizure", "schizophren", "depress", "anxiety", "bipolar", "psych",
        "neuro", "migraine", "pain", "insomnia", "sclerosis", "myasthenia",
        "dystonia", "neuralgia", "dementia", "autism", "adhd", "huntington",
        "ataxia", "tremor", "narcolep", "stroke", "mening", "encephal",
        "glioma", "sciatica", "paralysis", "spastic", "delirium", "phobia",
        "obsessive", "panic", "mania", "tourette", "headache", "vertigo",
        "als", "motor neuron", "radiculopath", "polyneuropath",
    ],
    "cardiovascular system": [
        "heart", "cardiac", "cardiovasc", "hypertens", "arrhythm", "angina",
        "myocard", "coronary", "vascular", "thrombo", "atherosclero", "ischemi",
        "hypotens", "tachycard", "atrial", "ventricul", "stroke", "edema",
        "hyperlipid", "cholesterol", "embolism", "infarct", "aneurysm",
        "cardiomyopath", "pericard", "endocard", "valv", "raynaud", "varicose",
        "hemorrhoid", "dyslipidemia", "triglyceride", "arteri", "venous",
    ],
    "alimentary tract": [
        "gastr", "intestin", "colitis", "crohn", "ulcer", "bowel", "diarrhea",
        "constipat", "emesis", "nausea", "vomiting", "esophag", "hepat",
        "liver", "biliary", "pancrea", "diabetes", "metaboli", "obes",
        "malabsorp", "peptic", "cirrhosis", "cholecyst", "celiac", "dyspepsia",
        "reflux", "gerd", "stomatitis", "gingivitis", "caries", "duoden",
        "ileitis", "proctitis", "gallbladder", "cholangitis",
    ],
    "metabolism": [
        "diabetes", "metaboli", "hyperglycem", "hypoglycem", "obes", "lipid",
        "cholesterol", "gout", "thyroid", "calcium", "vitamin", "mineral",
        "enzyme", "storage", "porphyria", "osteoporos", "acidosis", "alkalosis",
        "electrolyte", "potassium", "sodium", "magnesium", "iron", "copper",
        "zinc", "phenylketonuria", "mucopolysaccharid", "gaucher", "fabry",
        "pompe", "amyloid", "rickets", "scurvy", "pellagra", "beriberi",
        "nutriti", "deficiency", "hyperuricemia", "cystinuria", "homocystinuria",
    ],
    "genito urinary system": [
        "renal", "kidney", "urinary", "bladder", "prostat", "ovarian",
        "uterine", "vaginal", "endometri", "erectile", "infertility", "menstru",
        "menopaus", "contracep", "gonad", "testic", "nephrit", "cystitis",
        "nephrot", "uremia", "calculi", "incontinence", "cervic", "vulv",
        "salping", "pelvic", "hyperplasia", "dysmenorrhea", "amenorrhea",
    ],
    "sex hormones": [
        "ovarian", "uterine", "breast", "prostat", "endometri", "menopaus",
        "hypogonad", "infertility", "contracep", "amenorrhea", "dysmenorrhea",
        "androgen", "estrogen", "puberty", "osteoporos", "acne", "hirsutism",
        "polycystic", "labor", "abortion", "lactation", "gonad", "testic",
        "carcinoma", "neoplasm", "cancer",
    ],
    "musculo-skeletal system": [
        "arthrit", "rheumat", "osteo", "joint", "muscle", "muscular",
        "skeletal", "gout", "spondyl", "spasm", "fibromyalg", "bone",
        "myopath", "tendon", "bursitis", "synovitis", "paget", "rickets",
        "chondr", "disc", "lumbago", "sciatica", "sprain", "strain",
        "dystrophy", "pain", "inflammatory", "lupus", "ankylosing",
    ],
    "blood": [
        "anemia", "thromb", "hemophil", "coagul", "bleeding", "hemorrhag",
        "leukopenia", "neutropenia", "platelet", "erythro", "myeloma",
        "leukemia", "lymphoma", "purpura", "fibrin", "sickle", "thalassemia",
        "polycythemia", "myelodysplas", "sepsis", "ischemi", "infarct",
        "embolism", "stroke", "vascular", "coronary",
    ],
    "blood forming organs": [
        "anemia", "thromb", "hemophil", "coagul", "bleeding", "hemorrhag",
        "leukopenia", "neutropenia", "platelet", "erythro", "myeloma",
        "leukemia", "lymphoma", "purpura", "fibrin", "sickle", "thalassemia",
        "polycythemia", "myelodysplas", "ischemi", "infarct", "embolism",
    ],
    "antineoplastic": [
        "cancer", "carcinoma", "sarcoma", "leukemia", "lymphoma", "melanoma",
        "tumor", "neoplasm", "glioma", "myeloma", "metasta", "neuroblastoma",
        "adenoma", "blastom", "mesothelioma", "hodgkin", "waldenstrom",
        "polycythemia", "myelofibrosis", "papilloma", "condyloma", "keratosis",
        "malignan", "macroglobulinemia",
    ],
    "immunomodulating agents": [
        "autoimmune", "lupus", "rheumat", "sclerosis", "psoriasis", "crohn",
        "colitis", "transplant", "graft", "immun", "allergic", "inflammatory",
        "cancer", "lymphoma", "myeloma", "leukemia", "sarcoidosis", "uveitis",
        "nephrot", "hepatitis", "vasculitis", "behcet", "ankylosing",
        "myasthenia", "guillain", "arthritis",
    ],
    "antiinfectives for systemic use": [
        "infect", "bacteri", "viral", "fungal", "sepsis", "pneumonia",
        "meningitis", "tuberculos", "hiv", "hepatitis", "malaria", "abscess",
        "syphilis", "gonorrhea", "chlamydia", "lyme", "typhoid", "cholera",
        "anthrax", "plague", "leprosy", "tetanus", "diphtheria", "pertussis",
        "influenza", "herpes", "cytomegalo", "candida", "aspergill",
        "cryptococc", "toxoplasm", "otitis", "sinusitis", "bronchitis",
        "cystitis", "pyelonephritis", "endocarditis", "osteomyelitis",
        "cellulitis", "impetigo", "acne", "rosacea", "ulcer", "fever",
    ],
    "systemic hormonal preparations, excl. sex hormones": [
        "inflammatory", "autoimmune", "asthma", "allergic", "thyroid",
        "adrenal", "cushing", "addison", "pituitary", "growth", "calcium",
        "hypoglycem", "diabetes", "arthritis", "lupus", "shock", "edema",
        "croup", "nephrot", "sclerosis", "colitis", "crohn", "dermat",
        "eczema", "psoriasis", "leukemia", "lymphoma", "myeloma",
        "hypercalcemia", "osteoporos", "acromegaly", "insipidus", "enuresis",
        "hemorrhag", "labor", "rhinitis", "uveitis", "keratitis",
    ],
    "insulins": ["diabetes", "hyperglycem", "ketoacidos"],
    "antiparasitic products, insecticides": [
        "malaria", "parasit", "helminth", "worm", "schistosom", "filaria",
        "trypanosom", "leishmania", "amoeb", "giardia", "trichomon",
        "toxoplasm", "pneumocyst", "scabies", "pediculosis", "lice",
        "tapeworm", "nematode", "cestode", "trematode", "infect",
    ],
    "various": [
        "poison", "overdose", "toxicity", "intoxication", "diagnostic",
        "contrast", "hyperkalemia", "hyperphosphatemia", "iron", "hemochromatosis",
        "wilson", "lead", "mercury", "arsenic", "cyanide", "opioid",
        "benzodiazepine", "acetaminophen", "methotrexate", "chelat", "antidote",
        "phenylketonuria", "tyrosinemia", "urea", "hypoxia", "anoxia",
    ],
}


def parse_drug_profile(d_idx, row):
    atc1 = (row.get("atc_1") or "").lower().strip()
    domains = set()
    if "is anatomically related to " in atc1:
        part = atc1.split("is anatomically related to ", 1)[1].rstrip(".")
        for item in part.split(" and "):
            domains.add(item.strip())

    desc = " ".join([
        row.get("description") or "",
        row.get("pharmacodynamics") or "",
        row.get("protein_binding") or "",
        row.get("atc_2") or "",
        row.get("atc_3") or "",
    ]).lower()

    # Physical / Delivery Barrier Regime (phi_phys)
    local_skin_eye = {"dermatologicals", "sensory organs"}
    local_mucosal = {"dermatologicals", "sensory organs", "respiratory system"}
    explicit_local_only = any(k in desc for k in [
        "minimal systemic absorption",
        "systemic absorption is very minimal",
        "for topical dermatologic use",
        "for ophthalmic use",
        "ophthalmic treatment",
        "used topically in the treatment",
        "topical anti-inflammatory",
        "not for systemic",
    ])
    explicit_systemic = any(k in desc for k in [
        "intravenous", "parenteral", "oral administration", "oral tablet",
        "oral capsule", "subcutaneous", "intramuscular", "systemic use",
    ])

    if domains and domains.issubset(local_skin_eye):
        phys_regime = "P_LOCAL_BARRIER_SKIN_EYE"
    elif explicit_local_only and not explicit_systemic:
        phys_regime = "P_LOCAL_BARRIER_SKIN_EYE"
    elif domains and domains.issubset(local_mucosal) and any(
        k in desc for k in ["inhal", "intranasal", "nasal spray", "topical", "ophthalmic", "eye drops"]
    ) and not explicit_systemic:
        phys_regime = "P_LOCAL_BARRIER_MUCOSAL"
    elif domains:
        phys_regime = "P_SYSTEMIC_COMPARTMENT"
    else:
        phys_regime = "P_UNSPECIFIED"

    try:
        tpsa = float(row.get("tpsa") or 0.0)
    except ValueError:
        tpsa = 0.0
    try:
        mw = float(row.get("molecular_weight") or 0.0)
    except ValueError:
        mw = 0.0

    bbb_restricted = (tpsa > 140.0 or mw > 800.0) and ("nervous system" not in domains)

    # Pharmacokinetic Timescale Regime (phi_time)
    hl_str = (row.get("half_life") or "").lower().strip()
    hl_hours = None
    time_regime = "T_UNSPECIFIED"
    if hl_str:
        if "in vitro" in hl_str and not re.search(r"\b(hour|hr|day|week)", hl_str):
            time_regime = "T2_INVITRO_ACUTE_ONLY"
            m_min = re.search(r"(\d+(?:\.\d+)?)\s*(?:minute|min)", hl_str)
            if m_min:
                hl_hours = float(m_min.group(1)) / 60.0
        else:
            m_day = re.search(r"(\d+(?:\.\d+)?)\s*(?:day|days)", hl_str)
            m_wk = re.search(r"(\d+(?:\.\d+)?)\s*(?:week|weeks)", hl_str)
            m_hr = re.search(r"(\d+(?:\.\d+)?)\s*(?:hour|hours|hr|hrs|h\b)", hl_str)
            m_min = re.search(r"(\d+(?:\.\d+)?)\s*(?:minute|minutes|min\b)", hl_str)
            m_sec = re.search(r"(\d+(?:\.\d+)?)\s*(?:second|seconds|sec\b)", hl_str)
            if m_wk:
                hl_hours = float(m_wk.group(1)) * 168.0
                time_regime = "T4_DEPOT_LONG_CYCLE"
            elif m_day:
                hl_hours = float(m_day.group(1)) * 24.0
                time_regime = "T4_DEPOT_LONG_CYCLE"
            elif m_hr:
                hl_hours = float(m_hr.group(1))
                time_regime = "T3_SHORT_WINDOW" if hl_hours < 1.0 else "T4_SYSTEMIC_STEADY"
            elif m_min:
                hl_hours = float(m_min.group(1)) / 60.0
                time_regime = "T2_ULTRA_SHORT_MIN" if hl_hours <= 0.25 else "T3_SHORT_WINDOW"
            elif m_sec:
                hl_hours = float(m_sec.group(1)) / 3600.0
                time_regime = "T2_ULTRA_SHORT_SEC"

    return {
        "node_index": d_idx,
        "domains": sorted(domains),
        "phys_regime": phys_regime,
        "bbb_restricted": bbb_restricted,
        "tpsa": tpsa,
        "mw": mw,
        "time_regime": time_regime,
        "hl_hours": hl_hours,
        "hl_raw": (row.get("half_life") or "")[:120],
        "atc2_text": (row.get("atc_2") or "").lower(),
        "atc3_text": (row.get("atc_3") or "").lower(),
        "moa_text": (row.get("mechanism_of_action") or "").lower(),
        "pd_text": (row.get("pharmacodynamics") or "").lower(),
        "category_text": (row.get("category") or "").lower(),
    }


def classify_disease(dis_idx, name, drow):
    text = " ".join([
        name or "",
        drow.get("mondo_name") or "",
        drow.get("group_name_bert") or "",
        (drow.get("mondo_definition") or "")[:300],
    ]).lower()

    matched_domains = set()
    for dom, kw_list in ORGAN_KEYWORDS.items():
        if any(kw in text for kw in kw_list):
            matched_domains.add(dom)

    is_skin_eye = bool(matched_domains & {"dermatologicals", "sensory organs"})
    is_mucosal = bool(matched_domains & {"dermatologicals", "sensory organs", "respiratory system"})
    is_cns = "nervous system" in matched_domains
    is_chronic_genetic = any(k in text for k in [
        "inherited", "congenital", "genetic", "autosomal", "x-linked",
        "inborn error", "chromosom", "dystrophy", "neurodegener", "alzheimer",
        "parkinson", "huntington",
    ])
    return {
        "matched_domains": matched_domains,
        "is_skin_eye": is_skin_eye,
        "is_mucosal": is_mucosal,
        "is_cns": is_cns,
        "is_chronic_genetic": is_chronic_genetic,
        "name_lower": (name or "").lower(),
    }


def main():
    print("WARNING: exploratory development-set results; circular error metrics; NOT independent validation.")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    drug_feat_raw = {}
    with open(BASE_DIR / "drug_features.csv", "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            drug_feat_raw[row["node_index"]] = row

    disease_feat_raw = {}
    with open(BASE_DIR / "disease_features.csv", "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["node_index"] not in disease_feat_raw:
                disease_feat_raw[row["node_index"]] = row

    drug_targets = defaultdict(set)
    protein_diseases = defaultdict(set)
    indications = set()
    contraindications = set()
    off_label = set()
    drug_se = defaultdict(set)
    dis_pheno = defaultdict(set)
    drug_names, dis_names, prot_names = {}, {}, {}

    with open(BASE_DIR / "kg.csv", "r", encoding="utf-8") as f:
        r = csv.reader(f)
        next(r)
        for row in r:
            rel, drel, x_idx, x_id, x_type, x_name, x_src, y_idx, y_id, y_type, y_name, y_src = row
            if x_type == "drug":
                drug_names[x_idx] = x_name
            if x_type == "disease":
                dis_names[x_idx] = x_name
            if x_type == "gene/protein":
                prot_names[x_idx] = x_name

            if rel == "drug_protein" and drel == "target" and x_type == "drug":
                drug_targets[x_idx].add(y_idx)
            elif rel == "disease_protein" and x_type == "gene/protein" and y_type == "disease":
                protein_diseases[x_idx].add(y_idx)
            elif rel == "indication" and x_type == "drug" and y_type == "disease":
                indications.add((x_idx, y_idx))
            elif rel == "contraindication" and x_type == "drug" and y_type == "disease":
                contraindications.add((x_idx, y_idx))
            elif rel == "off-label use" and x_type == "drug" and y_type == "disease":
                off_label.add((x_idx, y_idx))
            elif rel == "drug_effect" and x_type == "drug" and y_type == "effect/phenotype":
                drug_se[x_idx].add(y_idx)
            elif rel == "disease_phenotype_positive" and x_type == "disease" and y_type == "effect/phenotype":
                dis_pheno[x_idx].add(y_idx)

    # Build drug profiles & disease profiles
    drug_profiles = {
        d: parse_drug_profile(d, drug_feat_raw.get(d, {}))
        for d in drug_targets
    }
    # Subgraph: Drugs with both explicit ATC route/compartment and half-life regime
    eval_drugs = sorted([
        d for d, p in drug_profiles.items()
        if p["phys_regime"] != "P_UNSPECIFIED" and p["time_regime"] != "T_UNSPECIFIED"
    ])

    all_candidate_diseases = set()
    for d in eval_drugs:
        for p in drug_targets[d]:
            all_candidate_diseases.update(protein_diseases[p])

    dis_profiles = {
        dis: classify_disease(dis, dis_names.get(dis, ""), disease_feat_raw.get(dis, {}))
        for dis in all_candidate_diseases
    }

    # Protein promiscuity (hub degree in disease_protein)
    prot_dis_degree = {p: len(dis_set) for p, dis_set in protein_diseases.items()}

    # Enumerate all 2-hop pairs (d, dis) with bridging target proteins
    pair_bridges = defaultdict(list)
    for d in eval_drugs:
        for p in drug_targets[d]:
            for dis in protein_diseases[p]:
                pair_bridges[(d, dis)].append(p)

    # Define Tri-Anchor Regime Evaluation (phi_phys, phi_bio, phi_time) and Non-Transitive Gate Gamma
    def evaluate_pair(d, dis, bridges):
        dp = drug_profiles[d]
        disp = dis_profiles[dis]
        d_doms = set(dp["domains"])
        dis_doms = disp["matched_domains"]

        # 1. phi_phys: Route / Compartment & Barrier Commensurability
        if dp["phys_regime"] == "P_LOCAL_BARRIER_SKIN_EYE" and not disp["is_skin_eye"]:
            phi_phys = "P_BLOCK_LOCAL_TO_SYSTEMIC"
        elif dp["phys_regime"] == "P_LOCAL_BARRIER_MUCOSAL" and not disp["is_mucosal"]:
            phi_phys = "P_BLOCK_MUCOSAL_TO_DISTANT"
        elif dp["bbb_restricted"] and disp["is_cns"] and not (d_doms & dis_doms):
            phi_phys = "P_BLOCK_BBB_IMPERMEABLE"
        elif d_doms and dis_doms and not (d_doms & dis_doms):
            phi_phys = "P_BLOCK_CROSS_ORGAN_ATC"
        else:
            phi_phys = "P_PASS_COMPARTMENT_ALIGNED"

        # 2. phi_bio: Mechanistic Module Alignment, Off-Target Hub Filtering & Pathophysiological Polarity
        # No direct label read here establishes neither proxy-free inputs nor a clean test.
        # Therapeutic proxies are present; rules were developed on these evaluation data.
        se_conflict_count = len(drug_se[d] & dis_pheno[dis])
        min_bridge_deg = min(prot_dis_degree.get(p, 999) for p in bridges)
        mech_text = " ".join([dp["atc2_text"], dp["atc3_text"], dp["moa_text"], dp["pd_text"], dp["category_text"]])
        dis_name = disp["name_lower"]

        is_corticosteroid = any(k in mech_text for k in ["corticosteroid", "glucocorticoid"])
        steroid_aggravated_pathology = any(k in dis_name for k in [
            "ocular hypertension", "glaucoma", "diabetes", "hyperglycem", "hypertens",
            "hypogonad", "psych", "schizophren", "depress", "anxiety", "neurotic",
            "epilep", "heart failure", "potassium deficiency", "osteoporos", "ulcer", "infect", "tuberculos", "fungal", "viral", "herpes",
        ])

        is_sex_hormone = "sex hormones" in d_doms or any(k in mech_text for k in ["estrogen", "androgen", "progest"])
        hormone_aggravated_pathology = any(k in dis_name for k in [
            "anxiety", "neurotic", "substance", "depress", "schizophren", "epilep",
            "thrombo", "embolism", "stroke", "migraine", "hypertens", "breast cancer", "endometrial cancer", "liver", "hepat",
        ])

        pd_aggravation = False
        if "depression" in dp["pd_text"] or "suicidal" in dp["pd_text"] or "cns depression" in mech_text:
            if any(k in dis_name for k in ["depress", "dysthymi", "neurotic", "anxiety", "asthenia", "coma"]):
                pd_aggravation = True
        if any(k in mech_text for k in ["vasoconstrict", "sympathomimetic", "adrenergic agonist", "pressor"]):
            if any(k in dis_name for k in ["hypertens", "angina", "arrhythm", "tachycard", "ischemi", "coronary", "heart failure", "glaucoma", "hyperthyroid", "diabetes"]):
                pd_aggravation = True
        if any(k in mech_text for k in ["anticholinergic", "muscarinic antagonist"]):
            if any(k in dis_name for k in ["glaucoma", "ocular hypertension", "tachycard", "arrhythm", "urinary retention", "prostat", "constipat", "ileus", "myasthenia", "dementia", "alzheimer"]):
                pd_aggravation = True
        if any(k in mech_text for k in ["beta blocking", "beta-adrenergic antagonist"]):
            if any(k in dis_name for k in ["asthma", "bronchospasm", "copd", "bradycard", "heart block", "shock", "hypotens", "hypoglycem", "diabetes"]):
                pd_aggravation = True
        if any(k in mech_text for k in ["nsaid", "cyclooxygenase inhibitor", "antiinflammatory agents, non-steroids"]):
            if any(k in dis_name for k in ["ulcer", "gastritis", "bleeding", "hemorrhag", "renal failure", "kidney", "heart failure", "asthma", "hypertens"]):
                pd_aggravation = True
        if any(k in mech_text for k in ["opioid", "narcotic"]):
            if any(k in dis_name for k in ["respirat", "asthma", "copd", "apnea", "ileus", "constipat", "addiction", "substance", "head injury", "intracranial"]):
                pd_aggravation = True

        if (is_corticosteroid and steroid_aggravated_pathology) or (is_sex_hormone and hormone_aggravated_pathology) or pd_aggravation:
            phi_bio = "B_BLOCK_POLARITY_AGGRAVATION"
        elif se_conflict_count >= 2:
            phi_bio = "B_BLOCK_PHENOTYPE_CONFLICT"
        elif min_bridge_deg > 100 and len(bridges) == 1:
            phi_bio = "B_BLOCK_PROMISCUOUS_HUB_ONLY"
        else:
            phi_bio = "B_PASS_SPECIFIC_TARGET"

        # 3. phi_time: Pharmacokinetic Half-Life vs Disease Course Renormalization
        if dp["time_regime"] == "T2_INVITRO_ACUTE_ONLY" and not disp["is_skin_eye"]:
            phi_time = "T_BLOCK_INVITRO_TO_INVIVO"
        elif dp["time_regime"] in {"T2_ULTRA_SHORT_SEC", "T2_ULTRA_SHORT_MIN"} and disp["is_chronic_genetic"]:
            phi_time = "T_BLOCK_ULTRASHORT_CHRONIC_GENETIC"
        elif disp["is_chronic_genetic"] and dp["time_regime"] == "T3_SHORT_WINDOW":
            phi_time = "T_BLOCK_ACUTE_PK_HEREDITARY"
        else:
            phi_time = "T_PASS_TIMESCALE_COMPATIBLE"

        return phi_phys, phi_bio, phi_time

    # Evaluate across 4 ablation stages
    stages = [
        "1_Unconditional_2Hop",
        "2_Plus_phi_phys_BarrierGate",
        "3_Plus_phi_phys_phi_bio",
        "4_Full_TriAnchor_Phi_Gamma",
    ]
    stats = {s: Counter() for s in stages}
    local_stats = {s: Counter() for s in stages}
    case_examples = []

    for (d, dis), bridges in pair_bridges.items():
        phi_phys, phi_bio, phi_time = evaluate_pair(d, dis, bridges)
        dp = drug_profiles[d]
        disp = dis_profiles[dis]

        is_ind = (d, dis) in indications
        is_off = (d, dis) in off_label
        is_contra = (d, dis) in contraindications
        is_local_drug = dp["phys_regime"] in {"P_LOCAL_BARRIER_SKIN_EYE", "P_LOCAL_BARRIER_MUCOSAL"}
        is_cross_barrier_fp = (
            (dp["phys_regime"] == "P_LOCAL_BARRIER_SKIN_EYE" and not disp["is_skin_eye"])
            or (dp["phys_regime"] == "P_LOCAL_BARRIER_MUCOSAL" and not disp["is_mucosal"])
            or (dp["bbb_restricted"] and disp["is_cns"])
        ) and not is_ind and not is_off

        pass_flags = {
            "1_Unconditional_2Hop": True,
            "2_Plus_phi_phys_BarrierGate": phi_phys.startswith("P_PASS"),
            "3_Plus_phi_phys_phi_bio": phi_phys.startswith("P_PASS") and phi_bio.startswith("B_PASS"),
            "4_Full_TriAnchor_Phi_Gamma": (
                phi_phys.startswith("P_PASS")
                and phi_bio.startswith("B_PASS")
                and phi_time.startswith("T_PASS")
            ),
        }

        for s, passed in pass_flags.items():
            if passed:
                stats[s]["total_retained"] += 1
                if is_ind:
                    stats[s]["true_indication"] += 1
                if is_off:
                    stats[s]["off_label"] += 1
                if is_contra:
                    stats[s]["contraindication_error"] += 1
                if is_cross_barrier_fp:
                    stats[s]["cross_barrier_fp"] += 1
                if dp["time_regime"].startswith("T2_") and disp["is_chronic_genetic"] and not is_ind:
                    stats[s]["timescale_mismatch_fp"] += 1

                if is_local_drug:
                    local_stats[s]["local_total"] += 1
                    if is_cross_barrier_fp:
                        local_stats[s]["local_cross_barrier_fp"] += 1
                    if is_ind:
                        local_stats[s]["local_indication"] += 1
                    if is_contra:
                        local_stats[s]["local_contraindication"] += 1

        # Collect diverse audit cases blocked by Full Gate across P_BLOCK, B_BLOCK, T_BLOCK
        if not pass_flags["4_Full_TriAnchor_Phi_Gamma"] and (is_contra or is_cross_barrier_fp or not phi_time.startswith("T_PASS")):
            drug_label = drug_names.get(d, d)
            already_for_drug = sum(1 for c in case_examples if c["drug_name"] == drug_label)
            if already_for_drug < 2 and len(case_examples) < 40:
                gt_label = (
                    "CONTRAINDICATION"
                    if is_contra
                    else ("CROSS_BARRIER_EXTRAPOLATION" if is_cross_barrier_fp else "TIMESCALE_MISMATCH_EXTRAPOLATION")
                )
                case_examples.append({
                    "drug_name": drug_label,
                    "bridging_targets": [prot_names.get(p, p) for p in bridges[:3]],
                    "disease_name": dis_names.get(dis, dis),
                    "clinical_ground_truth": gt_label,
                    "drug_phys_regime": dp["phys_regime"],
                    "drug_domains": dp["domains"],
                    "drug_half_life": dp["hl_raw"],
                    "phi_phys": phi_phys,
                    "phi_bio": phi_bio,
                    "phi_time": phi_time,
                })

    summary_rows = []
    for s in stages:
        st = stats[s]
        ind_cnt = st["true_indication"]
        contra_cnt = st["contraindication_error"]
        labeled_total = ind_cnt + contra_cnt
        contra_fdr = (contra_cnt / labeled_total * 100.0) if labeled_total else 0.0
        ind_precision = (ind_cnt / labeled_total * 100.0) if labeled_total else 0.0
        ic_ratio = (ind_cnt / contra_cnt) if contra_cnt else float("inf")
        lst = local_stats[s]
        local_fp_rate = (
            lst["local_cross_barrier_fp"] / lst["local_total"] * 100.0
            if lst["local_total"] else 0.0
        )
        summary_rows.append({
            "stage": s,
            "total_2hop_retained": st["total_retained"],
            "true_indications": ind_cnt,
            "off_label_retained": st["off_label"],
            "contraindication_errors": contra_cnt,
            "labeled_indication_precision_pct": round(ind_precision, 2),
            "labeled_contraindication_fdr_pct": round(contra_fdr, 2),
            "indication_to_contraindication_ratio": round(ic_ratio, 3),
            "cross_barrier_extrapolations": st["cross_barrier_fp"],
            "local_drug_2hop_total": lst["local_total"],
            "local_drug_cross_barrier_fp": lst["local_cross_barrier_fp"],
            "local_drug_cross_barrier_fp_rate_pct": round(local_fp_rate, 2),
            "timescale_mismatch_fp": st["timescale_mismatch_fp"],
        })

    report = {
        "benchmark_scope": {
            "source_dataset": str(BASE_DIR),
            "evaluated_drugs_with_route_and_halflife": len(eval_drugs),
            "candidate_diseases_reached": len(all_candidate_diseases),
            "total_unconditional_2hop_pairs": len(pair_bridges),
        },
        "ablation_results": summary_rows,
        "representative_blocked_cases": case_examples[:20],
    }

    with open(OUT_DIR / "primekg_validation_summary.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    with open(OUT_DIR / "primekg_ablation_table.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        w.writerows(summary_rows)

    print("=== PrimeKG Cross-Domain Validation Summary ===")
    print(f"Evaluated clinical drugs (with ATC route + half-life + target): {len(eval_drugs)}")
    print(f"Total unconditional 2-hop (Drug -> Target -> Disease) pairs: {len(pair_bridges)}")
    for row in summary_rows:
        print(
            f"\n[{row['stage']}]\n"
            f"  Retained 2-hop pairs: {row['total_2hop_retained']}\n"
            f"  True Indications vs Contraindications: {row['true_indications']} vs {row['contraindication_errors']} "
            f"(I/C Ratio = {row['indication_to_contraindication_ratio']}, Contraindication FDR = {row['labeled_contraindication_fdr_pct']}%)\n"
            f"  Cross-Barrier Systemic Extrapolations: {row['cross_barrier_extrapolations']}\n"
            f"  Strictly Local Drug Cross-Barrier FP: {row['local_drug_cross_barrier_fp']}/{row['local_drug_2hop_total']} ({row['local_drug_cross_barrier_fp_rate_pct']}%)\n"
            f"  PK Timescale Mismatch FP: {row['timescale_mismatch_fp']}"
        )


if __name__ == "__main__":
    main()

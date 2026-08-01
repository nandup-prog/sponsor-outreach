"""app.py — Streamlit UI for the visa-sponsor outreach engine.

Run:  streamlit run app.py
Needs .env with COMPANIES_HOUSE_API_KEY and ANTHROPIC_API_KEY, plus register.csv.
Wraps gtm_engine.py; persists to tracker.db via db.py. Nothing is sent to LinkedIn —
you get the searchable company name, website, personas to target, and copy-ready
messages, and you send by hand.
"""

import json
import os
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

import db
import gtm_engine as eng

load_dotenv()
db.init_db()
st.set_page_config(page_title="Sponsor Outreach Tracker", page_icon="📇", layout="wide")

HAS_CH = bool(os.environ.get("COMPANIES_HOUSE_API_KEY"))
HAS_AI = bool(os.environ.get("ANTHROPIC_API_KEY"))


def run_batch(profile_name, csv_path, limit, skip, do_draft):
    cfg = eng.PROFILES[profile_name]
    cache_path = Path(f".ch_cache_{profile_name}.json")
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    ch = eng.CompaniesHouse(os.environ["COMPANIES_HOUSE_API_KEY"], cache, cfg)
    ai = eng.anthropic.Anthropic() if (do_draft and HAS_AI) else None
    try:
        names = eng.load_sponsors(csv_path, False)[skip: skip + limit]
    except Exception as exc:
        st.error(f"Couldn't read the register CSV at '{csv_path}': {exc}")
        return
    if not names:
        st.warning("No companies in that range — likely the end of the register.")
        return

    added, scanned = 0, len(names)
    bar = st.progress(0.0, "Starting…")
    for i, name in enumerate(names, 1):
        bar.progress(i / scanned, f"{i}/{scanned} — {name[:48]}")
        try:
            d = eng.process_company(ch, cfg, name, ai)
        except Exception as exc:
            st.warning(f"Skipped {name}: {exc}")
            continue
        if d and db.upsert_lead(profile_name, d):
            added += 1
    cache_path.write_text(json.dumps(ch.cache))
    db.set_next_skip(profile_name, skip + limit)
    bar.empty()
    st.success(f"Scanned {scanned} → added {added} new leads. "
               f"Next batch starts at row {skip + limit}.")


# --- Sidebar ---------------------------------------------------------------
with st.sidebar:
    st.header("Run a batch")
    profile_name = st.radio("Candidate profile", list(eng.PROFILES), format_func=str.title)
    csv_path = st.text_input("Register CSV path", "register.csv")
    limit = st.number_input("How many to scan", 5, 200, 25, step=5)
    skip = st.number_input("Start at row (skip)", 0, 1_000_000,
                           db.get_next_skip(profile_name), step=5)
    do_draft = st.checkbox("Draft messages", value=True)
    disabled = not HAS_CH or (do_draft and not HAS_AI)
    if st.button("▶ Run batch", type="primary", disabled=disabled, use_container_width=True):
        run_batch(profile_name, csv_path, int(limit), int(skip), do_draft)
    st.divider()
    st.caption(f"Companies House key: {'✅' if HAS_CH else '❌ missing'}")
    st.caption(f"Anthropic key: {'✅' if HAS_AI else '❌ (uncheck Draft to run without)'}")
    st.caption(f"Personas targeted: {eng.PROFILES[profile_name]['personas']}")


# --- Main ------------------------------------------------------------------
st.title("📇 Sponsor Outreach Tracker")
st.caption(f"Profile: **{profile_name.title()}** — switch in the sidebar.")

counts = db.status_counts(profile_name)
cols = st.columns(len(db.STATUSES) + 1)
cols[0].metric("Total", sum(counts.values()))
for col, s in zip(cols[1:], db.STATUSES):
    col.metric(s, counts.get(s, 0))

tab_track, tab_queue = st.tabs(["📋 Tracker", "✉️ Work queue"])

with tab_track:
    c1, c2 = st.columns(2)
    f_status = c1.selectbox("Status", ["All"] + db.STATUSES, key="ts")
    f_sector = c2.selectbox("Sector", ["All"] + db.sectors(profile_name), key="tc")
    rows = db.fetch_leads(profile_name, f_status, f_sector)
    if not rows:
        st.info("No leads yet. Run a batch from the sidebar.")
    else:
        df = pd.DataFrame(rows)
        view = df[["id", "linkedin_name", "website", "target_personas", "sector",
                   "town", "status", "contact_name", "contact_url",
                   "linkedin_company_search", "user_notes"]].copy()
        edited = st.data_editor(
            view, key="editor", hide_index=True, use_container_width=True,
            disabled=["id", "linkedin_name", "website", "target_personas",
                      "sector", "town", "linkedin_company_search"],
            column_config={
                "id": None,
                "linkedin_name": "Company (search on LinkedIn)",
                "website": st.column_config.LinkColumn("Website"),
                "target_personas": st.column_config.TextColumn("Who to reach", width="medium"),
                "status": st.column_config.SelectboxColumn("Status", options=db.STATUSES, required=True),
                "contact_name": st.column_config.TextColumn("Contact"),
                "contact_url": st.column_config.LinkColumn("Contact URL"),
                "linkedin_company_search": st.column_config.LinkColumn("LinkedIn search"),
                "user_notes": st.column_config.TextColumn("Your notes", width="medium"),
            })
        if st.button("💾 Save changes"):
            orig = view.set_index("id")
            for _, r in edited.iterrows():
                lid = int(r["id"])
                changed = {f: r[f] for f in ["status", "contact_name", "contact_url", "user_notes"]
                           if f in r and r[f] != orig.loc[lid, f]}
                if changed:
                    db.update_lead(lid, changed)
            st.success("Saved."); st.rerun()

with tab_queue:
    active = [r for r in db.fetch_leads(profile_name)
              if r["status"] in ("New", "Connection sent", "Accepted")]
    st.caption(f"{len(active)} to work. Open LinkedIn, find one of the personas, "
               f"copy the note, send, then set status.")
    for r in active[:40]:
        with st.container(border=True):
            top = st.columns([3, 2])
            top[0].markdown(f"**{r['linkedin_name']}**  \n{r['sector']} · {r['town'] or 'UK'}  \n"
                            f"🎯 {r['target_personas']}")
            with top[1]:
                b = st.columns(2)
                b[0].link_button("LinkedIn search", r["linkedin_company_search"], use_container_width=True)
                if r["website"]:
                    b[1].link_button("Website", r["website"], use_container_width=True)
            if r["connection_note"]:
                st.text("Connection note (swap {{first_name}}, then copy):")
                st.code(r["connection_note"], language=None)
            if r["followup_dm"]:
                st.text("Follow-up DM (after they accept):")
                st.code(r["followup_dm"], language=None)
            act = st.columns([2, 1, 4])
            new_status = act[0].selectbox("Status", db.STATUSES,
                                          index=db.STATUSES.index(r["status"]),
                                          key=f"st_{r['id']}", label_visibility="collapsed")
            if act[1].button("Update", key=f"up_{r['id']}"):
                db.update_lead(r["id"], {"status": new_status}); st.rerun()
            act[2].text_input("Contact", value=r["contact_name"], key=f"cn_{r['id']}",
                              placeholder="who you messaged", label_visibility="collapsed",
                              on_change=lambda i=r["id"]: db.update_lead(
                                  i, {"contact_name": st.session_state[f"cn_{i}"]}))

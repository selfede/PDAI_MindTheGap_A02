import streamlit as st
import pandas as pd
import PyPDF2
import requests
import json
import re
import numpy as np
import plotly.express as px
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

#all my keys/tokens
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
LINKEDIN_TOKEN = st.secrets["LINKEDIN_TOKEN"]
HUNTER_API_KEY = st.secrets["HUNTER_API_KEY"]
GMAIL_CLIENT_ID     = st.secrets["GMAIL_CLIENT_ID"]
GMAIL_CLIENT_SECRET = st.secrets["GMAIL_CLIENT_SECRET"]
GMAIL_REFRESH_TOKEN = st.secrets["GMAIL_REFRESH_TOKEN"]
GMAIL_SENDER_EMAIL  = st.secrets.get("GMAIL_SENDER_EMAIL", "mindthegap.demo@gmail.com")
GMAIL_TOKEN_URL     = "https://oauth2.googleapis.com/token"

st.set_page_config(page_title="Mind the Gap", layout="wide")

st.title("Your personalized career development AI coach")
st.markdown("Uncover the gap, bridge it with knowledge, and accelerate your entry into the professional market.")

#saving info w session state to keep user data across page interactions
if "cv_text" not in st.session_state: st.session_state.cv_text = ""
if "transcript_text" not in st.session_state: st.session_state.transcript_text = ""
if "profile_summary" not in st.session_state: st.session_state.profile_summary = ""
if "ai_response" not in st.session_state: st.session_state.ai_response = None
if "jd_texts" not in st.session_state: st.session_state.jd_texts = {}
if "jd_skills" not in st.session_state: st.session_state.jd_skills = {}
if "app_df" not in st.session_state: st.session_state.app_df = pd.DataFrame(columns=["Company", "Industry", "Role", "Status"])
if "coffee_hooks" not in st.session_state: st.session_state.coffee_hooks = []
if "coffee_email" not in st.session_state: st.session_state.coffee_email = ""
if "active_hooks" not in st.session_state: st.session_state.active_hooks = []
if "vector_results" not in st.session_state: st.session_state.vector_results = None
if "li_connections"    not in st.session_state: st.session_state.li_connections    = []
if "filtered_contacts" not in st.session_state: st.session_state.filtered_contacts = []
if "selected_contacts" not in st.session_state: st.session_state.selected_contacts = []
if "contact_selection" not in st.session_state: st.session_state.contact_selection = []
if "enriched_contacts" not in st.session_state: st.session_state.enriched_contacts = []
if "draft_emails"      not in st.session_state: st.session_state.draft_emails      = {}
if "send_log"          not in st.session_state: st.session_state.send_log          = []
if "gmail_access_token" not in st.session_state: st.session_state.gmail_access_token = None

#########################################
#defining all helper functions below

#extracting raw text from pdf files
def extract_text(files):
    if not files:
        return ""
    text = ""
    file_list = files if isinstance(files, list) else [files]
    for file in file_list:
        try:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + " "
        except Exception:
            st.error(f"Error reading {file.name}")
    return text

#parsing LLM responses to safely handle formatting issues or extra text
def safe_json_parse(raw_text):
    try:
        cleaned = raw_text.strip()                        #removes spaces
        if cleaned.startswith("```"):                     #checks code block
            cleaned = "\n".join(cleaned.split("\n")[1:])  #drops the 1st line
        if cleaned.endswith("```"):
            cleaned = "\n".join(cleaned.split("\n")[:-1]) #drops the last line
        cleaned = cleaned.strip()
        start = cleaned.find("{")                         #finds where the { opens
        end = cleaned.rfind("}") + 1                      #finds where the last } closes
        return json.loads(cleaned[start:end])             #parses only that part as JSON
    except Exception:
        return None

#since used a lot, turned into helper func: calls llm + error handling  // n of tokens has been selected based on iterations and adjusted to ensure best response
def call_groq(messages, max_tokens=2500, temperature=0.3, model="openai/gpt-oss-safeguard-20b"):
    resp = requests.post(
        GROQ_URL,
        headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
        json={
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        },
        timeout=30,
    )
    if not resp.ok:
        raise Exception(f"API Error {resp.status_code}: {resp.text}")
    return resp.json()['choices'][0]['message']['content']

#caches loaded model to avoid reloading on every interaction -> improves performance
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")  #model for analyzing semantic similarity

#profile summary: vectorization-optimised paragraph for cv + transcript
def generate_profile_summary(cv_text, transcript_text):
    prompt = (
        #role
        "You are a career coach writing a semantic profile summary for vector similarity matching.\n\n"
        #instructions
        "INSTRUCTIONS:\n"
        "Write a single plain-text paragraph of 10 to 15 sentences. "
        "No bullet points, no headers, no bold, no markdown, pure prose only.\n"
        "Each sentence must state one skill or capability, immediately followed in parentheses by the concrete evidence from the CV or transcript: company names, course names, grades, or specific tasks.\n"
        "Pattern for every sentence: [Skill or capability] ([proof from CV or transcript]).\n\n"
        #example to guide model to right format
        "EXAMPLE OUTPUT STYLE (follow this exactly):\n"
        "Financial modelling and valuation (built DCF and LBO models during internship at Brattle Group, supported by Corporate Finance grade A at university). "
        "Data analysis using Excel and Python (automated reporting dashboards at KPMG internship, completed Python for Data Science course). "
        "Stakeholder communication and presentation (presented findings to senior partners at Mondelez, delivered group project presentations to academic panels at LSE). "
        "Cross-cultural collaboration (international education across three countries, worked in multinational teams at KPMG London and Brattle Group Brussels). "
        "Strategic thinking and structured problem solving (co-authored market entry report for FMCG client at Brattle Group, strategy dissertation graded distinction at university). "
        "Project management under deadlines (coordinated three-person deliverable team at KPMG, managed dissertation research independently over 8 months).\n\n"
        #rules for output
        "RULES:\n"
        "- Cover academic skills, technical tools, soft skills, domain knowledge, and practical experience. Use common sense to infer relevant skills from the provided information.\n"
        "- Use actual names from the CV and transcript. Do NOT invent anything.\n"
        "- Output only the paragraph. No introduction, no conclusion, no formatting.\n\n"
        "CV:\n" + cv_text[:3000] + "\n\n"
        "UNIVERSITY TRANSCRIPT:\n" + transcript_text[:5000]
    )
    return call_groq([{"role": "user", "content": prompt}], max_tokens=2000, temperature=0.2)


def extract_skill_keyword(sentence):
    """Extract 1-2 core words from a full skill sentence for display and Coursera search."""
    s = sentence.strip()
    #removes common openers
    for pattern in [
        r'^(Strong|Excellent|Proven|Clear|Effective|Advanced|Solid|Deep)\s+',
        r'^(Experience|Knowledge|Understanding|Proficiency|Expertise)\s+(in|of|with)\s+',
        r'^(Ability|Skills?)\s+to\s+',
        r'^[A-Z][a-z]+ing\s+',   #removes -ing verbs which were problematic
    ]:
        s = re.sub(pattern, '', s, flags=re.IGNORECASE)
    stop = {'and', 'or', 'to', 'for', 'in', 'with', 'of', 'a', 'the', 'skills', 'skill'} #drop fillers
    words = [w.rstrip('.,;:') for w in s.split()[:4] if w.lower().rstrip('.,;:') not in stop][:2]
    return ' '.join(words).title() or sentence.split()[0].title()


#JD skill extraction: top 6 priority skills sentences for semantic analysis
def extract_jd_skills(jd_text):
    def _parse_skills_from_raw(raw):  #first tries to parse JSON if fails it does line-by-line (handling wrong format)
        parsed = safe_json_parse(raw)
        if parsed and "skills" in parsed and len(parsed["skills"]) >= 3:
            return [s for s in parsed["skills"] if len(s.strip()) > 8][:6]
        skills = []
        for line in raw.split("\n"):
            line = re.sub(r'^[\d\-\*\•\.\)\"\']+\s*', '', line).strip().rstrip('"').rstrip("'")
            if 10 < len(line) < 250 and not line.startswith("{") and not line.startswith("}"):
                skills.append(line)
        return skills[:6] if len(skills) >= 3 else []

    prompt = (
        #role
        "You are a recruitment expert. Read the job description below and identify the 6 most important skills required to succeed in this role, to use for vector similarity matching.\n\n"
        #rules
        "RULES:\n"
        "- Only include real skills: technical abilities, domain knowledge, or soft skills.\n"
        "- Exclude visa requirements, availability, location, years of experience, degree requirements, compensation.\n"
        "- Write each skill as one plain-text sentence naming the skill and describing what competent performance looks like.\n"
        "- Each sentence must be self-contained and descriptive enough to be matched semantically, avoid one-word answers.\n\n"
        #example to guide model to right format
        "EXAMPLE OUTPUT:\n"
        "Financial modelling skills including building DCF and comparable company analysis to support investment decisions.\n"
        "Python or SQL programming to query large datasets, automate workflows, and surface actionable insights.\n"
        "Clear written and verbal communication to present data-driven findings to non-technical stakeholders.\n\n"
        "Return ONLY this JSON, no preamble, no explanation:\n"
        "{\"skills\": [\"sentence one\", \"sentence two\", \"sentence three\", \"sentence four\", \"sentence five\", \"sentence six\"]}\n\n"
        "JOB DESCRIPTION:\n" + jd_text[:8000]
    )
    try:
        raw = call_groq([{"role": "user", "content": prompt}], max_tokens=1000, temperature=0.2)
        skills = _parse_skills_from_raw(raw)
        if len(skills) >= 3:
            return skills
    except Exception:
        pass

    return []

#########################################
#vector analysis: takes sentences and embeds with sentence transformer, cosine similarity to jd skills -> finds strengths + gaps
#this func chunks profile into sentences of max 40 words to ensure embedding performance
def chunk_text_by_sentences(text, max_words=40):
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    chunks, current, current_len = [], [], 0
    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue
        words = sent.split()
        if current_len + len(words) > max_words and current:
            chunks.append(" ".join(current))
            current, current_len = [sent], len(words)
        else:
            current.append(sent)
            current_len += len(words)
    if current:
        chunks.append(" ".join(current))
    return [c for c in chunks if len(c.split()) >= 5]  #discard tiny chunks for good semantic interpretation

#while this checks vs JD skill sentence w cosine similarity -> finding top 3 strengths + gaps w evidence
def run_vector_gap_analysis(profile_summary, jd_skills_list, model):
    profile_chunks = chunk_text_by_sentences(profile_summary)
    if not profile_chunks or not jd_skills_list:
        return None

    skill_embeddings = model.encode(jd_skills_list, show_progress_bar=False)
    profile_embeddings = model.encode(profile_chunks, show_progress_bar=False)
    #run analysis and get scores
    sim_matrix = cosine_similarity(skill_embeddings, profile_embeddings)
    best_scores = sim_matrix.max(axis=1)
    best_match_idx = sim_matrix.argmax(axis=1)

    scored = sorted(
        zip(jd_skills_list, best_scores.tolist(), best_match_idx.tolist()),
        key=lambda x: x[1], reverse=True
    )

    strengths = [(skill, score, profile_chunks[idx]) for skill, score, idx in scored[:3]]  #top 3 strengths w evidence
    gaps = [(skill, score, profile_chunks[idx]) for skill, score, idx in scored[-3:]]  #top 3 gaps w evidence
    gaps.reverse()  #start from worst gap

    return {
        "overall_score": float(np.mean(best_scores) * 100),
        "strengths": strengths,
        "gaps": gaps,
        "all_scored": scored,
    }

#getting data from linkedin api: uploading everytime so newer connections/removed ones are up to date/in line w my latest applications
def fetch_linkedin_connections(token):
    connections, start = [], 0
    headers = {
        "Authorization": "Bearer " + token,
        "Linkedin-Version": "202312",
        "Content-Type": "application/json",
    }
    while True:
        url = (
            "https://api.linkedin.com/rest/memberSnapshotData"
            "?q=criteria&domain=CONNECTIONS&start=" + str(start)
        )
        try:
            resp = requests.get(url, headers=headers, timeout=15)
        except Exception as e:
            raise Exception("Network error: " + str(e))
        if resp.status_code in (204, 404):
            break
        if not resp.ok:
            raise Exception("LinkedIn API " + str(resp.status_code) + ": " + resp.text[:300])
        data = resp.json()
        elements = data.get("elements", [])
        if not elements:
            break
        # The API always returns exactly 1 element; snapshotData is the list of records
        snapshot_data = elements[0].get("snapshotData", [])
        if not snapshot_data:
            break
        for rec in snapshot_data:
            connections.append({
                "first_name":   rec.get("First Name", ""),
                "last_name":    rec.get("Last Name", ""),
                "position":     rec.get("Position", ""),
                "company":      rec.get("Company", ""),
                "connected_on": rec.get("Connected On", ""),
            })
        if len(snapshot_data) < 10:
            break
        start += 10
    return connections

#searching pool of linkedin 1st connections from my account based on user's filters
def filter_connections(connections, company_q, role_q):
    out = []
    for c in connections:
        cm = company_q.lower() in c["company"].lower()  if company_q.strip() else True
        rm = role_q.lower()    in c["position"].lower() if role_q.strip()    else True
        if cm and rm:
            out.append(c)
    return out

#searching for connection's email thru an external api
def lookup_email_hunter(first_name, last_name, domain, hunter_key, min_score=50):
    try:
        resp = requests.get(
            "https://api.hunter.io/v2/email-finder",
            params={"first_name": first_name, "last_name": last_name,
                    "domain": domain, "api_key": hunter_key},
            timeout=10,
        )
        if resp.ok:
            d = resp.json().get("data", {})
            email, score = d.get("email"), d.get("score", 0)
            if email and score >= min_score:
                return {"email": email, "score": score}
    except Exception:
        pass
    return None

#had to add cleaning of company for better matching on hunter api
def company_to_domain(company_name):
    name = company_name.lower().strip()
    for suffix in [" ltd", " limited", " inc", " llc", " plc", " gmbh", " s.a.", " sa"]:
        name = name.replace(suffix, "")
    name = re.sub(r"[^a-z0-9]", "", name)
    return name + ".com"

#llm fails to output msgs when personal info is involved -> template adapts to contact info, then user can personalise manually
def generate_personalised_email(contact, role_label, sender_name):
    return (
        "Hello " + contact["first_name"] + ",\n\n"
        "I hope all is well with you!\n\n"
        "My name is " + (sender_name or "[your name]") + " and I'm currently applying for the "
        + role_label + " opportunity. I find your experience as " + contact["position"]
        + " at " + contact["company"] + " very interesting and would love to hear more about your experience and background."
        + " Would you be open to a 20-minute virtual coffee chat to connect and share some advice with me?\n\n"
        "Thank you so much in advance and have a lovely day ahead!\n\n"
        + (sender_name or "[your name]")
    )

#needed to ensure token is refreshed before sending email to avoid failed send tries
def refresh_access_token(refresh_token):
    resp = requests.post(
        GMAIL_TOKEN_URL,
        data={
            "refresh_token": refresh_token,
            "client_id":     GMAIL_CLIENT_ID,
            "client_secret": GMAIL_CLIENT_SECRET,
            "grant_type":    "refresh_token",
        },
        timeout=15,
    )
    if not resp.ok:
        raise Exception("Token refresh failed: " + resp.text[:300])
    return resp.json().get("access_token")


#to automate coffee chat request via email
def send_via_gmail_api(sender_email, recipient_email, subject, body, access_token):
    import base64
    from email.mime.text import MIMEText
    msg = MIMEText(body, "plain")
    msg["to"]      = recipient_email
    msg["from"]    = sender_email
    msg["subject"] = subject
    raw = base64.urlsafe_b64encode(msg.as_bytes()).decode()
    resp = requests.post(
        "https://gmail.googleapis.com/gmail/v1/users/me/messages/send",
        headers={"Authorization": "Bearer " + access_token,
                 "Content-Type": "application/json"},
        json={"raw": raw},
        timeout=15,
    )
    if not resp.ok:
        raise Exception("Gmail send error: " + resp.text[:200])
    return True

#initializes gmail access token on app load to avoid delays when user wants to send coffee chat requests
def _init_gmail():
    if st.session_state.gmail_access_token:
        return
    try:
        token = refresh_access_token(GMAIL_REFRESH_TOKEN)
        if token:
            st.session_state.gmail_access_token = token
    except Exception:
        pass

_init_gmail()

######################################### starting w format
#side bar
with st.sidebar:
    st.title("Your academic profile")
    cv_file = st.file_uploader("Upload your CV (PDF)", type="pdf", key="sidebar_cv")
    transcript_file = st.file_uploader("Upload your uni transcript (PDF)", type="pdf", key="sidebar_transcript")
    st.divider()

    if st.button("Upload files", type="primary"):
        if cv_file and transcript_file:
            with st.spinner("Uploading and summarising your profile with AI..."):
                cv_file.seek(0)
                transcript_file.seek(0)
                cv_text = extract_text([cv_file])
                transcript_text = extract_text([transcript_file])
                if cv_text and transcript_text:
                    st.session_state.cv_text = cv_text
                    st.session_state.transcript_text = transcript_text
                    try:
                        summary = generate_profile_summary(cv_text, transcript_text)  #profile summarized here vs A01 version (which was in gap analysis = had to load)
                        if summary and len(summary.strip()) > 50:
                            st.session_state.profile_summary = summary
                            st.success("Academic profile uploaded and summarised!")
                        else:  #handling cases where model fails to give an acceptable summary
                            st.warning("AI returned an empty summary: using raw text as fallback!")
                            st.session_state.profile_summary = cv_text + " " + transcript_text
                    except Exception as e:
                        st.warning(f"AI summary failed ({e}): using raw text as fallback!")
                        st.session_state.profile_summary = cv_text + " " + transcript_text
                else:
                    st.warning("Could not read one or more files. Please check and try again.")
        else:
            st.error("Please select both CV and uni transcript.")

    st.divider()
    st.caption("Academic profile status")
    st.write("CV:", "Uploaded" if st.session_state.cv_text else "Not uploaded")
    st.write("Uni transcript:", "Uploaded" if st.session_state.transcript_text else "Not uploaded")
    st.write("AI Summary:", "Ready" if st.session_state.profile_summary else "Not generated")

st.header("Career development console")

tab1, tab2, tab3, tab4 = st.tabs(["Applications tracking", "Analytics", "Learning path", "Time for a coffee chat?"])

######################################### tab1: tracker
with tab1:
    st.subheader("Add new applications")
    new_jds = st.file_uploader(
        "Drop job description (PDFs) here. Please wait a few seconds to ensure a successful upload.",
        type="pdf", accept_multiple_files=True, key="bulk_uploader"
    )

    if st.button("Retrieve application information") and new_jds:
        with st.spinner("AI is reading the job descriptions. Please hold..."):
            import time
            new_rows, new_jd_texts, new_jd_skills = [], [], []
            for jd_file in new_jds:
                jd_text = extract_text(jd_file)
                parse_prompt = (
                    "Identify company, industry (1 word only, consistent terminology, check for consistency), and role. "
                    "Return ONLY JSON: {\"company\": \"...\", \"industry\": \"...\", \"role\": \"...\"} for: "
                    + jd_text[:1000]
                )
                try:
                    content = call_groq([{"role": "user", "content": parse_prompt}], max_tokens=300, temperature=0.1)
                    data = safe_json_parse(content)
                    if data:
                        new_rows.append({
                            "Company": data.get("company", "Unknown"),
                            "Industry": data.get("industry", "Unknown"),
                            "Role": data.get("role", "Unknown"),
                            "Status": "Not Applied"
                        })
                        new_jd_texts.append(jd_text)
                        try:
                            time.sleep(3)  #avoid hitting groq 8000 TPM rate limit when processing multiple JDs
                            skills_list = extract_jd_skills(jd_text)  #vs A01 version i already extract skills > raw text here, so ready for analysis
                        except Exception:
                            skills_list = []
                        new_jd_skills.append(skills_list)
                except Exception as e:
                    st.warning(f"Could not parse {jd_file.name}: {e}")
                time.sleep(2)  #pause between JDs to stay within rate limit

            if new_rows:
                start_idx = len(st.session_state.app_df)
                st.session_state.app_df = pd.concat(
                    [st.session_state.app_df, pd.DataFrame(new_rows)], ignore_index=True
                )
                for i, (jd_text, skills_list) in enumerate(zip(new_jd_texts, new_jd_skills)):
                    st.session_state.jd_texts[start_idx + i] = jd_text
                    st.session_state.jd_skills[start_idx + i] = skills_list
                st.success(f"Added {len(new_rows)} applications.")

    st.divider()
    st.subheader("Your application tracker")
    edited = st.data_editor(
        st.session_state.app_df,
        num_rows="dynamic",
        use_container_width=True,
        key="main_app_editor",
        column_config={
            "Status": st.column_config.SelectboxColumn(
                "Application Status",
                options=["Not Applied", "Applied", "Interviewing", "Rejected"],
                required=True
            )
        }
    )
    st.session_state.app_df = edited

######################################### tab2: insights
with tab2:
    st.header("Application insights")

    if st.session_state.app_df.empty:
        st.info("Add some applications in the Applications tracking Tab to see your analytics!")
    else:
        total_apps = len(st.session_state.app_df)
        interviewing = len(st.session_state.app_df[st.session_state.app_df['Status'] == 'Interviewing'])
        m1, m2 = st.columns(2)
        m1.metric("Total Applications", total_apps)
        m2.metric("Active Interviews", interviewing)
        st.divider()

        #quick overview of which stage the user is at overall
        st.subheader("Application status")
        status_counts = st.session_state.app_df['Status'].value_counts().reset_index()
        status_counts.columns = ['Status', 'Count']
        fig_pie = px.pie(
            status_counts, values='Count', names='Status', hole=0.5,
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_pie, use_container_width=True)
        st.divider()

        #experimenting with tags to show different industry distribution of applications
        st.subheader("Industry distribution")
        all_industries = sorted(st.session_state.app_df['Industry'].unique())
        selected_industries = st.multiselect(
            "Select industries to compare:",
            options=all_industries,
            default=all_industries[:3] if len(all_industries) > 3 else all_industries
        )
        if selected_industries:
            filtered_df = st.session_state.app_df[st.session_state.app_df['Industry'].isin(selected_industries)]
            industry_data = filtered_df['Industry'].value_counts().reset_index()
            industry_data.columns = ['Industry', 'Applications']
            fig_bar = px.bar(
                industry_data, x='Industry', y='Applications', text='Applications',
                color='Applications', color_continuous_scale='Reds'
            )
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.warning("Please select at least one industry to view the chart")

######################################### tab3: gap analysis
#section w major changes as in A01 version the LLM was returning everything. now there's a pipeline as follows:
#profile + JD skills (llm) -> cosine similarity (math, no longer subjective to llm) = strengths/gaps -> overview + courses (llm)
with tab3:
    st.header("Gap analysis & course recommendations")
    st.markdown(
        "Your CV and transcript are summarised into a semantic profile by AI. "
        "The job description is distilled into 6 core skill sentences. "
        "Vector similarity analysis identifies where you are strong and where to focus next."
    )

    if st.session_state.app_df.empty:
        st.info("Add applications in the Applications tracking Tab first.")
    else:
        st.subheader("Step 1: select an application")  #for any application, we have the extracted skills
        target_idx = st.selectbox(
            "Which role do you want to analyse?",
            options=st.session_state.app_df.index,
            format_func=lambda x: (
                st.session_state.app_df.loc[x, 'Company']
                + " — "
                + st.session_state.app_df.loc[x, 'Role']
            )
        )

        jd_skills_list = st.session_state.jd_skills.get(target_idx, [])  #gets list of skills for each application

        col_left, col_right = st.columns(2)  #here user can check the summaries (inputs to analysis) extracted by the llm for reference
        with col_left:
            if st.session_state.profile_summary:
                with st.expander("📄 Your AI profile summary"):
                    st.write(st.session_state.profile_summary)
            else:
                st.warning("Upload your CV and transcript in the sidebar to generate your profile summary.")
        with col_right:
            if jd_skills_list:
                with st.expander("📋 6 core skills identified for this role"):
                    for i, skill in enumerate(jd_skills_list, 1):
                        st.write(str(i) + ". " + skill)
            else:
                st.warning("No JD skills found for this application. Please re-upload the JD.")
        st.divider()

        #gap analysis starts
        if st.button("Run gap analysis", type="primary"):
            if not st.session_state.profile_summary:
                st.warning("Please upload and process your CV and uni transcript in the sidebar first.")
            elif not jd_skills_list:
                st.warning("No job description skills available for this application. Re-upload the JD.")
            else:
                with st.spinner("Computing semantic similarity across 6 role skills..."):
                    try:
                        model = load_model()
                        results = run_vector_gap_analysis(
                            st.session_state.profile_summary,
                            jd_skills_list,
                            model
                        )

                        if not results:
                            st.error("Could not extract enough text from the documents to run analysis.")
                        else:
                            role_label = (
                                st.session_state.app_df.loc[target_idx, 'Role']
                                + " at "
                                + st.session_state.app_df.loc[target_idx, 'Company']
                            )
                            #formats strengths and gaps: skill sentence + similarity score % + matching profile chunk
                            strengths_detail = "\n".join([
                                "Strength: " + s[0]
                                + " | Score: " + str(round(s[1] * 100)) + "%"
                                + " | Best profile evidence: " + s[2]
                                for s in results["strengths"]
                            ])
                            gaps_detail = "\n".join([
                                "Gap: " + g[0] + " | Score: " + str(round(g[1] * 100)) + "%"
                                for g in results["gaps"]
                            ])

                            #profile overview + relevant to role
                            #strengths + gaps identified by vectors: LLM translates in plain english
                            overview_prompt = (
                                "You are a career coach. Write a personalised profile overview for a student applying for the role of " + role_label + ".\n\n"
                                "Write ONE concise paragraph of 5 to 7 sentences that:\n"
                                "1. Opens with a brief introduction of the candidate's academic and professional background.\n"
                                "2. Highlights their 3 key strengths for this role, naming each skill and citing the concrete experience that substantiates it (e.g. their strength in marketing analytics is backed by their internship at Mondelez where they ran campaign performance tracking).\n"
                                "3. Closes with a brief note that there are development areas to address.\n\n"
                                "Tone: professional, coaching, motivating. No bullet points, no headers, plain prose only. Refer to the candidate as - the candidate - not in first person.\n\n"
                                "PROFILE SUMMARY:\n" + st.session_state.profile_summary[:2000] + "\n\n"
                                "STRENGTHS (vector-identified):\n" + strengths_detail + "\n\n"
                                "GAPS (vector-identified):\n" + gaps_detail
                            )
                            #this produces the short summary
                            overview_paragraph = call_groq(
                                [{"role": "user", "content": overview_prompt}],
                                max_tokens=600, temperature=0.4
                            )

                            #course recos
                            gap_courses = []
                            for gap_skill, gap_score, _ in results["gaps"]:
                                skill_keyword = extract_skill_keyword(gap_skill)
                                course_prompt = (
                                    "A candidate has a skill gap in: " + skill_keyword + ".\n"
                                    "Recommend ONE real, specific Coursera course that would address this gap.\n"
                                    "The course_name must be the actual title of the course as it appears on Coursera — "
                                    "not a description of the skill gap. For example: 'Marketing Analytics' or "
                                    "'Strategic Management' or 'Business Communication'.\n"
                                    "Return ONLY this JSON with no preamble:\n"
                                    "{\"course_name\": \"actual course title here\", \"provider\": \"university or organisation name\", "
                                    "\"reason\": \"one sentence why this course addresses the gap\"}"
                                )
                                try:
                                    raw = call_groq(
                                        [{"role": "user", "content": course_prompt}],
                                        max_tokens=200, temperature=0.2
                                    )
                                    course_data = safe_json_parse(raw)
                                    if not course_data or "course_name" not in course_data:
                                        raise ValueError("parse failed")
                                except Exception:  #in case the reco fails from the llm, this looks up basic level courses for the relevant skill gap
                                    course_data = {
                                        "course_name": skill_keyword + " — Foundations",
                                        "provider": "Coursera",
                                        "reason": "Covers the core concepts needed to address this skill gap."
                                    }
                                gap_courses.append(course_data)

                            results["overview_paragraph"] = overview_paragraph
                            results["gap_courses"] = gap_courses
                            st.session_state.vector_results = results

                    except Exception as e:
                        st.error(f"Analysis failed: {e}")

        #saving all results
        if st.session_state.vector_results:
            res = st.session_state.vector_results

            st.divider()

            col_score, col_spacer = st.columns([1, 3])
            with col_score:
                st.metric("Overall profile match", f"{res['overall_score']:.0f}%")
                st.caption("Mean cosine similarity across 6 role skills")

            st.divider()

            st.subheader("📋 Profile overview")
            st.write(res["overview_paragraph"])

            st.divider()

            col_str, col_gap = st.columns(2)

            with col_str:
                st.subheader("Top strengths")
                st.caption("Role skills where your profile scores highest")
                for skill, skill_score, evidence in res["strengths"]:
                    pct = int(skill_score * 100)
                    bar_col = "#2ecc71" if pct >= 50 else "#f0a500"
                    with st.expander("**" + skill + "** - " + str(pct) + "% match", expanded=True):
                        st.markdown(
                            "<div style='background:#e8e8e8;border-radius:6px;height:8px;margin-bottom:10px;'>"
                            "<div style='width:" + str(pct) + "%;background:" + bar_col
                            + ";height:8px;border-radius:6px;'></div></div>",
                            unsafe_allow_html=True
                        )
                        st.markdown("**Evidence from your profile:**")
                        st.write(evidence)

            #formatting results pretty w progress bars + expanders + course recos
            with col_gap:
                st.subheader("Top gaps to address")
                st.caption("Role skills where your profile scores lowest: prioritise these")
                for (gap_skill, gap_score, profile_match), course_data in zip(
                    res["gaps"], res.get("gap_courses", [{}] * 3)
                ):
                    gap_pct = int(gap_score * 100)
                    bar_col = "#e05252" if gap_pct < 40 else "#f0a500"
                    skill_keyword = extract_skill_keyword(gap_skill)  #getting a keyword > full sentence ensures more accurate search on coursera
                    course_name = course_data.get("course_name", skill_keyword + " Foundations")
                    course_provider = course_data.get("provider", "Coursera")
                    course_reason = course_data.get("reason", "")
                    coursera_query = "+".join(skill_keyword.split())

                    with st.expander("**" + skill_keyword + "** - " + str(gap_pct) + "% match", expanded=True):
                        st.markdown(
                            "<div style='background:#e8e8e8;border-radius:6px;height:8px;margin-bottom:10px;'>"
                            "<div style='width:" + str(gap_pct) + "%;background:" + bar_col
                            + ";height:8px;border-radius:6px;'></div></div>",
                            unsafe_allow_html=True
                        )
                        st.caption("Full skill requirement: " + gap_skill)
                        st.markdown("**Closest match in your profile:**")
                        st.write(profile_match[:400])
                        st.markdown(
                            "**Recommended course:** " + course_name
                            + " *(" + course_provider + ")*"
                        )
                        if course_reason:
                            st.caption(course_reason)
                        st.link_button(
                            "🎓 Find this course on Coursera",
                            "https://www.coursera.org/search?query=" + coursera_query,
                            use_container_width=True
                        )

            st.warning(
                "**Disclaimer:** if this analysis does not fully reflect your profile, consider it an opportunity to "
                "strengthen your CV by incorporating keywords and language that are closely aligned with the role and "
                "industry you are targeting. Semantic similarity is used by many ATS screening systems and rewards CVs "
                "that mirror the vocabulary of the job description. Tailoring your wording is one of the most effective "
                "ways to improve both this score and your visibility to recruiters!"
            )

######################################### tab4: coffee chat outreach process
#biggest feature addition to my prototype: combines multiple apis (linkedIn, hunter.io, gmail) for end to end coffee chat outreach workflow and AI personalisation
with tab4:
    st.header("Coffee chat outreach ☕")
    st.caption(
        "Filter through LinkedIn connections, find verified emails via Hunter.io, "
        "generate a personalised draft for each contact, then send in one click!"
    )

    #setting gmail status with a demo gmail account for testing purposes
    if st.session_state.gmail_access_token:
        st.success("Gmail ready - sending from **" + GMAIL_SENDER_EMAIL + "**")
    else:
        st.error("Gmail authentication failed.")

    sender_name  = st.text_input("Your name (for email sign-off)", placeholder="Name + Last name")
    email_subject = st.text_input(
        "Email subject line",
        value="Quick coffee chat request - exploring roles in your field",
    )

    st.divider()

    #here i pull linkedin connections from my profile thru the linkedin API / pulling each time to ensure they're updated w latest connections sent/revoked
    st.subheader("Step 2: pull LinkedIn connections")

    col_fetch, col_status = st.columns([2, 3])
    with col_fetch:
        if st.button("Fetch the pool of LinkedIn connections"):
            with st.spinner("Calling LinkedIn API to get connections..."):
                try:
                    conns = fetch_linkedin_connections(LINKEDIN_TOKEN)
                    st.session_state.li_connections = conns
                    if conns:
                        st.success(f"Fetched {len(conns)} connections.")
                    else:
                        st.warning("No connections returned.")
                except Exception as e:
                    st.error(f"LinkedIn fetch failed: {e}")
    with col_status:
        if st.session_state.li_connections:
            st.info(f"{len(st.session_state.li_connections)} connections loaded in session.")

    st.divider()

    #filter by company + role
    st.subheader("Step 3: filter by target company and/or role")
    st.caption(
        "For higher chances of getting a confident email match, aim to select 10-20 connections who are in a similar role OR company "
        "as your target application. The more the filters, the less the contacts to search the email address for."
    )

    f1, f2 = st.columns(2)
    with f1:
        company_filter = st.text_input("Company keyword", placeholder="Google")
    with f2:
        role_filter = st.text_input("Role keyword", placeholder="Associate")

    if st.button("Filter connections", disabled=not st.session_state.li_connections):
        filtered = filter_connections(
            st.session_state.li_connections, company_filter, role_filter
        )
        st.session_state.filtered_contacts = filtered
        st.session_state.selected_contacts = filtered  #default is all selected
        st.session_state.contact_selection = [True] * len(filtered)
        st.session_state.enriched_contacts = []
        st.session_state.draft_emails      = {}

    if st.session_state.filtered_contacts:
        st.success(f"{len(st.session_state.filtered_contacts)} contacts match your filters.")

        #selectable dataframe w checkboxes so user can pick who to contact
        sel_df = pd.DataFrame(st.session_state.filtered_contacts)[
            ["first_name", "last_name", "position", "company", "connected_on"]
        ].copy()
        sel_df.columns = ["First name", "Last name", "Position", "Company", "Connected on"]

        #initializing selection
        if "contact_selection" not in st.session_state or \
                len(st.session_state.contact_selection) != len(sel_df):
            st.session_state.contact_selection = [True] * len(sel_df)

        #select/deselect buttons
        btn_all, btn_none, _ = st.columns([1, 1, 5])
        if btn_all.button("✅ Select all"):
            st.session_state.contact_selection = [True] * len(sel_df)
        if btn_none.button("⬜ Deselect all"):
            st.session_state.contact_selection = [False] * len(sel_df)

        sel_df.insert(0, "Send to", st.session_state.contact_selection)
        edited_sel = st.data_editor(
            sel_df,
            use_container_width=True,
            hide_index=True,
            column_config={"Send to": st.column_config.CheckboxColumn("Send to", default=True)},
            key="contact_selector",
        )
        #checkbox state + selected contacts
        st.session_state.contact_selection = edited_sel["Send to"].tolist()
        selected_indices = [i for i, v in enumerate(st.session_state.contact_selection) if v]
        st.caption(f"{len(selected_indices)} of {len(sel_df)} contacts selected for outreach.")

        #these then are used for mail search and drafting
        st.session_state.selected_contacts = [
            st.session_state.filtered_contacts[i] for i in selected_indices
        ]

    st.divider()

    #email search on hunter api + manual email editing
    st.subheader("Step 4: find verified emails via Hunter.io")
    st.caption(
        "Emails below the confidence threshold are excluded automatically. "
        "You can manually override any email address before generating drafts."
    )

    #setting a nice slider for threshold
    confidence_threshold = st.slider("Minimum confidence threshold", 50, 99, 60, step=5)

    if st.button(
        "Find emails",
        disabled=not st.session_state.selected_contacts,
    ):
        enriched = []
        progress = st.progress(0, text="Looking up emails...")
        total = len(st.session_state.selected_contacts)
        for i, contact in enumerate(st.session_state.selected_contacts):
            # LinkedIn API does not expose connection emails — use Hunter.io for all lookups
            domain = company_to_domain(contact["company"])
            result = lookup_email_hunter(
                contact["first_name"], contact["last_name"],
                domain, HUNTER_API_KEY,
                min_score=confidence_threshold,
            )
            if result:
                enriched.append({
                    **contact,
                    "email":      result["email"],
                    "confidence": result["score"],
                })
            progress.progress((i + 1) / total, text=f"Checked {i+1}/{total}...")
        progress.empty()
        st.session_state.enriched_contacts = enriched
        if enriched:
            st.success(
                f"{len(enriched)} verified emails found above {confidence_threshold}% confidence."
            )
        else:
            st.warning(
                "No emails found above the confidence threshold. "
                "Try lowering it or broadening your filter."
            )

    #making an editable email table w hunter results + modifiable field
    if st.session_state.enriched_contacts:
        st.markdown("**Review and edit email addresses before proceeding:**")
        updated_enriched = []
        for idx, contact in enumerate(st.session_state.enriched_contacts):
            col_info, col_email = st.columns([3, 2])
            with col_info:
                st.markdown(
                    "**" + contact["first_name"] + " " + contact["last_name"] + "**  "
                    + contact["position"] + " @ " + contact["company"]
                )
                st.caption("Hunter confidence: " + str(contact.get("confidence", "?")) + "%")
            with col_email:
                overridden = st.text_input(
                    "Email address",
                    value=contact["email"],
                    key="email_override_" + str(idx),
                    label_visibility="collapsed",
                )
            updated_enriched.append({**contact, "email": overridden})
        st.session_state.enriched_contacts = updated_enriched

    st.divider()

    #personalised email drafts
    st.subheader("Step 5: generate personalised email drafts")

    role_options = (
        [
            st.session_state.app_df.loc[i, "Company"]
            + " - "
            + st.session_state.app_df.loc[i, "Role"]
            for i in st.session_state.app_df.index
        ]
        if not st.session_state.app_df.empty else []
    )
    role_options_full = ["Type manually below..."] + role_options
    selected_role_option = st.selectbox(
        "Target role (for context in the email)", role_options_full
    )
    if selected_role_option == "Type manually below..." or not role_options:
        role_label_input = st.text_input(
            "Or type target role manually",
            placeholder="Investment Banking Analyst",
        )
    else:
        role_label_input = selected_role_option

    if st.button(
        "Generate personalised drafts",
        disabled=not (st.session_state.enriched_contacts and st.session_state.profile_summary),
    ):
        drafts = {}
        progress2 = st.progress(0, text="Writing personalised emails...")
        total2 = len(st.session_state.enriched_contacts)
        for i, contact in enumerate(st.session_state.enriched_contacts):
            key = contact["email"]
            draft = generate_personalised_email(contact, role_label_input, sender_name or "[your name]")
            drafts[key] = draft
            progress2.progress((i + 1) / total2, text=f"Drafted {i+1}/{total2}...")
        progress2.empty()
        st.session_state.draft_emails = drafts
        st.success(f"{len(drafts)} personalised drafts ready.")

    #before sending drafts must be editable and checked by user
    if st.session_state.draft_emails:
        st.markdown("**Preview and edit drafts before sending:**")
        updated_drafts = {}
        for contact in st.session_state.enriched_contacts:
            key = contact["email"]
            if key not in st.session_state.draft_emails:
                continue
            label = (
                contact["first_name"] + " " + contact["last_name"]
                + " · " + contact["position"]
                + " @ " + contact["company"]
            )
            with st.expander(label):
                st.caption(
                    "📧 " + key
                    + "  |  Hunter confidence: "
                    + str(contact.get("confidence", "?")) + "%"
                )
                edited = st.text_area(
                    "Edit draft",
                    value=st.session_state.draft_emails[key],
                    height=180,
                    key="draft_" + key,
                )
                updated_drafts[key] = edited
        if updated_drafts:
            st.session_state.draft_emails = updated_drafts

    st.divider()

    #and finally sending!
    st.subheader("Step 6: send emails")

    st.warning(
        "For testing purposes, all emails are sent to a single override address so no emails are sent to any real LinkedIn contacts. "
        "If you want to test the functionality, you can add your own email below (or leave blank to use the default demo address)."
    )
    test_recipient = st.text_input(
        "Send all test emails to this address",
        placeholder="your.email@example.com or leave blank to use " + GMAIL_SENDER_EMAIL,
        key="test_recipient_override",
    )

    gmail_ready  = bool(st.session_state.gmail_access_token)
    sender_email = GMAIL_SENDER_EMAIL

    ready_to_send = bool(
        st.session_state.draft_emails
        and gmail_ready
        and email_subject
    )
    #gmail tends to block automated emails as spam, so i space them apart
    delay_seconds = st.slider(
        "Delay between emails (seconds) — recommended >= 90 to avoid spam filters",
        min_value=30, max_value=300, value=120, step=30,
    )

    if not ready_to_send:
        missing = []
        if not gmail_ready:                       missing.append("connect Gmail (Step 1)")
        if not st.session_state.draft_emails:     missing.append("generate drafts (Step 5)")
        if not email_subject:                      missing.append("subject line (Step 1)")
        st.info("To send, please complete: " + ", ".join(missing) + ".")

    if st.button(
        "🚀 Send " + str(len(st.session_state.draft_emails)) + " emails",
        type="primary",
        disabled=not ready_to_send,
    ):
        import time

        #we refresh access token to avoid mid-batch expiry
        try:
            new_token = refresh_access_token(GMAIL_REFRESH_TOKEN)
            if new_token:
                st.session_state.gmail_access_token = new_token
        except Exception:
            pass

        send_log = []
        contacts_to_send = [
            c for c in st.session_state.enriched_contacts
            if c["email"] in st.session_state.draft_emails
        ]

        #sending to all of them in spaced batch
        total_send  = len(contacts_to_send)
        progress3   = st.progress(0, text="Starting send sequence...")
        status_box  = st.empty()

        override = st.session_state.get("test_recipient_override", "").strip()
        actual_recipient = override if override else GMAIL_SENDER_EMAIL

        for idx, contact in enumerate(contacts_to_send):
            key        = contact["email"]
            body       = st.session_state.draft_emails[key]
            name_label = contact["first_name"] + " " + contact["last_name"]

            status_box.info(f"Sending {idx+1}/{total_send}: {name_label} — redirected to {actual_recipient}")
            try:
                send_via_gmail_api(
                    sender_email, actual_recipient, email_subject, body,
                    st.session_state.gmail_access_token,
                )
                send_log.append({"Name": name_label, "Email": actual_recipient, "Status": "Sent"})
            except Exception as e:
                send_log.append({
                    "Name": name_label, "Email": key,
                    "Status": "Failed: " + str(e)[:80],
                })

            progress3.progress((idx + 1) / total_send)

            if idx < total_send - 1:
                for remaining in range(delay_seconds, 0, -5):
                    status_box.info(
                        f"✉️ Sent to {name_label}. Next send in {remaining}s..."
                    )
                    time.sleep(5)

        progress3.empty()
        status_box.empty()
        st.session_state.send_log = send_log
        sent_count = sum(1 for r in send_log if "Sent" in r["Status"])
        st.success(f"Done — {sent_count}/{total_send} emails sent successfully.")

    if st.session_state.send_log:
        st.subheader("Send log")
        st.dataframe(pd.DataFrame(st.session_state.send_log), use_container_width=True)

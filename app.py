import streamlit as st
import pandas as pd
import PyPDF2
import requests
import json
import plotly.express as px

#using groq to avoid paying for credits on openai
#created th account n the groq api and got the api key/set up instructios from there
# Replace the long "gsk_..." string with this:
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"


st.set_page_config(page_title="Mind the Gap", layout="wide")

st.title("Your personalized career development AI coach")
st.markdown("Uncover the gap, bridge it with knowledge, and accelerate your entry into the professional market.")

#setting all my session state variables to store text from the uploaded files + ai response so it doesnt lose them at every click/rerun of the code
if "cv_text" not in st.session_state: st.session_state.cv_text = ""
if "transcript_text" not in st.session_state: st.session_state.transcript_text = ""
if "ai_response" not in st.session_state: st.session_state.ai_response = None
if "jd_texts" not in st.session_state: st.session_state.jd_texts = {}
if "app_df" not in st.session_state:
    st.session_state.app_df = pd.DataFrame(columns=["Company", "Industry", "Role", "Status"])

#here i create the function to extract text from files (i use for cv, transcript + job descriptions)
def extract_text(files):
    if not files:
        return ""
    text = ""
    file_list = files if isinstance(files, list) else [files]  #i wanna upload 1 or more files, so making them a list to be sure it works for both
    for file in file_list:
        try:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted: text += extracted + " "
        except Exception as e:
            st.error(f"Error reading {file.name}")  #adding error handling
    return text

#adding the function below bc sometimes the ai does not return only json despite instructions so this splits the response to retrieve only json part
def safe_json_parse(raw_text):
    try:
        start = raw_text.find("{")
        end = raw_text.rfind("}") + 1
        return json.loads(raw_text[start:end])
    except:
        return None

#creating sidebar as first component
#here the user can upload cv and uni trasncript
with st.sidebar:
    st.title("Your academic profile")
    cv_file = st.file_uploader("Upload your CV (PDF)", type="pdf", key="sidebar_cv")
    transcript_file = st.file_uploader("Upload your uni transcript (PDF)", type="pdf", key="sidebar_transcript")
    st.divider()

    if st.button("Upload files", type="primary"):
        if cv_file and transcript_file:
            with st.spinner("Uploading documents"):
                cv_file.seek(0)
                transcript_file.seek(0)
                st.session_state.cv_text = extract_text([cv_file])  #saving exytracted text to session state
                st.session_state.transcript_text = extract_text([transcript_file])
                if st.session_state.cv_text and st.session_state.transcript_text:
                    st.success("Academic profile updated")
                else:
                    st.warning("It was not possible to read one or more files. Please double check and try again.")
        else:
            st.error("Please check that both CV and uni transcript are selected.")

    st.divider()
    st.caption("Academic profile status") #to help user check if both files were uploaded successfully
    st.write("CV:" , "Uploaded" if st.session_state.cv_text else "Not uploaded")
    st.write("Uni transcript:", "Uploaded" if st.session_state.transcript_text else "Not uploaded")

st.header("Career development console")

tab1, tab2, tab3 = st.tabs(["Applications tracking", "Analytics", "Learning path"])

#in the first tab i want to display the first step of the user jounery
#we start w tracking applications, but in a time efficient and automated way
#we call the ai to extract company, industry + role from the jd and populate the table automatically
with tab1:
    st.subheader("Add new applications")
    #the user should wait a few secs to ensure the files are uploaded, otherwise the parsing will skip the file. can be imporved in next iterations
    new_jds = st.file_uploader("Drop job description (PDFs) here. Please wait a few seconds to ensure a successful upload.", type="pdf", accept_multiple_files=True, key="bulk_uploader")

    if st.button("Retrieve application information") and new_jds:
        with st.spinner("AI is reading the job descriptions. Please hold..."):
            new_rows = []
            new_jd_texts = []
            for jd_file in new_jds:
                jd_text = extract_text(jd_file)

                #first prompt to ai to extract the info in json format, structuring it as done in the ai for productivity skill seminar
                parse_prompt = f"Identify company, industry (1 word only, not shortenings, make sure to use consistent terminology for companies from the same industry), and role. Return ONLY JSON: {{\"company\": \"...\", \"industry\": \"...\", \"role\": \"...\"}} for: {jd_text[:1000]}"
                try:
                    resp = requests.post(GROQ_URL, headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
                        json={"model": "openai/gpt-oss-safeguard-20b",
                            "messages": [{"role": "user", "content": parse_prompt}],
                            "max_tokens": 256,
                            "temperature": 0.1})
                    
                    data = safe_json_parse(resp.json()['choices'][0]['message']['content'])
                    if data:
                        new_rows.append({"Company": data.get("company", "Unknown"),
                            "Industry": data.get("industry", "Unknown"),
                            "Role": data.get("role", "Unknown"),
                            "Status": "Not Applied"}) #setting this as default but user will be able to edit later
                        new_jd_texts.append(jd_text)

                except Exception as e:
                    st.warning(f"Could not parse {jd_file.name}: {e}")  #error handling

            if new_rows:
                start_idx = len(st.session_state.app_df)
                st.session_state.app_df = pd.concat([st.session_state.app_df, pd.DataFrame(new_rows)], ignore_index=True)

                for i, jd_text in enumerate(new_jd_texts):
                    st.session_state.jd_texts[start_idx + i] = jd_text
                st.success(f"Added {len(new_rows)} applications.")

    st.divider()
    st.subheader("Your application tracker")

    #updating edits to tab back to app_df after every interaction
    #when editing the status, it lags a bit and has t ìo be clicked twice, to be improved w next iterations but it works for base prototype
    edited = st.data_editor(st.session_state.app_df,
        num_rows="dynamic",
        use_container_width=True,
        key="main_app_editor",
        column_config={"Status": st.column_config.SelectboxColumn("Application Status", options=["Not Applied", "Applied", "Interviewing", "Rejected"], required=True)})
    #saving edited tab
    st.session_state.app_df = edited

#in the second tab i experimented a bit w visual charts
#here the user can get a visual overview of their applications at quick glance
#code source for charts in plotly from the python course from last semester
with tab2:
    st.header("Application insights")
    
    if st.session_state.app_df.empty:
        st.info("Add some applications in the Applications tracking Tab to see your analytics!")
    else:
        total_apps = len(st.session_state.app_df) #how many applications
        interviewing = len(st.session_state.app_df[st.session_state.app_df['Status'] == 'Interviewing']) #how many in interviewing step
        m1, m2 = st.columns(2)
        m1.metric("Total Applications", total_apps)
        m2.metric("Active Interviews", interviewing)

        st.divider()

        #quick overview of which stage the user is at overall
        st.subheader("Application status")
        status_counts = st.session_state.app_df['Status'].value_counts().reset_index()
        status_counts.columns = ['Status', 'Count']
        
        fig_pie = px.pie(status_counts, 
            values='Count', 
            names='Status', 
            hole=0.5,
            color_discrete_sequence=px.colors.qualitative.Pastel)
        
        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_pie, use_container_width=True)

        st.divider()

        #experimenting with tags to show different industry distribution of applications
        st.subheader("Industry distribution")
        
        all_companies = sorted(st.session_state.app_df['Industry'].unique())  #sometimes if the ai produces different names for the same industry it creates duplicates. will be imporved in next iterations
        selected_companies = st.multiselect("Select industries to compare:", 
            options=all_companies,
            default=all_companies[:3] if len(all_companies) > 3 else all_companies,)

        if selected_companies:
            filtered_df = st.session_state.app_df[st.session_state.app_df['Industry'].isin(selected_companies)]
            company_data = filtered_df['Industry'].value_counts().reset_index()
            company_data.columns = ['Industry', 'Applications']

            fig_bar = px.bar(company_data, 
                x='Industry', 
                y='Applications',
                text='Applications',
                color='Applications',
                color_continuous_scale='Reds')
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.warning("Please select at least one industry to view the chart")

with tab3:
    st.header("Gap analysis & course recommendations")

    if st.session_state.app_df.empty:
        st.info("Add applications in the Applications tracking Tab first.")
    else:
        st.subheader("Step 1: select an application")
        target_idx = st.selectbox("Which role do you want to analyse?",  #getting index of the wanted jd to retrieve from table
            options=st.session_state.app_df.index,
            format_func=lambda x: f"{st.session_state.app_df.loc[x, 'Company']} — {st.session_state.app_df.loc[x, 'Role']}") #getting company + role in dropdown

        jd_text = st.session_state.jd_texts.get(target_idx)
        if not jd_text:
            st.warning("No job description found for this application. Please re upload on the Applications tracking tab.")
        else:
            st.success(f"Job description loaded for this role.")

        if st.button("Generate gap analysis", type="primary"):
            if not st.session_state.cv_text or not st.session_state.transcript_text:
                st.warning("Please upload and process your CV and uni transcript in the sidebar first. Ensure you clicked the button.")
            elif not jd_text:
                st.warning("No job description available for this application.")
            else:
                with st.spinner("AI is cross-referencing your academic profile against the role..."):
                    job_desc = jd_text

                    system_instruction = """\You are an expert career coach. Analyze the User CV and uni transcript against the Job Description.
Return ONLY a valid JSON object with no extra text, markdown, or code fences.

{"profile_summary": "3-4 sentence paragraph on the candidate's match quality, key strengths, and overall readiness for this role.",
  "top_3_gaps": [
    {"skill": "Short skill name (2-4 words)", "match_gap": "-X% Match", "suggested_course": "Specific Coursera course name"},
    {"skill": "Short skill name (2-4 words)", "match_gap": "-X% Match", "suggested_course": "Specific Coursera course name"},
    {"skill": "Short skill name (2-4 words)", "match_gap": "-X% Match", "suggested_course": "Specific Coursera course name"}],
  "growth_plan": "A 6-month action plan addressing the gaps above. Mention specific Coursera courses. Make it time-realistic and actionable."}"""

                    payload = {"model": "openai/gpt-oss-safeguard-20b",
                        "messages": [
                            {"role": "system", "content": system_instruction},
                            {"role": "user", "content": f"JOB DESCRIPTION:\n{job_desc[:3000]}\n\nUSER CV:\n{st.session_state.cv_text[:2000]}\n\nTRANSCRIPT:\n{st.session_state.transcript_text[:2000]}"}],
                        "temperature": 0.3,
                        "max_tokens": 2048}

                    try:
                        resp = requests.post(GROQ_URL,
                            headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
                            json=payload)
                        if not resp.ok:
                            st.error(f"API Error {resp.status_code}: {resp.text}")
                        else:
                            raw = resp.json()['choices'][0]['message']['content']
                            st.session_state.ai_response = safe_json_parse(raw)
                            if not st.session_state.ai_response:
                                st.error("Could not parse AI response. Raw output:")
                                st.code(raw)
                    except Exception as e:
                        st.error(f"Analysis failed: {e}")

        #saving response from ai to session state
        if st.session_state.ai_response:
            res = st.session_state.ai_response
            st.divider()

            #giving the user a short profile overview
            st.subheader("Profile overview")
            st.write(res["profile_summary"])

            st.divider()

            #3 gaps as identified by ai
            st.subheader("Top 3 critical gaps")
            c1, c2, c3 = st.columns(3)
            for i, (col, gap) in enumerate(zip([c1, c2, c3], res["top_3_gaps"])):
                col.metric(label=gap["skill"],
                    value=f"Gap {i+1}",
                    delta=gap["match_gap"],
                    delta_color="inverse")
            st.info("Based on a cross-check of your academic files and the job description, these skills are the most critical gaps in your profile.")

            st.divider()

            #here the user gets redirected to the relevant courses on the coursera platform
            #future iterations i'd liek to make the ai compare platforms (e.g. maybe datacamp is better for a course on python vs coursera is for a course on negotiation)
            st.subheader("Recommended courses")
            for gap in res["top_3_gaps"]:
                with st.expander(f"Close the gap in **{gap['skill']}**"):
                    st.write(f"**Recommended Course:** {gap['suggested_course']}")
                    st.link_button("Find on Coursera", f"https://www.coursera.org/search?query={gap['skill'].replace(' ', '+')}")

            st.divider()

            #finally the user gets a short overview of what the 6-month development roadmap looks like
            #this is minimum viable aswer but potential to structure it better
            st.subheader("Your 6 months development roadmap")
            st.write(res["growth_plan"])
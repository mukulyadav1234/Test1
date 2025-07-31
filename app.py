"""
Resume Screening System using CrewAI - Complete Working Solution
"""

import streamlit as st
from crewai import Agent, Task, Crew
from langchain.chat_models import ChatOpenAI
import os
from dotenv import load_dotenv
from io import BytesIO
from PyPDF2 import PdfReader
import json
import re

# Load environment variables
load_dotenv()

def initialize_llm():
    """Initialize the Language Model"""
    return ChatOpenAI(
        model_name="gpt-4",
        temperature=0.3,
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )

@st.cache_resource
def create_agents(_llm):
    """Create all the required agents"""
    
    jd_analyzer = Agent(
        role="JD Analyzer",
        goal="Extract required skills from the job description",
        backstory="You are an expert HR analyst skilled at breaking down JDs into technical and soft skills.",
        llm=_llm
    )

    resume_parser = Agent(
        role="Resume Parser",
        goal="Extract skills from a candidate's resume",
        backstory="You are an AI that specializes in resume parsing and skill extraction.",
        llm=_llm
    )

    skill_matcher = Agent(
        role="Skill Matcher",
        goal="Compare JD and resume skills to find overlaps and gaps",
        backstory="An AI skilled at comparing candidate profiles with job needs.",
        llm=_llm
    )

    scoring_agent = Agent(
        role="Scoring Agent",
        goal="Generate a compatibility score and recommendation",
        backstory="HR bot expert at evaluating fit based on matched and missing skills.",
        llm=_llm
    )
    
    return jd_analyzer, resume_parser, skill_matcher, scoring_agent

def extract_text_from_pdf(uploaded_file):
    """Extract text content from PDF file using BytesIO"""
    try:
        # Use BytesIO to avoid temporary file issues
        pdf_bytes = BytesIO(uploaded_file.getvalue())
        
        # Read the PDF directly from memory
        reader = PdfReader(pdf_bytes)
        text = "\n".join(page.extract_text() for page in reader.pages if page.extract_text())
        
        # Close the BytesIO object
        pdf_bytes.close()
        
        return text
    except Exception as e:
        st.error(f"Error reading PDF: {str(e)}")
        return None

def create_tasks(jd_text, resume_text, agents):
    """Create all analysis tasks"""
    jd_analyzer, resume_parser, skill_matcher, scoring_agent = agents
    
    jd_task = Task(
        description=f"Analyze the following job description and extract required skills:\n{jd_text}",
        agent=jd_analyzer,
        expected_output="A structured summary of key skills, qualifications, and experience requirements."
    )

    resume_task = Task(
        description=f"Parse the following resume and extract candidate skills:\n{resume_text}",
        agent=resume_parser,
        expected_output="A structured summary of candidate's skills, experience, and qualifications."
    )

    skill_match_task = Task(
        description="Match candidate skills against job description requirements and identify matches, gaps, and extras.",
        agent=skill_matcher,
        expected_output="List of matching, missing, and extra skills.",
        context=[jd_task, resume_task]
    )

    scoring_task = Task(
        description="Evaluate overall candidate-job fit and generate a final score with reasoning.",
        agent=scoring_agent,
        expected_output="Compatibility score out of 100 with a summary.",
        context=[skill_match_task]
    )

    return [jd_task, resume_task, skill_match_task, scoring_task]

def extract_score_from_text(text):
    """Extract numerical score from text"""
    patterns = [
        r'(\d+)\s*(?:out of|/)\s*100',
        r'score:?\s*(\d+)',
        r'compatibility:?\s*(\d+)',
        r'rating:?\s*(\d+)'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text.lower())
        if match:
            return int(match.group(1))
    return None

def parse_skills_list(text):
    """Extract skills from text into a list"""
    skills = []
    
    # Pattern for numbered lists (1. skill, 2. skill)
    numbered_pattern = r'\d+\.\s*([^\n\d]+?)(?=\n\d+\.|$)'
    numbered_matches = re.findall(numbered_pattern, text, re.MULTILINE)
    
    # Pattern for bullet points (- skill, • skill)
    bullet_pattern = r'[-•]\s*([^\n-•]+?)(?=\n[-•]|$)'
    bullet_matches = re.findall(bullet_pattern, text, re.MULTILINE)
    
    # Combine and clean
    all_matches = numbered_matches + bullet_matches
    
    for match in all_matches:
        skill = match.strip().rstrip(',').rstrip('.')
        if skill and len(skill) > 2:
            skills.append(skill)
    
    return skills if skills else [text.strip()]

def structure_output(tasks, result):
    """Convert CrewAI output to structured format"""
    
    # Extract task outputs
    jd_analysis = str(tasks[0].output) if len(tasks) > 0 else ""
    resume_analysis = str(tasks[1].output) if len(tasks) > 1 else ""
    skill_matching = str(tasks[2].output) if len(tasks) > 2 else ""
    final_score = str(tasks[3].output) if len(tasks) > 3 else ""
    
    # Extract compatibility score
    score = extract_score_from_text(str(result))
    if not score:
        score = extract_score_from_text(final_score)
    
    # Parse job requirements
    job_technical_skills = parse_skills_list(jd_analysis) if jd_analysis else []
    
    # Parse candidate skills from resume analysis
    candidate_technical_skills = []
    candidate_experience = []
    candidate_education = []
    
    if resume_analysis:
        lines = resume_analysis.split('\n')
        current_section = ""
        
        for line in lines:
            line = line.strip()
            if any(keyword in line.lower() for keyword in ['skills:', 'technical', 'proficiency']):
                current_section = "technical"
            elif any(keyword in line.lower() for keyword in ['experience:', 'work', 'intern']):
                current_section = "experience" 
            elif any(keyword in line.lower() for keyword in ['education:', 'qualification', 'degree']):
                current_section = "education"
            elif line and current_section == "technical":
                candidate_technical_skills.extend(parse_skills_list(line))
            elif line and current_section == "experience":
                candidate_experience.append(line)
            elif line and current_section == "education":
                candidate_education.append(line)
    
    # Parse skill matching
    matching_skills = []
    missing_skills = []
    additional_skills = []
    
    if skill_matching:
        lines = skill_matching.split('\n')
        current_section = ""
        
        for line in lines:
            line = line.strip()
            if any(keyword in line.lower() for keyword in ['matching', 'match']):
                current_section = "matching"
            elif any(keyword in line.lower() for keyword in ['missing', 'lack', 'absent']):
                current_section = "missing"  
            elif any(keyword in line.lower() for keyword in ['additional', 'extra', 'bonus']):
                current_section = "additional"
            elif line and current_section == "matching":
                matching_skills.extend(parse_skills_list(line))
            elif line and current_section == "missing":
                missing_skills.extend(parse_skills_list(line))
            elif line and current_section == "additional":
                additional_skills.extend(parse_skills_list(line))
    
    structured_result = {
        "compatibility_score": score if score else "Not Available",
        "reason": str(result).strip(),
        "job_requirements": {
            "technical_skills": job_technical_skills[:10],
            "soft_skills": []
        },
        "candidate_profile": {
            "technical_skills": candidate_technical_skills[:15],
            "soft_skills": [],
            "experience": candidate_experience[:5],
            "education": candidate_education[:3]
        },
        "skill_comparison": {
            "matching_skills": matching_skills[:10],
            "missing_skills": missing_skills[:10], 
            "additional_skills": additional_skills[:10]
        }
    }
    
    return structured_result

def display_main_results(structured_data):
    """Display main results in a user-friendly format"""
    
    # Main Score Display
    st.subheader("🎯 Compatibility Analysis Results")
    
    score = structured_data.get("compatibility_score", "Not Available")
    
    # Score with color coding
    if score != "Not Available" and isinstance(score, int):
        if score >= 80:
            st.success(f"🎉 **Compatibility Score: {score}/100**")
            st.success("**Recommendation: Highly Recommended**")
        elif score >= 60:
            st.warning(f"👍 **Compatibility Score: {score}/100**")
            st.warning("**Recommendation: Recommended**")
        else:
            st.error(f"⚠️ **Compatibility Score: {score}/100**")
            st.error("**Recommendation: Needs Review**")
    else:
        st.info("**Compatibility Score: Not Available**")
    
    # Reason/Summary
    st.subheader("📝 Summary")
    st.write(structured_data.get("reason", "No summary available"))
    
    # Skills Analysis in columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("✅ Matching Skills")
        matching_skills = structured_data.get("skill_comparison", {}).get("matching_skills", [])
        if matching_skills:
            for skill in matching_skills:
                st.write(f"• {skill}")
        else:
            st.write("No matching skills identified")
            
        st.subheader("➕ Additional Skills (Bonus)")
        additional_skills = structured_data.get("skill_comparison", {}).get("additional_skills", [])
        if additional_skills:
            for skill in additional_skills:
                st.write(f"• {skill}")
        else:
            st.write("No additional skills identified")
    
    with col2:
        st.subheader("❌ Missing Skills")
        missing_skills = structured_data.get("skill_comparison", {}).get("missing_skills", [])
        if missing_skills:
            for skill in missing_skills:
                st.write(f"• {skill}")
        else:
            st.write("No missing skills identified")
            
        st.subheader("📊 Quick Stats")
        st.metric("Total Matching Skills", len(matching_skills))
        st.metric("Missing Skills", len(missing_skills))
        st.metric("Bonus Skills", len(additional_skills))
    
    # Candidate Profile Summary
    st.subheader("👤 Candidate Profile")
    candidate_profile = structured_data.get("candidate_profile", {})
    
    # Experience
    experience = candidate_profile.get("experience", [])
    if experience:
        st.write("**Experience:**")
        for exp in experience:
            st.write(f"• {exp}")
    
    # Education  
    education = candidate_profile.get("education", [])
    if education:
        st.write("**Education:**")
        for edu in education:
            st.write(f"• {edu}")
    
    # Technical Skills
    tech_skills = candidate_profile.get("technical_skills", [])
    if tech_skills:
        st.write("**Technical Skills:**")
        skills_text = ", ".join(tech_skills)
        st.write(skills_text)

def main():
    """Main function"""
    st.title("💼 Resume Screening System")
    st.markdown("### Powered by CrewAI & OpenAI GPT-4")
    
    # Check API key
    if not os.getenv("OPENAI_API_KEY"):
        st.error("❌ OPENAI_API_KEY not found. Please add it to your .env file.")
        st.info("Create a .env file in your project directory and add: OPENAI_API_KEY=your_api_key_here")
        return
    
    # Initialize components
    try:
        llm = initialize_llm()
        agents = create_agents(llm)
        st.success("✅ AI agents initialized successfully!")
    except Exception as e:
        st.error(f"❌ Error initializing AI components: {str(e)}")
        return
    
    # Input Section
    st.markdown("---")
    
    # Job Description Input
    st.subheader("📋 Job Description")
    jd_method = st.radio("Choose input method:", ["Text Input", "PDF Upload"], key="jd_method")
    
    jd_text = None
    
    if jd_method == "Text Input":
        jd_text = st.text_area(
            "Enter Job Description:", 
            height=200,
            placeholder="Paste the job description here..."
        )
    else:
        jd_file = st.file_uploader(
            "Upload Job Description PDF:", 
            type=['pdf'], 
            key="jd_file"
        )
        if jd_file:
            with st.spinner("Extracting text from JD PDF..."):
                jd_text = extract_text_from_pdf(jd_file)
            if jd_text:
                st.success("✅ Job description loaded!")
                with st.expander("Preview JD Content"):
                    st.text_area("Extracted Text:", jd_text[:500] + "...", height=100, disabled=True)
    
    # Resume Input
    st.subheader("📄 Resume")
    resume_file = st.file_uploader(
        "Upload Resume PDF:", 
        type=['pdf'], 
        key="resume_file"
    )
    
    resume_text = None
    if resume_file:
        with st.spinner("Extracting text from Resume PDF..."):
            resume_text = extract_text_from_pdf(resume_file)
        if resume_text:
            st.success("✅ Resume loaded!")
            with st.expander("Preview Resume Content"):
                st.text_area("Extracted Text:", resume_text[:500] + "...", height=100, disabled=True)
    
    # Analysis Section
    st.markdown("---")
    st.subheader("🚀 Analysis")
    
    # Status check
    if not (jd_text and resume_text):
        if not jd_text:
            st.warning("⚠️ Please provide a job description")
        if not resume_text:
            st.warning("⚠️ Please upload a resume")
    
    # Analysis Button
    if st.button("🔍 Run Compatibility Analysis", type="primary", use_container_width=True):
        if jd_text and resume_text:
            with st.spinner("🤖 AI agents are analyzing... This may take a few moments."):
                try:
                    # Progress bar
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    status_text.text("Creating analysis tasks...")
                    progress_bar.progress(25)
                    
                    tasks = create_tasks(jd_text, resume_text, agents)
                    
                    status_text.text("Running AI crew analysis...")
                    progress_bar.progress(50)
                    
                    crew = Crew(
                        agents=list(agents),
                        tasks=tasks,
                        verbose=False
                    )
                    
                    result = crew.kickoff()
                    progress_bar.progress(75)
                    
                    status_text.text("Processing results...")
                    structured_output = structure_output(tasks, result)
                    progress_bar.progress(100)
                    
                    status_text.text("✅ Analysis complete!")
                    
                    # Clear progress indicators
                    progress_bar.empty()
                    status_text.empty()
                    
                    # Display main results
                    st.markdown("---")
                    display_main_results(structured_output)
                    
                    # Advanced view section
                    st.markdown("---")
                    with st.expander("🔧 Advanced View & Downloads"):
                        
                        # Tabs for organized view
                        tab1, tab2, tab3 = st.tabs(["📊 Structured Data", "📥 Downloads", "🔍 Raw Analysis"])
                        
                        with tab1:
                            st.subheader("Structured JSON Output")
                            st.json(structured_output)
                        
                        with tab2:
                            st.subheader("Export Options")
                            
                            # JSON download
                            json_str = json.dumps(structured_output, indent=2)
                            st.download_button(
                                label="📥 Download JSON Report",
                                data=json_str,
                                file_name=f"resume_analysis_{resume_file.name if resume_file else 'candidate'}.json",
                                mime="application/json",
                                key="download_json"
                            )
                            
                            # Text report
                            candidate_name = resume_file.name.replace('.pdf', '') if resume_file else "Candidate"
                            text_report = f"""RESUME SCREENING REPORT
{'='*50}
Candidate: {candidate_name}
Compatibility Score: {structured_output.get('compatibility_score', 'N/A')}/100

SUMMARY:
{structured_output.get('reason', 'No summary available')}

MATCHING SKILLS:
{chr(10).join(f"• {skill}" for skill in structured_output.get('skill_comparison', {}).get('matching_skills', []))}

MISSING SKILLS:
{chr(10).join(f"• {skill}" for skill in structured_output.get('skill_comparison', {}).get('missing_skills', []))}

ADDITIONAL SKILLS:
{chr(10).join(f"• {skill}" for skill in structured_output.get('skill_comparison', {}).get('additional_skills', []))}
"""
                            
                            st.download_button(
                                label="📄 Download Text Report",
                                data=text_report,
                                file_name=f"resume_analysis_{candidate_name}.txt",
                                mime="text/plain",
                                key="download_text"
                            )
                        
                        with tab3:
                            st.subheader("Raw AI Analysis")
                            for i, task in enumerate(tasks, 1):
                                st.write(f"**Task {i} - {task.agent.role}:**")
                                st.write(task.output)
                                st.divider()
                            
                            st.write("**Final Result:**")
                            st.write(result)
                    
                except Exception as e:
                    st.error(f"❌ Analysis failed: {str(e)}")
                    st.info("💡 Please check your API key and internet connection, then try again.")
        else:
            st.error("❌ Please provide both Job Description and Resume before running analysis.")
    
    # Footer
    st.markdown("---")
    st.markdown("Built with 💝 using CrewAI, OpenAI GPT-4, and Streamlit")

if __name__ == "__main__":
    main()

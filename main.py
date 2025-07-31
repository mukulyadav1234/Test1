"""
Resume Screening System using CrewAI - Simple Streamlit Version with Structured Output
"""

import streamlit as st
from crewai import Agent, Task, Crew
from langchain.chat_models import ChatOpenAI
import os
from dotenv import load_dotenv
import tempfile
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
    """Extract text content from PDF file"""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file.flush()
            
            reader = PdfReader(tmp_file.name)
            text = "\n".join(page.extract_text() for page in reader.pages if page.extract_text())
            
            os.unlink(tmp_file.name)
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
    # Look for numbered lists, bullet points, or comma-separated items
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
        if skill and len(skill) > 2:  # Filter out very short matches
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
    job_technical_skills = []
    job_soft_skills = []
    if "technical" in jd_analysis.lower() or "skills" in jd_analysis.lower():
        # Extract technical skills from JD analysis
        job_technical_skills = parse_skills_list(jd_analysis)
    
    # Parse candidate skills
    candidate_technical_skills = []
    candidate_soft_skills = []
    candidate_experience = []
    candidate_education = []
    
    if resume_analysis:
        # Split resume analysis into sections
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
    
    # Create structured output
    structured_result = {
        "compatibility_score": score if score else "Not Available",
        "reason": str(result).strip(),
        "job_requirements": {
            "technical_skills": job_technical_skills[:10],  # Limit to 10 for readability
            "soft_skills": job_soft_skills[:5]
        },
        "candidate_profile": {
            "technical_skills": candidate_technical_skills[:15],
            "soft_skills": candidate_soft_skills[:5],
            "experience": candidate_experience[:5],
            "education": candidate_education[:3]
        },
        "skill_comparison": {
            "matching_skills": matching_skills[:10],
            "missing_skills": missing_skills[:10],
            "additional_skills": additional_skills[:10]
        },
        "raw_analysis": {
            "job_description_analysis": jd_analysis,
            "resume_analysis": resume_analysis,
            "skill_matching": skill_matching,
            "final_scoring": final_score
        }
    }
    
    return structured_result

def main():
    """Main function"""
    st.title("💼 Resume Screening System")
    
    # Check API key
    if not os.getenv("OPENAI_API_KEY"):
        st.error("❌ OPENAI_API_KEY not found. Please add it to your .env file.")
        return
    
    # Initialize components
    try:
        llm = initialize_llm()
        agents = create_agents(llm)
        st.success("✅ AI agents ready!")
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        return
    
    # Job Description Input
    st.subheader("📋 Job Description")
    jd_method = st.radio("Input method:", ["Text", "PDF"])
    
    jd_text = None
    
    if jd_method == "Text":
        jd_text = st.text_area("Enter Job Description:", height=200)
    else:
        jd_file = st.file_uploader("Upload JD PDF:", type=['pdf'])
        if jd_file:
            jd_text = extract_text_from_pdf(jd_file)
            if jd_text:
                st.success("✅ JD loaded")
    
    # Resume Input
    st.subheader("📄 Resume")
    resume_file = st.file_uploader("Upload Resume PDF:", type=['pdf'])
    
    resume_text = None
    if resume_file:
        resume_text = extract_text_from_pdf(resume_file)
        if resume_text:
            st.success("✅ Resume loaded")
    
    # Analysis
    if st.button("🔍 Run Analysis") and jd_text and resume_text:
        with st.spinner("Running analysis..."):
            try:
                tasks = create_tasks(jd_text, resume_text, agents)
                
                crew = Crew(
                    agents=list(agents),
                    tasks=tasks,
                    verbose=False
                )
                
                result = crew.kickoff()
                
                # Structure the output
                structured_output = structure_output(tasks, result)
                
                # Display structured results
                st.subheader("✅ Structured Results:")
                
                # Display as formatted JSON
                st.json(structured_output)
                
                # Display key metrics prominently
                st.subheader("📊 Key Metrics:")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Compatibility Score", 
                             f"{structured_output['compatibility_score']}/100" if structured_output['compatibility_score'] != "Not Available" else "N/A")
                
                with col2:
                    st.metric("Matching Skills", len(structured_output['skill_comparison']['matching_skills']))
                
                with col3:
                    st.metric("Missing Skills", len(structured_output['skill_comparison']['missing_skills']))
                
                # Download structured output
                st.subheader("💾 Download Results:")
                
                # JSON download
                json_str = json.dumps(structured_output, indent=2)
                st.download_button(
                    label="📥 Download Structured JSON",
                    data=json_str,
                    file_name="structured_resume_analysis.json",
                    mime="application/json"
                )
                
                # Optional: Display raw results
                with st.expander("🔍 View Raw Analysis"):
                    for i, task in enumerate(tasks, 1):
                        st.write(f"**Task {i} ({task.agent.role}):**")
                        st.write(task.output)
                        st.write("---")
                    
                    st.write("**🎯 Final Score and Summary:**")
                    st.write(result)
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
    
    elif st.button("🔍 Run Analysis"):
        st.error("Please provide both Job Description and Resume")

if __name__ == "__main__":
    main()

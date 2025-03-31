import streamlit as st
import base64
import pandas as pd
import numpy as np
import PyPDF2
from io import BytesIO
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download NLTK resources
nltk.download('stopwords')
nltk.download('punkt')

# Set page config
st.set_page_config(
    page_title="Resume Screening AI",
    page_icon="📄",
    layout="wide"
)
# Custom background and UI styling
def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
    bg_image = f"""
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{encoded_string}");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    </style>
    """
    st.markdown(bg_image, unsafe_allow_html=True)

# Set futuristic theme and UI enhancements
add_bg_from_local("bg4.jpg")

# Header
st.markdown("""
<div class="header">
    <h1>Resume Screening <span style="color: #5DA0F6;">AI</span></h1>
    <p>Find the perfect candidates effortlessly. Upload job descriptions and resumes to get AI-powered rankings.</p>
</div>
""", unsafe_allow_html=True)

# Add custom CSS
st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #EBF5FF 0%, #E1EFFE 100%);
    }
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .card {
        background: white;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.05);
    }
    .header {
        text-align: center;
        padding: 20px 0;
    }
    .footer {
        text-align: center;
        padding: 20px 0;
        color: #666;
        border-top: 1px solid #eaeaea;
        margin-top: 40px;
    }
    .skill-badge {
        display: inline-block;
        background: #5DA0F6;
        color: white;
        padding: 5px 10px;
        margin: 3px;
        border-radius: 15px;
        font-size: 14px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Function to preprocess text
def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
    tokens = nltk.word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(filtered_tokens)

# Function to calculate similarity
def calculate_similarity(job_desc, resume_text):
    corpus = [job_desc, resume_text]
    vectorizer = CountVectorizer()
    count_matrix = vectorizer.fit_transform(corpus)
    similarity = cosine_similarity(count_matrix)[0][1]
    return round(similarity * 100, 2)

# Function to extract key skills
def extract_key_skills(text, common_skills):
    skills_found = []
    for skill in common_skills:
        if re.search(r'\b' + re.escape(skill.lower()) + r'\b', text.lower()):
            skills_found.append(skill)
    return skills_found

# Common tech skills
common_skills = ['Python', 'JavaScript', 'TypeScript', 'React', 'Node.js', 'Java', 'C++', 'C#', 
    'SQL', 'NoSQL', 'MongoDB', 'PostgreSQL', 'MySQL', 'AWS', 'Azure', 'GCP', 
    'Docker', 'Kubernetes', 'Machine Learning', 'AI', 'Data Science', 'Data Analysis',
    'HTML', 'CSS', 'Git', 'TensorFlow', 'PyTorch', 'Django', 'Flask', 'Express',
    'REST API', 'GraphQL', 'DevOps', 'CI/CD', 'Agile', 'Scrum', 'Product Management',
    'Leadership', 'Communication', 'Problem Solving', 'Critical Thinking',
    'Redux', 'Vue.js', 'Angular', 'Spring', 'Hibernate', 'JPA', 'ORM',
    'Unit Testing', 'TDD', 'BDD', 'Jenkins', 'Travis CI', 'CircleCI',
    'Linux', 'Unix', 'Windows', 'Scripting', 'Shell', 'Bash',
    'Frontend', 'Backend', 'Full Stack', 'Mobile Development', 'iOS', 'Android',
    'UX/UI', 'Design', 'Figma', 'Adobe', 'Photoshop', 'Illustrator',
    'Project Management', 'JIRA', 'Confluence', 'Trello', 'Asana',
    'Data Visualization', 'Tableau', 'Power BI', 'D3.js',
    'Big Data', 'Hadoop', 'Spark', 'MapReduce', 'ETL',
    'Networking', 'Security', 'Cryptography', 'Authentication', 'Authorization',
    'Cloud Computing', 'Serverless', 'Lambda', 'Functions', 'Microservices',
    'RESTful', 'SOAP', 'API Design', 'Swagger', 'OpenAPI',
     'MATLAB', 'SAS', 'SPSS', 'Excel', 'VBA',
    'Swift', 'Kotlin', 'Objective-C', 'Flutter', 'React Native',
    'Scala', 'Rust', 'Go', 'Ruby', 'PHP', 'Perl', 'Haskell',
    'Blockchain', 'Smart Contracts', 'Ethereum', 'Solidity',
    'Natural Language Processing', 'Computer Vision', 'Deep Learning',
    'Statistics', 'Probability', 'Linear Algebra', 'Calculus',
    'Database Design', 'Normalization', 'Denormalization', 'Indexing',
    'Load Balancing', 'Caching', 'CDN', 'Performance Optimization',
    'UI/UX', 'Responsive Design', 'Mobile First', 'Accessibility',
    'SEO', 'Google Analytics', 'Marketing', 'Growth Hacking',
    'Leadership', 'Management', 'Team Building', 'Mentoring',
    'Scientific Computing', 'Computational Physics', 'Bioinformatics',
    'Game Development', 'Unity', 'Unreal Engine', 'WebGL',
    'Embedded Systems', 'IoT', 'Robotics', 'Hardware', 'Firmware']

# UI Components
col1, col2 = st.columns([1, 1])
with col1:
    st.subheader("Job Description")
    job_description = st.text_area("Paste the job description here...", height=150)
with col2:
    st.subheader("Upload Resumes (PDF)")
    uploaded_files = st.file_uploader("Select files", accept_multiple_files=True, type=['pdf'])

process_button = st.button("Rank Resumes")

if process_button and job_description and uploaded_files:
    with st.spinner("Analyzing resumes..."):
        processed_jd = preprocess_text(job_description)
        results = []
        
        for file in uploaded_files:
            resume_text = extract_text_from_pdf(BytesIO(file.read()))
            file.seek(0)
            processed_resume = preprocess_text(resume_text)
            match_score = calculate_similarity(processed_jd, processed_resume)
            key_skills = extract_key_skills(resume_text, common_skills)
            results.append({"filename": file.name, "match_score": match_score, "key_skills": key_skills})
        
        results.sort(key=lambda x: x["match_score"], reverse=True)
        
        st.subheader("Resume Ranking Results")
        for i, result in enumerate(results):
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(f"**{i+1}. {result['filename']}**")
                st.progress(result["match_score"] / 100)
                st.markdown(f"Match Score: **{result['match_score']}%**")
            with col2:
                st.markdown("**Key Skills:**")
                if result['key_skills']:
                    skill_tags = "".join([f'<span class="skill-badge">{skill}</span>' for skill in result['key_skills']])
                    st.markdown(skill_tags, unsafe_allow_html=True)
                else:
                    st.markdown("_No matching skills found._")
            st.divider()

# Footer with additional information
st.markdown("""
    <div style="text-align: center; padding: 20px; margin-top: 50px; background: rgba(255, 255, 255, 0.1); border-radius: 10px;">
    <h2 style="color: white;">About This Tool</h2>
    <p style="color: white;"> 
        Author: Gaurav Kumar <br>
        Technology: NLP, Machine Learning & Scikit-Learn <br>
        Model: Cosine Similarity & Bag-of-Words (BoW)
    </p>
    </div>
""", unsafe_allow_html=True)

st.markdown("<div class='footer'><p>© 2023 Resume Screening AI. All rights reserved.</p></div>", unsafe_allow_html=True)

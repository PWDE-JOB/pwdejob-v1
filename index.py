from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from model import User
from collections import Counter
import os

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.get("/")
def root():
    return {"message": "working"}



user_temp = []

# Jobs-Type with category
jobs_tags = pd.DataFrame({
    "job_type": [
        "Email Management", "Bookkeeping", "Research", "Basic Programming",
        "Music Production", "Photoshop", "Figma", "Canva", "SEO", "Typing",
        "Sewing", "Pattern Cutting", "Wood Cutting", "Measuring Tools",
        "Beading", "Glue Gun", "Craft Assembly", "Tool Handling",
        "Tire Patching", "Blog Writing", "Grammarly", "Zendesk",
        "Customer Service", "Facebook Ads", "Instagram", "TranscribeMe",
        "Audio Typing"
    ],
    "category": [
        "Digital Management", "Administrative", "Professional", "Digital Coding",
        "Creative Audio", "Creative Digital", "Creative Digital", "Creative Digital", "Marketing", "Administrative",
        "Manual", "Manual", "Labor", "Manual",
        "Manual", "Manual", "Manual", "Mechanical",
        "Mechanical", "Content Writing", "Content Writing", "Support",
        "Support", "Marketing", "Marketing", "Transcription",
        "Transcription"
    ]
})

sample_jobs = pd.DataFrame([
    {
        "job_title": "Graphic Designers Needed",
        "pwd_accepted?": "Yes",
        "job_type": ["Photoshop", "Canva"],
    },
    {
        "job_title": "Remote Data Entry Specialist",
        "pwd_accepted?": "No",
        "job_type": ["Excel", "Typing"],
    },
    {
        "job_title": "Sewing Machine Operator",
        "pwd_accepted?": "Yes",
        "job_type": ["Sewing", "Pattern Cutting"],
    },
    {
        "job_title": "Carpentry Assistant",
        "pwd_accepted?": "Yes",
        "job_type": ["Wood Cutting", "Measuring Tools"],
    },
    {
        "job_title": "Handmade Crafts Assembler",
        "pwd_accepted?": "Yes",
        "job_type": ["Beading", "Glue Gun", "Craft Assembly"],
    },
    {
        "job_title": "Bicycle Repair Technician",
        "pwd_accepted?": "No",
        "job_type": ["Tool Handling", "Tire Patching"],
    },
    {
        "job_title": "Content Writer",
        "pwd_accepted?": "Yes",
        "job_type": ["SEO", "Blog Writing", "Grammarly"],
    },
    {
        "job_title": "Customer Support Chat Agent",
        "pwd_accepted?": "Yes",
        "job_type": ["Zendesk", "Typing", "Customer Service"],
    },
    {
        "job_title": "Social Media Manager",
        "pwd_accepted?": "Yes",
        "job_type": ["Facebook Ads", "Canva", "Instagram"],
    },
    {
        "job_title": "Transcriptionist",
        "pwd_accepted?": "No",
        "job_type": ["TranscribeMe", "Audio Typing"],
    },
    {
        "job_title": "Marketing Specialist",
        "pwd_accepted?": "No",
        "job_type": ["SEO", "Email Management"]
    },
    {
        "job_title": "Support Service Representative",
        "pwd_accepted?": "No",
        "job_type": ["Customer Service", "Basic Programming", "Research"]
    },
    {
        "job_title": "Creative Digital Creative Specialist",
        "pwd_accepted?": "No",
        "job_type": ["Music Production", "Figma"]
    },
    {
        "job_title": "Manual Operator",
        "pwd_accepted?": "No",
        "job_type": ["Pattern Cutting"]
    },
    {
        "job_title": "Administrative Assistant",
        "pwd_accepted?": "No",
        "job_type": ["Typing"]
    },
    {
        "job_title": "Content Writing Writer",
        "pwd_accepted?": "Yes",
        "job_type": ["Glue Gun", "Grammarly"]
    },
    {
        "job_title": "Professional Consultant",
        "pwd_accepted?": "No",
        "job_type": ["Beading", "Photoshop", "Research"]
    },
    {
        "job_title": "Support Helpdesk Staff",
        "pwd_accepted?": "No",
        "job_type": ["Customer Service"]
    },
    {
        "job_title": "Professional Researcher",
        "pwd_accepted?": "Yes",
        "job_type": ["Bookkeeping", "Research"]
    },
    {
        "job_title": "Creative Digital Creative Specialist",
        "pwd_accepted?": "No",
        "job_type": ["Tool Handling", "Grammarly", "Figma"]
    }
])

#making a user choose from sambple jobs to fill the "hitsory part" initially

def weightingSkills(skills):
    # weights = [3,2,1]
    
    # return ' '.join(
    #     ' '.join([skill]*weight) for skill, weight in zip(skills, weights)
    # )
    user_categories = jobs_tags[jobs_tags["job_type"].isin(skills)]["category"]

    # Count category occurrences
    category_weights = Counter(user_categories)
    return category_weights

def score_job(job_skills, category_weights):
    score = 0
    for skill in job_skills:
        category = jobs_tags[jobs_tags["job_type"] == skill]["category"]
        if not category.empty:
            score += category_weights.get(category.values[0], 0)
    return score

# connect each job with primary category (based on job_type to category)
def inferPrimaryCategory(job_types):
    categories = jobs_tags[jobs_tags["job_type"].isin(job_types)]["category"].tolist()
    
    if categories:
        return categories[0]
    else:
        return "Unknown"
    
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
    
def creatingRecommendedJobs(skills, disability, alpha=0.8, beta=0.2, top_N=5):
    
    if disability.strip().lower() != "none":
        jobs_to_score = sample_jobs[sample_jobs["pwd_accepted?"] == "Yes"].copy()
    else:
        jobs_to_score = sample_jobs.copy()
        
    # Add tfidf to lessen the braod recos by just using category matching (tbh i don't see the diference but lets add it just to be sure.....y'know)
    jobs_to_score["skills_text"] = jobs_to_score["job_type"].apply(lambda x: ''.join(x))
    user_skills_text = ' '.join(skills)
    
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform([user_skills_text]+jobs_to_score["skills_text"].tolist())
    cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
    
    #category matching
    category_weights = weightingSkills(skills)
    total_weight = sum(category_weights.values())
    if total_weight == 0:
        return pd.DataFrame(columns=sample_jobs.columns)
    
    
    # Apply scoring
    category_scores  = jobs_to_score["job_type"].apply(lambda x: score_job(x, category_weights))
    print(category_scores)
    

    #ADD SCORES OF TIFDIF (THE BOOSTER) and category scores
    jobs_to_score["match_score"] = alpha * cosine_sim + beta * category_scores
        
    jobs_to_score["primary_category"] = jobs_to_score["job_type"].apply(inferPrimaryCategory)
    
    results = []
    for category, count in category_weights.items():
        portion = round((count/total_weight) * top_N)
        top_jobs = jobs_to_score[jobs_to_score["primary_category"] == category]
        top_jobs = top_jobs.sort_values(by="match_score", ascending=False).head(portion)
        results.append(top_jobs)
        
    
    # To avoid unsahble types shiiii    
    df = pd.concat(results)
    for col in df.columns:
        if df[col].apply(lambda x: isinstance(x, list)).any():
            df[col] = df[col].apply(str)
    
    #FINALZATION
    final = df.drop_duplicates().sort_values(by="match_score", ascending=False).head(top_N)
    
    return final.reset_index(drop=True)

@app.post("/user")
async def creationOfUser(user: User):
    
    #create a custom user data structure to be appended in the temporaray storage
    user_data = {
        "Name": user.name,
        "Disability": user.disability,
        "skills": user.skills,
        "Most Priority Skills": user.skills[0],
        "2nd Choice Skill": user.skills[1],
        "3rd Choice Skill": user.skills[2]
    }
    
    user_temp.append(user_data)
    
    recommendations = creatingRecommendedJobs(user.skills, user.disability)
    top_jobs = recommendations.head(5)["job_title"].tolist()
    
    return{
        "status": "Successfull",
        "Name": user.name,
        "Disability": user.disability,
        "Most Priority Skills": user.skills[0],
        "2nd Choice Skill": user.skills[1],
        "3rd Choice Skill": user.skills[2],
        "Recomeded Jobs": top_jobs
    }

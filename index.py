from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
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

from model import User

user_temp = []

import pandas as pd
sample_jobs = pd.DataFrame([
    {
        "job_title": "Graphic Designers Needed",
        "pwd_accepted?": "Yes",
        "job_type": ["Photoshop", "Figma"],
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
    }
])

#making a user choose from sambple jobs to fill the "hitsory part" initially

def weightingSkills(skills):
    weights = [3,2,1]
    
    return ' '.join(
        ' '.join([skill]*weight) for skill, weight in zip(skills, weights)
    )
    
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
    
def creatingRecommendedJobs():
    if not user_temp:
        return pd.DataFrame()
    
    # This weighting is for the users and sample jobs
    user_frame = pd.DataFrame(user_temp)
    user_frame['weighted_skills'] = user_frame['skills'].apply(weightingSkills)
    
    jobs_frame = sample_jobs.copy()
    
    #Check the user have a disability or not
    if user_frame.iloc[-1]["Disability"].strip().lower() != "none":
        jobs_frame = jobs_frame[jobs_frame["pwd_accepted?"] == "Yes"]
        
    jobs_frame['jobs_info'] = jobs_frame['job_type'].apply(lambda x: ' '.join(x))
    
    combined_data = user_frame['weighted_skills'].tolist() + jobs_frame['jobs_info'].tolist()
    
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(combined_data)
    
    user_vectors = tfidf_matrix[:len(user_frame)]
    job_vectors = tfidf_matrix[len(user_frame):]
    
    # getting the similarity
    similarity = cosine_similarity(user_vectors, job_vectors)
    
    top_N = 3
    reccomendations = []
    
    for i, user in user_frame.iterrows():
        sim_scores = list(enumerate(similarity[i]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        
        # use jobs_frame isntead of sample_jobs since we filter out already if the user have disability or not just need skills matching
        top_jobs = [jobs_frame.iloc[idx]['job_title'] for idx, _ in sim_scores[:top_N]]
        reccomendations.append({
            "Name": user["Name"],
            "Recomended Jobs": top_jobs
        })
    return reccomendations
    

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
    
    recommendations = creatingRecommendedJobs()
    
    return{
        "status": "Successfull",
        "Name": user.name,
        "Disability": user.disability,
        "Most Priority Skills": user.skills[0],
        "2nd Choice Skill": user.skills[1],
        "3rd Choice Skill": user.skills[2],
        "Recomeded Jobs": recommendations[-1]["Recomended Jobs"] if recommendations else []
    }

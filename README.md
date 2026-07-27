# 👥 ProfileMatch: Intelligent User Recommendation System

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-red?logo=streamlit)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-orange?logo=scikitlearn)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?logo=pandas)
![NumPy](https://img.shields.io/badge/NumPy-Numerical%20Computing-013243?logo=numpy)
![NLP](https://img.shields.io/badge/NLP-TF--IDF-success)
![KNN](https://img.shields.io/badge/KNN-Recommendation%20Engine-blueviolet)
![Recommendation System](https://img.shields.io/badge/Recommendation-System-8A2BE2)
![License](https://img.shields.io/badge/License-MIT-green)

An intelligent user recommendation system that leverages **Machine Learning**, **Natural Language Processing (NLP)**, and **Personality Compatibility** to generate personalized user recommendations. 
The application combines **TF-IDF Vectorization**, **K-Nearest Neighbors (KNN)**, and **MBTI Compatibility Analysis** to identify users with similar profiles through an interactive **Streamlit** web application.

---

## 📌 Table of Contents

- Project Overview
- Features
- Technologies Used
- System Architecture
- Machine Learning Workflow
- Dataset
- Installation
- Usage
- Project Structure
- Results
- Future Enhancements
- Acknowledgements
- Author

---

# 📖 Project Overview

Finding users with similar interests, skills, personalities, and career goals is an important problem in social networking, professional networking, and recruitment platforms.

**ProfileMatch** is an intelligent recommendation system that analyzes user profiles using textual information and structured attributes to recommend the most compatible users.

Unlike traditional recommendation systems that rely on a single similarity metric, this project combines multiple factors including:

- Profile Similarity using TF-IDF
- K-Nearest Neighbors (KNN)
- MBTI Personality Compatibility
- Location Matching
- Income Similarity

The final recommendation score is calculated using a hybrid scoring approach to provide more meaningful and personalized recommendations.

---

# ✨ Features

- Intelligent user recommendation system
- NLP-based profile similarity using TF-IDF
- K-Nearest Neighbors (KNN) recommendation engine
- MBTI personality compatibility analysis
- Location-based matching
- Income similarity analysis
- Interactive Streamlit web application
- Dynamic recommendation scoring
- Dataset statistics dashboard
- Data validation before model training
- Interactive charts and visualizations

---

# 🛠️ Technologies Used

### Programming Language

- Python

### Libraries

- Streamlit
- Pandas
- NumPy
- Scikit-learn
- SciPy

### Machine Learning

- TF-IDF Vectorization
- K-Nearest Neighbors (KNN)

### Development Tools

- VS Code
- Git
- GitHub

---

# 🏗️ System Architecture

```
User Profile
      │
      ▼
Data Preprocessing
      │
      ▼
Profile Text Generation
      │
      ▼
TF-IDF Vectorization
      │
      ▼
K-Nearest Neighbors (KNN)
      │
      ▼
MBTI Compatibility
      │
      ▼
Location & Income Matching
      │
      ▼
Hybrid Score Calculation
      │
      ▼
Top Recommended Users
```

---

# 🤖 Machine Learning Workflow

### Step 1 – Data Loading

The dataset is validated to ensure all required fields are available.

### Step 2 – Data Preprocessing

- Missing value handling
- Text cleaning
- Profile generation

### Step 3 – Feature Engineering

A combined profile is created using:

- About Me
- Professional Summary
- Career Goal
- Skillset

---

### Step 4 – TF-IDF Vectorization

Each user profile is converted into numerical vectors using TF-IDF.

---

### Step 5 – KNN Recommendation

K-Nearest Neighbors identifies users with the most similar profiles.

---

### Step 6 – Hybrid Recommendation Score

The final recommendation score is calculated using:

| Component | Weight |
|-----------|--------|
| Profile Similarity (TF-IDF + KNN) | 60% |
| MBTI Compatibility | 25% |
| Location Match | 10% |
| Income Similarity | 5% |

---

# 📂 Dataset

The project uses a custom dataset containing over **100 user profiles**.

Each profile includes:

- User ID
- Name
- Location
- Skills
- Income
- MBTI Personality Type
- About Me
- Professional Summary
- Career Goal
- Experience

---

# 📸 Application Screenshots

## Home Page

<img width="959" height="537" alt="1 home" src="https://github.com/user-attachments/assets/65032f95-deac-4ea7-91f0-185dc637b940" />


---

## Selected User

<img width="959" height="538" alt="2 selected_user" src="https://github.com/user-attachments/assets/fefc5c29-0d5e-44f0-bb8a-884b6d4ac926" />


---

## Recommendations

<img width="959" height="539" alt="3 recommendations" src="https://github.com/user-attachments/assets/1ed374dd-d19c-43ff-9824-f95bfd455095" />


---

## MBTI Distribution

<img width="959" height="540" alt="4 mbti_distribution" src="https://github.com/user-attachments/assets/ea9e1af1-92fb-4ba4-bd29-f199d47856b4" />

---

# 📁 Project Structure

```
ProfileMatch-Intelligent-User-Recommendation-System/

│── dataset/
│     └── users.csv

│── screenshots/

│── app.py
│── users.py
│── requirements.txt
│── commands.txt
│── README.md
│── Project_Report.docx
│── LICENSE
```

---

# ⚙️ Installation

Clone the repository

```bash
git clone https://github.com/yourusername/ProfileMatch-Intelligent-User-Recommendation-System.git
```

Move into the project directory

```bash
cd ProfileMatch-Intelligent-User-Recommendation-System
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the application

```bash
streamlit run app.py
```

---

# 🚀 Usage

1. Launch the application.
2. Select a user from the sidebar.
3. Choose the number of recommendations.
4. Click **Find Best Matches**.
5. View the recommended users and similarity scores.

---

# 📊 Results

The application successfully generates personalized user recommendations by combining textual profile similarity with personality compatibility and additional profile attributes.

The hybrid recommendation approach produces more meaningful recommendations compared to relying on a single similarity metric.

---

# 🔮 Future Enhancements

- Real-time database integration
- User authentication
- Deep Learning–based recommendation models
- Sentence Transformer embeddings
- Collaborative filtering
- Deployment on Streamlit Cloud or AWS
- REST API integration

---

# 🙏 Acknowledgements

This project was developed as part of my B.Tech Major Project.

Special thanks to **Unlox Academy** and my project mentor Sruthi Tarimana for their guidance and support throughout the development of this project.

---

# 👨‍💻 Author

Aditya Raj
B.Tech Computer Science Student

Aspiring Machine Learning Engineer

GitHub: https://github.com/ADVIN-dev

LinkedIn: www.linkedin.com/in/aditya-raj-903b52349

---

⭐ If you found this project useful, consider giving it a star!

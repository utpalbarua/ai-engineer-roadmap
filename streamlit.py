import streamlit as st
import datetime
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import pandas as pd

# Page configuration with custom styling
st.set_page_config(
    page_title="Junior AI Engineer Quest | Your Path to Success", 
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🧠"
)

# Custom CSS for advanced styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
    }
    
    .quest-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 5px solid #667eea;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .achievement-badge {
        background: linear-gradient(45deg, #FFD700, #FFA500);
        color: #000;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
        margin: 0.25rem;
    }
    
    .progress-section {
        background: #f8f9ff;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    .skill-tag {
        background: #e3f2fd;
        color: #1976d2;
        padding: 0.25rem 0.75rem;
        border-radius: 15px;
        font-size: 0.8rem;
        margin: 0.25rem;
        display: inline-block;
    }
    
    .week-title {
        color: #2c3e50;
        font-size: 1.4rem;
        font-weight: 600;
        margin-bottom: 1rem;
    }
    
    .task-description {
        color: #5d6d7e;
        font-size: 0.9rem;
        margin-left: 1.5rem;
        font-style: italic;
    }
</style>
""", unsafe_allow_html=True)

# Main header with gradient background
st.markdown("""
<div class="main-header">
    <h1>🧠 Junior AI Engineer Quest</h1>
    <h3>🎯 Transform from Beginner to Industry-Ready AI Professional</h3>
    <p>A comprehensive 12-week journey to master machine learning, deep learning, and AI deployment</p>
</div>
""", unsafe_allow_html=True)

# Enhanced sections with detailed descriptions and skills
sections_data = {
    "Week 1-2: Machine Learning Foundations": {
        "icon": "📊",
        "color": "#3498db",
        "description": "Master the fundamental building blocks of machine learning with hands-on practice",
        "skills": ["Python", "Scikit-learn", "Pandas", "NumPy", "Matplotlib"],
        "quests": [
            {
                "title": "Supervised Learning Algorithms Mastery",
                "description": "Implement and understand Linear Regression, Logistic Regression, Decision Trees, and Support Vector Machines",
                "difficulty": "Beginner",
                "time": "8-10 hours"
            },
            {
                "title": "Model Evaluation & Metrics Deep Dive",
                "description": "Learn to evaluate models using Accuracy, Precision, Recall, F1-Score, ROC-AUC, and cross-validation techniques",
                "difficulty": "Beginner",
                "time": "6-8 hours"
            },
            {
                "title": "Titanic Survival Prediction Project",
                "description": "Build an end-to-end ML pipeline for the classic Titanic dataset with feature engineering and model comparison",
                "difficulty": "Intermediate",
                "time": "12-15 hours"
            }
        ]
    },
    "Week 3-4: Unsupervised Learning & Data Analysis": {
        "icon": "🔍",
        "color": "#e74c3c",
        "description": "Discover hidden patterns in data without labeled examples",
        "skills": ["K-Means", "DBSCAN", "PCA", "t-SNE", "Data Visualization"],
        "quests": [
            {
                "title": "Clustering Algorithms Implementation",
                "description": "Master K-Means, DBSCAN, and Hierarchical clustering with real-world datasets",
                "difficulty": "Intermediate",
                "time": "10-12 hours"
            },
            {
                "title": "Dimensionality Reduction Techniques",
                "description": "Apply PCA, t-SNE, and UMAP for data visualization and feature reduction",
                "difficulty": "Intermediate",
                "time": "8-10 hours"
            },
            {
                "title": "Customer Segmentation Analytics Dashboard",
                "description": "Create an interactive dashboard for customer behavior analysis and market segmentation",
                "difficulty": "Advanced",
                "time": "15-18 hours"
            }
        ]
    },
    "Week 5-6: Deep Learning Neural Networks": {
        "icon": "🧠",
        "color": "#9b59b6",
        "description": "Build and train neural networks from scratch using modern frameworks",
        "skills": ["TensorFlow", "Keras", "PyTorch", "Neural Networks", "Backpropagation"],
        "quests": [
            {
                "title": "Neural Network Architecture Design",
                "description": "Build feedforward neural networks using Keras/TensorFlow with custom architectures",
                "difficulty": "Intermediate",
                "time": "12-14 hours"
            },
            {
                "title": "Advanced Training Techniques",
                "description": "Implement regularization, dropout, batch normalization, and learning rate scheduling",
                "difficulty": "Advanced",
                "time": "10-12 hours"
            },
            {
                "title": "Fashion MNIST Classification Challenge",
                "description": "Build a high-accuracy fashion item classifier with data augmentation and hyperparameter tuning",
                "difficulty": "Advanced",
                "time": "15-20 hours"
            }
        ]
    },
    "Week 7-8: Computer Vision Mastery": {
        "icon": "👁️",
        "color": "#f39c12",
        "description": "Process and analyze visual data using convolutional neural networks",
        "skills": ["CNNs", "OpenCV", "Transfer Learning", "Image Processing", "ResNet"],
        "quests": [
            {
                "title": "Convolutional Neural Networks & Data Augmentation",
                "description": "Design CNN architectures and implement advanced data augmentation techniques",
                "difficulty": "Advanced",
                "time": "14-16 hours"
            },
            {
                "title": "Transfer Learning with Pre-trained Models",
                "description": "Fine-tune ResNet50, VGG, and other pre-trained models for custom image classification",
                "difficulty": "Advanced",
                "time": "12-14 hours"
            },
            {
                "title": "Real-world Computer Vision Application",
                "description": "Build a face mask detection system or waste classification app with deployment-ready code",
                "difficulty": "Expert",
                "time": "20-25 hours"
            }
        ]
    },
    "Week 9-10: Natural Language Processing": {
        "icon": "💬",
        "color": "#1abc9c",
        "description": "Process and understand human language using cutting-edge NLP techniques",
        "skills": ["NLTK", "spaCy", "Transformers", "BERT", "Word Embeddings"],
        "quests": [
            {
                "title": "Text Processing & Feature Extraction",
                "description": "Master TF-IDF, Word2Vec, and GloVe embeddings for text representation",
                "difficulty": "Intermediate",
                "time": "10-12 hours"
            },
            {
                "title": "Advanced NLP with Transformers",
                "description": "Implement LSTM networks and work with BERT for text classification and analysis",
                "difficulty": "Advanced",
                "time": "14-16 hours"
            },
            {
                "title": "Interactive Sentiment Analysis Dashboard",
                "description": "Create a real-time tweet sentiment analyzer with visualization and trend analysis",
                "difficulty": "Expert",
                "time": "18-22 hours"
            }
        ]
    },
    "Week 11-12: MLOps & Production Deployment": {
        "icon": "🚀",
        "color": "#34495e",
        "description": "Deploy ML models to production with industry-standard practices",
        "skills": ["Flask", "FastAPI", "Docker", "AWS", "MLOps", "CI/CD"],
        "quests": [
            {
                "title": "API Development for ML Models",
                "description": "Create robust APIs using Flask and FastAPI with proper error handling and documentation",
                "difficulty": "Advanced",
                "time": "12-15 hours"
            },
            {
                "title": "Cloud Deployment & Monitoring",
                "description": "Deploy models to Render, Heroku, or HuggingFace Spaces with monitoring and logging",
                "difficulty": "Advanced",
                "time": "10-12 hours"
            },
            {
                "title": "End-to-End ML Application Portfolio",
                "description": "Build a complete ML application with frontend, backend, database, and deployment pipeline",
                "difficulty": "Expert",
                "time": "25-30 hours"
            }
        ]
    }
}

# Calculate total tasks
total_tasks = sum(len(section_data["quests"]) for section_data in sections_data.values())

# Session state setup
if "progress" not in st.session_state:
    st.session_state.progress = [False] * total_tasks
if "start_date" not in st.session_state:
    st.session_state.start_date = datetime.now()

# Sidebar - Enhanced Progress Tracker
st.sidebar.markdown("## 📊 Quest Progress Dashboard")

completed_tasks = sum(st.session_state.progress)
progress_pct = (completed_tasks / total_tasks) * 100

# Progress visualization
fig_progress = go.Figure(go.Indicator(
    mode = "gauge+number+delta",
    value = progress_pct,
    domain = {'x': [0, 1], 'y': [0, 1]},
    title = {'text': "Completion %"},
    delta = {'reference': 0},
    gauge = {
        'axis': {'range': [None, 100]},
        'bar': {'color': "#667eea"},
        'steps': [
            {'range': [0, 25], 'color': "#ffebee"},
            {'range': [25, 50], 'color': "#fff3e0"},
            {'range': [50, 75], 'color': "#f3e5f5"},
            {'range': [75, 100], 'color': "#e8f5e8"}
        ],
        'threshold': {
            'line': {'color': "red", 'width': 4},
            'thickness': 0.75,
            'value': 90
        }
    }
))
fig_progress.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))
st.sidebar.plotly_chart(fig_progress, use_container_width=True)

# Achievement badges
st.sidebar.markdown("### 🏆 Achievement Badges")
if progress_pct >= 25:
    st.sidebar.markdown('<div class="achievement-badge">🚀 Getting Started</div>', unsafe_allow_html=True)
if progress_pct >= 50:
    st.sidebar.markdown('<div class="achievement-badge">📈 Halfway Hero</div>', unsafe_allow_html=True)
if progress_pct >= 75:
    st.sidebar.markdown('<div class="achievement-badge">🔥 Almost There</div>', unsafe_allow_html=True)
if progress_pct >= 90:
    st.sidebar.markdown('<div class="achievement-badge">🎓 AI Engineer</div>', unsafe_allow_html=True)

# Time tracking
weeks_elapsed = (datetime.now() - st.session_state.start_date).days // 7
st.sidebar.markdown(f"### ⏱️ Journey Timeline")
st.sidebar.markdown(f"**Weeks Elapsed:** {weeks_elapsed}")
st.sidebar.markdown(f"**Tasks Completed:** {completed_tasks}/{total_tasks}")

# Estimated completion time
if completed_tasks > 0:
    avg_time_per_task = weeks_elapsed / completed_tasks
    remaining_tasks = total_tasks - completed_tasks
    estimated_weeks_left = int(remaining_tasks * avg_time_per_task)
    st.sidebar.markdown(f"**Estimated Completion:** {estimated_weeks_left} weeks")

# Main content - Enhanced quest display
col1, col2 = st.columns([3, 1])

with col1:
    st.markdown("## 🗺️ Your Learning Journey")
    
    task_idx = 0
    for section_name, section_data in sections_data.items():
        with st.expander(f"{section_data['icon']} {section_name}", expanded=False):
            # Section header with description
            st.markdown(f"""
            <div class="quest-card">
                <div class="week-title">{section_name}</div>
                <p>{section_data['description']}</p>
                <div>
                    <strong>Key Skills:</strong><br>
                    {''.join([f'<span class="skill-tag">{skill}</span>' for skill in section_data['skills']])}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Quest tasks
            for quest in section_data["quests"]:
                col_check, col_content = st.columns([1, 10])
                
                with col_check:
                    st.session_state.progress[task_idx] = st.checkbox(
                        "", 
                        value=st.session_state.progress[task_idx], 
                        key=f"task_{task_idx}"
                    )
                
                with col_content:
                    # Task status indicator
                    status = "✅ Completed" if st.session_state.progress[task_idx] else "⏳ Pending"
                    difficulty_color = {
                        "Beginner": "#2ecc71",
                        "Intermediate": "#f39c12", 
                        "Advanced": "#e74c3c",
                        "Expert": "#8e44ad"
                    }
                    
                    st.markdown(f"""
                    **{quest['title']}** 
                    <span style="color: {difficulty_color[quest['difficulty']]}; font-weight: bold;">
                        [{quest['difficulty']}]
                    </span> 
                    <span style="color: #7f8c8d;">• {quest['time']}</span> 
                    <span style="float: right;">{status}</span>
                    """, unsafe_allow_html=True)
                    
                    st.markdown(f'<div class="task-description">{quest["description"]}</div>', 
                              unsafe_allow_html=True)
                    st.markdown("---")
                
                task_idx += 1

with col2:
    st.markdown("## 📈 Progress Analytics")
    
    # Section progress breakdown
    section_progress = []
    task_idx = 0
    for section_name, section_data in sections_data.items():
        section_completed = sum(st.session_state.progress[task_idx:task_idx + len(section_data["quests"])])
        section_total = len(section_data["quests"])
        section_progress.append({
            "Section": section_name.split(":")[0],
            "Completed": section_completed,
            "Total": section_total,
            "Percentage": (section_completed / section_total) * 100
        })
        task_idx += len(section_data["quests"])
    
    df_progress = pd.DataFrame(section_progress)
    
    # Progress bar chart
    fig_bar = px.bar(
        df_progress, 
        x="Percentage", 
        y="Section",
        orientation='h',
        title="Progress by Section",
        color="Percentage",
        color_continuous_scale="Viridis"
    )
    fig_bar.update_layout(height=400, margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig_bar, use_container_width=True)
    
    # Weekly goal tracker
    st.markdown("### 🎯 Weekly Goals")
    if weeks_elapsed < 12:
        current_week_section = list(sections_data.keys())[min(weeks_elapsed // 2, 5)]
        st.info(f"**Current Focus:** {current_week_section}")
    else:
        st.success("🎉 Congratulations! Journey Complete!")

# Footer with enhanced styling
st.markdown("---")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    ### 🚀 Next Steps
    - Join AI communities
    - Build portfolio projects  
    - Apply for junior roles
    """)

with col2:
    st.markdown("""
    ### 📚 Resources
    - Kaggle competitions
    - GitHub repositories
    - Technical blogs
    """)

with col3:
    st.markdown("""
    ### 🤝 Connect
    - LinkedIn networking
    - Tech meetups
    - Open source contributions
    """)

st.markdown("""
<div style="text-align: center; margin-top: 2rem; padding: 1rem; background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); border-radius: 10px; color: white;">
    <h3>🎓 Build. Learn. Deploy. Become a Junior AI Engineer.</h3>
    <p>Your journey to becoming an industry-ready AI professional starts here!</p>
</div>
""", unsafe_allow_html=True)

st.caption("🧠 Advanced AI Learning Dashboard | Crafted for your success journey")

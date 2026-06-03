from flask import Flask, render_template, request
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# --------------------------
# Sample Dataset
# --------------------------
projects = {
    "Website Development": "HTML CSS JavaScript React Node",
    "Data Dashboard": "Python SQL PowerBI DataVisualization",
    "Logo Design": "Photoshop Illustrator Creativity Branding",
    "Mobile App UI": "Figma UXDesign Creativity Mobile",
    "Cybersecurity Audit": "Networking Security Linux PenTesting",
    "Machine Learning Model": "Python ScikitLearn TensorFlow DataPreprocessing",
    "Chatbot Development": "Python NLP AI Flask API",
    "IoT Home Automation": "Arduino RaspberryPi Sensors MQTT",
    "Blockchain Application": "Solidity Ethereum SmartContracts Web3",
    "Game Development": "Unity C# 3DAnimation Physics",
    "Data Mining Project": "Python Pandas SQL DataAnalysis",
    "Cloud Deployment": "AWS Docker Kubernetes DevOps",
    "Robotics Simulation": "ROS Python C++ Sensors Actuators",
    "Augmented Reality App": "ARKit Unity C# Mobile",
    "Computer Vision System": "OpenCV Python ImageProcessing",
    "Network Traffic Analyzer": "Python Wireshark Networking Security",
    "Embedded Systems Project": "C Microcontroller RTOS Sensors",
    "Voice Assistant": "Python SpeechRecognition NLP AI",
    "E-commerce Backend": "Node Express MongoDB RESTAPI",
    "DevOps Pipeline": "Jenkins Git Docker CI/CD"
}

jobs = {
    "Frontend Developer": "HTML CSS JavaScript React ResponsiveDesign",
    "Data Analyst": "Python Excel SQL Tableau Analytics",
    "Graphic Designer": "Illustrator Photoshop Creativity Design",
    "Android Developer": "Kotlin Java Android Firebase",
    "Network Engineer": "Cisco Networking Security Troubleshooting",
    "Backend Developer": "Python Java NodeJS SQL APIs",
    "DevOps Engineer": "Docker Kubernetes CI/CD AWS Linux",
    "Machine Learning Engineer": "Python TensorFlow PyTorch DataModeling",
    "Cybersecurity Analyst": "NetworkSecurity EthicalHacking RiskAssessment Firewalls",
    "Cloud Engineer": "AWS Azure GCP CloudArchitecture Automation",
    "Full Stack Developer": "HTML CSS JavaScript NodeJS React SQL",
    "Database Administrator": "SQL NoSQL PerformanceTuning BackupRecovery",
    "Embedded Systems Engineer": "C C++ Microcontrollers RTOS IoT",
    "Software Tester": "Selenium Automation Testing QA Debugging",
    "UI/UX Designer": "Figma AdobeXD Wireframing Prototyping",
    "Blockchain Developer": "Solidity Ethereum SmartContracts Cryptography",
    "Systems Administrator": "Linux Windows Scripting Monitoring Security",
    "AI Engineer": "Python NLP ComputerVision DeepLearning",
    "Game Developer": "C++ Unity Unreal3D Graphics Physics",
    "Robotics Engineer": "ROS Python Sensors Actuators ControlSystems"
}

# --------------------------
# Courses
# --------------------------
course_sources = {
    "Python": ["https://www.coursera.org/learn/python", "https://www.learnpython.org/"],
    "Data Structures": ["https://www.geeksforgeeks.org/data-structures/", "https://www.coursera.org/specializations/data-structures-algorithms"],
    "Web Development": ["https://www.freecodecamp.org/", "https://www.codecademy.com/learn/paths/web-development"],
    "Machine Learning": ["https://www.coursera.org/learn/machine-learning", "https://www.kaggle.com/learn/intro-to-machine-learning"],
    "HTML": ["https://www.w3schools.com/html/", "https://developer.mozilla.org/en-US/docs/Web/HTML"],
    "CSS": ["https://www.w3schools.com/css/", "https://developer.mozilla.org/en-US/docs/Web/CSS"],
    "JavaScript": ["https://www.javascript.com/", "https://developer.mozilla.org/en-US/docs/Web/JavaScript"],
    "SQL": ["https://www.w3schools.com/sql/", "https://www.khanacademy.org/computing/computer-programming/sql"],
    "Excel": ["https://support.microsoft.com/en-us/excel", "https://www.excel-easy.com/"]
}

# --------------------------
# Cosine Similarity Logic
# --------------------------
def get_recommendations(user_input, dataset):
    names = list(dataset.keys())
    corpus = [user_input] + list(dataset.values())

    vectorizer = CountVectorizer()
    vectors = vectorizer.fit_transform(corpus)

    cosine_scores = cosine_similarity(vectors[0], vectors[1:])[0]

    user_skill_set = set(s.strip().title() for s in user_input.split(','))

    results = []
    for i, name in enumerate(names):
        req_skills = set(s.strip().title() for s in dataset[name].split())
        missing = req_skills - user_skill_set
        matched = user_skill_set & req_skills

        results.append({
            "name": name,
            "score": float(cosine_scores[i]),  # STRICTLY 0–1
            "missing": [s for s in missing if s in course_sources],
            "matched": list(matched)
        })

    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:3]

# --------------------------
# Routes
# --------------------------
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    user_skills = request.form['skills']

    project_results = get_recommendations(user_skills, projects)
    job_results = get_recommendations(user_skills, jobs)

    all_missing = set()
    for r in project_results + job_results:
        all_missing.update(r["missing"])

    courses = {c: course_sources[c] for c in all_missing}

    return render_template(
        'result.html',
        project_results=project_results,
        job_results=job_results,
        skills=user_skills,
        courses_with_sources=courses
    )

if __name__ == "__main__":
    app.run(debug=True)

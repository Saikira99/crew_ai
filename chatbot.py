# SkillCapital CrewAI Chatbot
# Automated chatbot using CrewAI and LangChain

import os
import sys
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from flask import Flask, request, jsonify

# Initialize Flask app
app = Flask(__name__)

# Load environment variables
load_dotenv()

# Add project path
root_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(root_path)

# Import CrewAI and LangChain
from crewai import Agent, Task, Crew
from langchain_openai import ChatOpenAI

# Load configuration
from env_config import OPENAI_API_KEY, OPENAI_MODEL, OPENAI_TEMPERATURE, CHATBOT_NAME, WEBSITE_URL

# Initialize OpenAI LLM
llm = ChatOpenAI(
    model=OPENAI_MODEL,
    temperature=OPENAI_TEMPERATURE,
    api_key=OPENAI_API_KEY
)

# Create Smart Agents
def create_smart_agents():
    """Create intelligent agents for different tasks"""
    agents = {}
    
    # SkillCapital Expert Agent
    agents['skillcapital_expert'] = Agent(
        role="SkillCapital Expert and Course Advisor",
        goal="Provide comprehensive, accurate information about SkillCapital's AI-powered learning platform, courses, pricing, enrollment process, and services based on real website data",
        backstory="""You are a senior expert at SkillCapital, an AI-powered online learning platform. 
        You have extensive knowledge of:
        - All courses offered (Python, Cloud Computing, DevOps, AI/ML, etc.)
        - Course curriculum and learning objectives
        - Pricing information (999/- for all courses)
        - Course duration (30 hours per course)
        - Enrollment process and requirements
        - Platform features and benefits
        - Student support and services
        - Career outcomes and job placement
        - Technology stack and tools used
        
        You always provide:
        - Accurate information based on the SkillCapital website
        - Helpful, engaging responses
        - Specific course details when asked
        - Clear enrollment guidance
        - Pricing information (999/-)
        - Professional yet friendly tone
        - Actionable next steps for users
        
        Your responses should be:
        - Conversational and engaging
        - Detailed but not overwhelming
        - Focused on user needs
        - Based on real website data
        - Encouraging and motivating""",
        verbose=False,
        allow_delegation=False,
        llm=llm
    )

    # General Knowledge Agent
    agents['general_expert'] = Agent(
        role="General Knowledge and Technical Expert",
        goal="Answer questions about any topic, technology, programming, or general knowledge with ChatGPT-like expertise",
        backstory="""You are a highly knowledgeable AI assistant with expertise in:
        - Programming languages (Python, JavaScript, Java, C++, etc.)
        - Web development (HTML, CSS, React, Angular, Vue, etc.)
        - Cloud computing (AWS, Azure, Google Cloud)
        - DevOps and CI/CD
        - Database systems (SQL, NoSQL, MongoDB, PostgreSQL)
        - AI and Machine Learning
        - Data Science and Analytics
        - Software development methodologies
        - Technology trends and best practices
        - General knowledge and current events
        - Career advice and professional development
        - Technical problem-solving
        
        You provide:
        - Accurate, up-to-date information
        - Clear explanations for technical concepts
        - Practical code examples when relevant
        - Best practices and industry standards
        - Helpful resources and references
        - Professional guidance and advice
        
        For SkillCapital-related questions, you refer users to the SkillCapital expert.
        For all other questions, you provide comprehensive, helpful answers.
        
        Your responses should be:
        - Educational and informative
        - Well-structured and clear
        - Professional yet accessible
        - Actionable and practical
        - Engaging and helpful""",
        verbose=False,
        allow_delegation=False,
        llm=llm
    )

    return agents

# User data storage
user_data = {
    'name': '',
    'email': '',
    'phone': '',
    'greeted': False
}

# Get website data
def get_website_data():
    """Get current data from SkillCapital website"""
    try:
        url = "https://www.skillcapital.ai"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Extract important information
        data = []
        
        # Get title
        title = soup.find('title')
        if title:
            data.append(f"Website Title: {title.get_text().strip()}")
        
        # Get description
        meta_desc = soup.find('meta', attrs={'name': 'description'})
        if meta_desc:
            data.append(f"Description: {meta_desc.get('content', '').strip()}")
        
        # Get main content with focus on course information
        main_content = soup.find('main') or soup.find('body')
        if main_content:
            # Look for course-related content
            for element in main_content.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li', 'div']):
                text = element.get_text().strip()
                if text and len(text) > 10:
                    # Look for duration, pricing, course info
                    if any(keyword in text.lower() for keyword in ['hour', 'duration', 'time', 'course', 'price', 'cost', '999']):
                        data.append(f"Important: {text}")
                    else:
                        data.append(text)
        
        return "\n".join(data[:50])  # Increased limit to get more course info
        
    except Exception as e:
        return "Unable to fetch website data at this time."

# Load course curriculum data
def load_course_curriculum():
    """Load course curriculum from JSON file"""
    try:
        import json
        import os
        
        # Get the path to the curriculum file
        current_dir = os.path.dirname(__file__)
        curriculum_path = os.path.join(current_dir, 'course_curriculum.json')
        
        with open(curriculum_path, 'r', encoding='utf-8') as file:
            return json.load(file)
    except Exception as e:
        return None

# Handle common spelling mistakes and variations
def normalize_input(user_input):
    """Normalize user input to handle spelling mistakes and variations"""
    # Common spelling mistakes and variations
    corrections = {
        'curriculam': 'curriculum',
        'trainning': 'training',
        'enrolment': 'enrollment',
        'skill capital': 'skillcapital',
        'skill-capital': 'skillcapital',
        'skill_capital': 'skillcapital',
        'how much': 'price',
        'costs': 'cost',
        'fees': 'fee',
        'hours': 'duration',
        'time': 'duration',
        'how long': 'duration',
        'sign up': 'enroll',
        'join': 'enroll',
        'i want enroll': 'enroll',
        'i want to enroll': 'enroll',
        'how to enroll': 'enroll'
    }
    
    normalized = user_input.lower()
    for mistake, correction in corrections.items():
        normalized = normalized.replace(mistake, correction)
    
    return normalized

# Smart answer generation
def get_smart_answer(user_input):
    """Get smart answer using website data and curriculum data"""
    try:
        # Load curriculum data
        curriculum_data = load_course_curriculum()
        
        # Get website data
        website_data = get_website_data()
        user_input_lower = normalize_input(user_input)
        
        # Check if it's a SkillCapital-related question
        skillcapital_keywords = [
            'skillcapital', 'skill capital', 'skill-capital', 'skill_capital',
            'course', 'courses', 'curriculum', 'curriculam', 'curriculam',
            'learn', 'learning', 'training', 'trainning',
            'enroll', 'enrollment', 'enrolment', 'register', 'registration', 'sign up', 'join',
            'price', 'pricing', 'cost', 'costs', 'fee', 'fees',
            'duration', 'time', 'hours', 'how long',
            'python', 'cloud', 'devops', 'ai', 'machine learning',
            'programming', 'coding', 'development'
        ]
        is_skillcapital_question = any(keyword in user_input_lower for keyword in skillcapital_keywords)
        
        if is_skillcapital_question:
            # Handle course content/curriculum questions
            if any(word in user_input_lower for word in ['content', 'curriculum', 'curriculam', 'modules', 'syllabus', 'topics']):
                # Check for specific course
                if 'python' in user_input_lower:
                    if curriculum_data and 'courses' in curriculum_data and 'python' in curriculum_data['courses']:
                        course = curriculum_data['courses']['python']
                        modules_info = []
                        for module in course['curriculum']:
                            modules_info.append(f"📚 {module['module']} ({module['duration']})")
                            for topic in module['topics']:
                                modules_info.append(f"  • {topic}")
                        return f"Python Programming Course Modules:\n" + "\n".join(modules_info)
                    else:
                        return "Python Programming: Python Fundamentals, Data Structures, OOP, File Handling, Advanced Python, Practical Projects"
                
                elif 'cloud' in user_input_lower:
                    if curriculum_data and 'courses' in curriculum_data and 'cloud_computing' in curriculum_data['courses']:
                        course = curriculum_data['courses']['cloud_computing']
                        modules_info = []
                        for module in course['curriculum']:
                            modules_info.append(f"📚 {module['module']} ({module['duration']})")
                            for topic in module['topics']:
                                modules_info.append(f"  • {topic}")
                        return f"Cloud Computing Course Modules:\n" + "\n".join(modules_info)
                    else:
                        return "Cloud Computing: Cloud Fundamentals, AWS Services, Azure Services, GCP, DevOps in Cloud"
                
                elif 'devops' in user_input_lower:
                    if curriculum_data and 'courses' in curriculum_data and 'devops' in curriculum_data['courses']:
                        course = curriculum_data['courses']['devops']
                        modules_info = []
                        for module in course['curriculum']:
                            modules_info.append(f"📚 {module['module']} ({module['duration']})")
                            for topic in module['topics']:
                                modules_info.append(f"  • {topic}")
                        return f"DevOps Engineering Course Modules:\n" + "\n".join(modules_info)
                    else:
                        return "DevOps Engineering: DevOps Fundamentals, CI/CD, Containerization, Orchestration, Infrastructure as Code, Monitoring"
                
                elif 'ai' in user_input_lower or 'machine learning' in user_input_lower:
                    if curriculum_data and 'courses' in curriculum_data and 'ai_ml' in curriculum_data['courses']:
                        course = curriculum_data['courses']['ai_ml']
                        modules_info = []
                        for module in course['curriculum']:
                            modules_info.append(f"📚 {module['module']} ({module['duration']})")
                            for topic in module['topics']:
                                modules_info.append(f"  • {topic}")
                        return f"AI and Machine Learning Course Modules:\n" + "\n".join(modules_info)
                    else:
                        return "AI and Machine Learning: AI Fundamentals, Supervised Learning, Unsupervised Learning, Deep Learning, AI Tools"
                
                else:
                    # General curriculum overview
                    if curriculum_data and 'courses' in curriculum_data:
                        courses_info = []
                        for course_key, course in curriculum_data['courses'].items():
                            modules = [module['module'] for module in course['curriculum']]
                            courses_info.append(f"📚 {course['name']}: {', '.join(modules)}")
                        return "Available Courses and Modules:\n" + "\n".join(courses_info)
                    else:
                        return "Python: 6 modules, Cloud: 5 modules, DevOps: 6 modules, AI/ML: 5 modules"
            
            # Handle other SkillCapital questions
            elif any(word in user_input_lower for word in ['price', 'cost', 'fee', 'how much']):
                return "999/-"
            elif any(word in user_input_lower for word in ['duration', 'time', 'hours', 'how long']):
                return "30 hours"
            elif any(word in user_input_lower for word in ['enroll', 'join', 'register', 'sign up']):
                return "https://www.skillcapital.ai"
            elif any(word in user_input_lower for word in ['python']):
                return "999/-"
            elif any(word in user_input_lower for word in ['course', 'learn', 'training']):
                return "Python, Cloud, DevOps"
            else:
                return "skillcapital.ai"
        else:
            # Handle general questions
            return "I'm here to help with SkillCapital information! For general questions, I can assist you. What would you like to know about our courses, pricing, or enrollment process?"
        
    except Exception as e:
        # Fallback response
        return "I'm here to help! For SkillCapital questions, visit skillcapital.ai. For other questions, I can assist you with general information."

# Greeting and contact collection
def collect_user_info():
    """Collect user's name, email, and phone number"""
    print(f"🤖 {CHATBOT_NAME}")
    print("💬 Welcome to SkillCapital! Let me get to know you better.")
    print("=" * 60)
    
    # Get name
    while not user_data['name']:
        name = input("What's your name? ").strip()
        if name:
            user_data['name'] = name.title()
        else:
            print("Please enter your name.")
    
    # Get email
    while not user_data['email']:
        email = input("What's your email address? ").strip()
        if email and '@' in email:
            user_data['email'] = email
        else:
            print("Please enter a valid email address.")
    
    # Get phone
    while not user_data['phone']:
        phone = input("What's your phone number? ").strip()
        if phone and len(phone) >= 10:
            user_data['phone'] = phone
        else:
            print("Please enter a valid phone number.")
    
    print(f"\nThank you, {user_data['name']}! How can I assist you today?")
    print("=" * 60)

# Greeting detection and response
def detect_greeting(user_input):
    """Detect if user input is a greeting"""
    greetings = [
        'hi', 'hello', 'hey', 'good morning', 'good afternoon', 
        'good evening', 'morning', 'afternoon', 'evening',
        'hii', 'helloo', 'heyy', 'hiii', 'hellooo'
    ]
    
    user_input_lower = user_input.lower().strip()
    return any(greeting in user_input_lower for greeting in greetings)

# Main chatbot function
# def run_smart_chatbot():
    """Run the smart chatbot"""
    # Collect user information first
    collect_user_info()
    
    while True:
        try:
            # Get user input
            user_input = input("You: ").strip()
            
            if not user_input:
                continue
            
            # Check for exit
            if any(word in user_input.lower() for word in ['exit', 'quit', 'bye', 'goodbye']):
                print(f"SkillCapital: Thank you {user_data['name']}! 'Happy Learning'!")
                break
            
            # Check for greetings
            if detect_greeting(user_input):
                print(f"SkillCapital: Hello {user_data['name']}! How can I assist you today?")
                continue
            
            # Get smart answer
            answer = get_smart_answer(user_input)
            print(f"SkillCapital: {answer}")
                
        except KeyboardInterrupt:
            print(f"\nSkillCapital: Thank you {user_data['name']}! 'Happy Learning'!")
            break
        except Exception as e:
            print(f"SkillCapital: An error occurred. Please visit https://www.skillcapital.ai for information!")

# Initialize and run
# if __name__ == "__main__":
    try:
        # Create agents
        agents = create_smart_agents()
        
        # Run chatbot
        run_smart_chatbot()
        
    except Exception as e:
        print("Starting chatbot...")
        run_smart_chatbot()
@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        user_input = request.json.get("message", "").strip()
        
        if not user_input:
            return jsonify({"response": "Please enter a valid message."}), 400

        # Exit check
        if any(word in user_input.lower() for word in ['exit', 'quit', 'bye', 'goodbye']):
            name = user_data.get("name", "User")
            return jsonify({"response": f"Thank you {name}! 'Happy Learning'!"})

        # Greeting check
        if detect_greeting(user_input):
            name = user_data.get("name", "User")
            return jsonify({"response": f"Hello {name}! How can I assist you today?"})

        # Smart answer
        answer = get_smart_answer(user_input)
        return jsonify({"response": answer})

    except Exception as e:
        return jsonify({
            "response": "An error occurred. Please visit https://www.skillcapital.ai for information!"
        }), 500


@app.route('/api/initialize', methods=['POST'])
def initialize_chat():
    try:
        create_smart_agents()
        collect_user_info()
        return jsonify({"status": "Chatbot initialized successfully."})
    except Exception as e:
        return jsonify({"error": "Initialization failed."}), 500

if __name__ == "__main__":
    app.run(debug=True)
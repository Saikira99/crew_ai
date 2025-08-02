import os
import json
import requests
import traceback

from flasgger import Swagger
from flask import Flask, request, jsonify
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, CrewOutput
from langchain_openai import ChatOpenAI

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Setup Flask app
app = Flask(__name__)
swagger = Swagger(app)

# ========== Website Scraper ==========
def get_website_data():
    try:
        url = "https://www.skillcapital.ai"
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')

        data = []
        title = soup.find('title')
        if title:
            data.append(f"Website Title: {title.get_text().strip()}")

        desc = soup.find('meta', attrs={'name': 'description'})
        if desc:
            data.append(f"Description: {desc.get('content', '').strip()}")

        main = soup.find('main') or soup.find('body')
        if main:
            for tag in main.find_all(['p', 'li', 'div', 'h1', 'h2', 'h3']):
                text = tag.get_text().strip()
                if text and len(text) > 20:
                    data.append(text)

        return "\n".join(data[:50])
    except Exception:
        return "No website data available."

# ========== Curriculum Loader ==========
def load_course_curriculum():
    try:
        curriculum_path = os.path.join(os.path.dirname(__file__), 'course_curriculum.json')
        with open(curriculum_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            print("✅ Loaded curriculum:", type(data), "Keys:", list(data.keys()) if isinstance(data, dict) else "Not a dict")
            return data
    except Exception as e:
        print(f"❌ Failed to load curriculum: {e}")
        return {}

# ========== LLM Setup ==========
llm = ChatOpenAI(model="gpt-4", temperature=0.4, api_key=OPENAI_API_KEY)

# ========== Agents ==========
course_expert = Agent(
    role="Course Expert",
    goal="Provide clear and structured answers about SkillCapital's course content and structure",
    backstory="Expert in educational program design and technical upskilling paths",
    verbose=True,
    llm=llm
)

pricing_assistant = Agent(
    role="Pricing Assistant",
    goal="Help users understand SkillCapital's pricing, value, and affordability",
    backstory="Specialist in explaining course pricing models and offers in detail",
    verbose=True,
    llm=llm
)

summary_generator = Agent(
    role="Summary Generator",
    goal="Generate a final natural-language response combining course and pricing responses",
    backstory="Expert in user communication and formatting answers for conversational delivery",
    verbose=True,
    llm=llm
)

# ========== Task Builder ==========
def get_tasks(user_question: str, context: str, matched_courses):
    courses_str = json.dumps(matched_courses, indent=2)

    return [
        Task(
            description=f"""You are the Course Expert. A user asked: '{user_question}'.
Use this context to answer with course insights:
{context}""",
            agent=course_expert,
            expected_output="Explain course content, modules, duration, and outcomes."
        ),
        Task(
            description=f"""You are the Pricing Assistant. A user asked: '{user_question}'.
Use this context to explain pricing, payment plans, or say 'No pricing info needed.' if irrelevant:
{context}""",
            agent=pricing_assistant,
            expected_output="Give clear pricing, offers, or respond 'not requested'."
        ),
        Task(
            description=f"""You are the Summary Generator. Combine the answers from Course Expert and Pricing Assistant to create a final response.

Use this course data:
{courses_str}

Ensure the response is warm, friendly, and includes both course and pricing insights for: "{user_question}".""",
            agent=summary_generator,
            expected_output="Single final summary combining course + pricing info."
        )
    ]

# ========== Response Formatter ==========
def summarize_response(user_question, matched_courses, tasks):
    cleaned_courses = []
    for i, c in enumerate(matched_courses):
        if isinstance(c, dict):
            cleaned_courses.append({
                "name": c.get("name", "N/A"),
                "duration": c.get("duration", "N/A"),
                "modules": c.get("modules", "N/A"),
                "outcome": c.get("outcome", "N/A")
            })

    return {
        "title": "🎓 SkillCapital Smart Assistant",
        "question": user_question,
        "matched_courses": cleaned_courses,
        "agents": [
            {"agent": "Course Expert", "response": str(tasks[0].output) if tasks[0].output else "No course expert output."},
            {"agent": "Pricing Assistant", "response": str(tasks[1].output) if tasks[1].output else "No pricing output."},
            {"agent": "Summary Generator", "response": str(tasks[2].output) if tasks[2].output else "No summary output."}
        ],
        "final_summary": str(tasks[2].output) if tasks[2].output else "Summary not available."
    }

# ========== Crew Runner ==========
def run_chatbot_crew(user_input: str, context_data: str, matched_courses):
    tasks = get_tasks(user_input, context_data, matched_courses)
    crew = Crew(
        agents=[task.agent for task in tasks],
        tasks=tasks,
        verbose=False
    )
    result: CrewOutput = crew.kickoff()

    # Optional debug
    print("\n✅ Agent Outputs:")
    for task in tasks:
        print(f"\n🔹 {task.agent.role} Output:\n{task.output}\n")

    return summarize_response(user_input, matched_courses, tasks)

# ========== Match Relevant Courses ==========
def find_relevant_courses(question, curriculum_data):
    matched = []
    q = question.lower()

    courses = []
    if isinstance(curriculum_data, dict) and "courses" in curriculum_data:
        raw_courses = curriculum_data["courses"]
        if isinstance(raw_courses, dict):
            courses = list(raw_courses.values())
        elif isinstance(raw_courses, list):
            courses = raw_courses

    for course in courses:
        if isinstance(course, dict):
            name = course.get("name", "").lower()
            if name and name in q:
                matched.append(course)

    return matched if matched else courses

# ========== Flask API ==========
@app.route('/smart-chatbot', methods=['POST'])
def smart_chatbot():
    """
    Smart Chatbot API
    ---
    post:
      summary: Get intelligent chatbot response based on user input and course curriculum
      description: >
        This endpoint receives a user message, analyzes it using website data and curriculum context,
        and returns a contextual chatbot response using CrewAI or similar backend service.
      requestBody:
        required: true
        content:
          application/json:
            schema:
              type: object
              properties:
                message:
                  type: string
                  example: "I'm looking for courses on data science"
                  description: User's query or message to the chatbot
      responses:
        200:
          description: Chatbot response based on context and user message
          content:
            application/json:
              schema:
                type: object
                properties:
                  response:
                    type: string
                    example: "Based on your interest, you may like our Data Science Bootcamp..."
        400:
          description: Bad request - message is missing
        500:
          description: Internal server error
    """
    try:
        user_input = request.json.get("message", "")
        if not user_input:
            return jsonify({"error": "Message is required"}), 400

        website_info = get_website_data()
        curriculum_data = load_course_curriculum()
        matched_courses = find_relevant_courses(user_input, curriculum_data)

        curriculum_text = json.dumps(matched_courses, indent=2)
        context = f"{website_info}\n\nRelevant Courses:\n{curriculum_text}"

        response = run_chatbot_crew(user_input, context, matched_courses)
        return jsonify(response)

    except Exception as e:
        print("❌ Error in /smart-chatbot:")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# ========== Health Check ==========
@app.route('/ping', methods=['GET'])
def ping():
    """
    Health Check
    ---
    get:
      summary: Check if the chatbot service is running
      description: Returns a simple status message to confirm the API is alive.
      responses:
        200:
          description: Service is running
          content:
            application/json:
              schema:
                type: object
                properties:
                  status:
                    type: string
                    example: ✅ SkillCapital AI Assistant is running.
    """
    return jsonify({"status": "✅ SkillCapital AI Assistant is running."})
# ========== Main ==========
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))


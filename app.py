import os
import traceback
from flask import Flask, request, jsonify
from flasgger import Swagger, swag_from
from dotenv import load_dotenv
from langchain_openai import OpenAI
from crewai import Agent, Task, Crew

# Load .env variables (like OPENAI_API_KEY)
load_dotenv()

# Initialize Flask app and Swagger docs
app = Flask(__name__)
swagger = Swagger(app)

# Initialize OpenAI LLM (ensure key is set in env)
llm = OpenAI(
    temperature=0.6,
    model_name="gpt-4"  # Or "gpt-3.5-turbo"
)

# Define Agents
researcher = Agent(
    role="AI Researcher",
    goal="Collect accurate and concise research data",
    backstory="Expert in fact-checking and deep research.",
    verbose=False,
    allow_delegation=False,
    llm=llm
)

explainer = Agent(
    role="Insight Generator",
    goal="Summarize and format data into bullet points with markdown and emojis",
    backstory="Specialist in transforming technical info into engaging summaries.",
    verbose=False,
    allow_delegation=False,
    llm=llm
)

# CrewAI Logic
def run_crew_chatbot_pipeline(query: str) -> str:
    research_task = Task(
        description=f"Research this question thoroughly: '{query}'",
        expected_output="Bullet points with facts, examples, and clarity",
        agent=researcher
    )

    explanation_task = Task(
        description=f"Use the above research to create a concise, markdown-formatted response with bullet points and emojis.",
        expected_output=(
            "• Use bullet points\n"
            "• Make it markdown formatted\n"
            "• Include emojis to highlight key facts\n"
            "• Focus on clarity and engagement"
        ),
        agent=explainer,
        context=[research_task]
    )

    crew = Crew(
        agents=[researcher, explainer],
        tasks=[research_task, explanation_task],
        verbose=True,
        async_execution=True,
        max_rpm=100  # Throttle if needed
    )

    result = crew.kickoff()
    return result.output.strip() if hasattr(result, "output") else str(result).strip()

# Flask API Endpoint
@app.route('/chat', methods=['POST'])
@swag_from({
    'tags': ['Chat'],
    'parameters': [{
        'name': 'body',
        'in': 'body',
        'required': True,
        'schema': {
            'type': 'object',
            'properties': {
                'message': {
                    'type': 'string',
                    'example': 'How does quantum computing affect cybersecurity?'
                }
            },
            'required': ['message']
        }
    }],
    'responses': {
        200: {
            'description': 'Formatted response',
            'schema': {
                'type': 'object',
                'properties': {
                    'response': {'type': 'string'}
                }
            }
        },
        400: {'description': 'Missing input'},
        500: {'description': 'Internal server error'}
    }
})
def chat():
    data = request.get_json()
    message = data.get("message", "").strip()

    if not message:
        return jsonify({"error": "Message is required"}), 400

    try:
        response = run_crew_chatbot_pipeline(message)
        return jsonify({"response": response})
    except Exception as e:
        error_trace = traceback.format_exc()
        return jsonify({
            "error": f"{str(e)}",
            "trace": error_trace
        }), 500

# Health check
@app.route('/')
def home():
    return jsonify({
        "message": "✅ CrewAI-powered chatbot is running!",
        "usage": "POST /chat",
        "docs": "/apidocs"
    })

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8000, debug=True)

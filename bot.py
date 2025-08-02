import os
from flask import Flask, request, jsonify
from flasgger import Swagger, swag_from
from dotenv import load_dotenv
from langchain_openai import OpenAI
from crewai import Agent, Task, Crew

# Load environment variables
load_dotenv()

# Initialize Flask app and Swagger
app = Flask(__name__)
swagger = Swagger(app)

# Initialize LLM
llm = OpenAI(
    temperature=0.7,
    model_name="gpt-4"  # Or "gpt-3.5-turbo"
)

# Define Researcher Agent
researcher = Agent(
    role="AI Researcher",
    goal="Thoroughly research and gather factual, relevant information",
    backstory="Expert in AI and information retrieval. Always delivers factual, reliable content.",
    verbose=True,
    allow_delegation=True,
    llm=llm
)

# Define Explainer Agent
explainer = Agent(
    role="Response Formatter",
    goal="Convert technical or detailed data into bullet points and clear insights",
    backstory="A communication expert who presents data clearly using bullet points, markdown, and emojis.",
    verbose=True,
    allow_delegation=False,
    llm=llm
)

# Core function to run CrewAI task
def run_crew_chatbot_pipeline(query: str) -> str:
    # Research Task
    research_task = Task(
        description=f"Conduct research on: '{query}' and gather key factual points.",
        expected_output="List detailed findings in a factual format with context and relevance.",
        agent=researcher
    )

    # Explanation Task
    explanation_task = Task(
        description=f"Using the research, create a clear and engaging bullet-point explanation for: '{query}'",
        expected_output="Respond in bullet points with markdown, use emojis if helpful, and be concise yet informative.",
        agent=explainer,
        context=[research_task]  # Depends on output from researcher
    )

    # Crew setup
    crew = Crew(
        agents=[researcher, explainer],
        tasks=[research_task, explanation_task],
        verbose=False
    )

    result = crew.kickoff()
    return str(result).strip()


# API Endpoint
@app.route('/chat', methods=['POST'])
@swag_from({
    'tags': ['Chat'],
    'parameters': [
        {
            'name': 'body',
            'in': 'body',
            'required': True,
            'schema': {
                'type': 'object',
                'properties': {
                    'message': {
                        'type': 'string',
                        'example': 'How does blockchain impact supply chain transparency?'
                    }
                },
                'required': ['message']
            }
        }
    ],
    'responses': {
        200: {
            'description': 'Chatbot response',
            'schema': {
                'type': 'object',
                'properties': {
                    'response': {'type': 'string'}
                }
            }
        },
        400: {'description': 'Validation error'},
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
        return jsonify({"error": str(e)}), 500

# Home / Health Check
@app.route('/')
def home():
    return jsonify({
        "message": "CrewAI Multi-Agent Chatbot is live.",
        "usage": "POST /chat",
        "docs": "/apidocs"
    })

# Run
if __name__ == '__main__':
    app.run(debug=True)

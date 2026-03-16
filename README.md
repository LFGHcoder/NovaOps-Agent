NovaOps Agent

NovaOps Agent is a prototype system that shows how AI agents can automate parts of the recruiting workflow.

Instead of manually reviewing resumes, preparing interview questions, evaluating candidates, and updating hiring systems, NovaOps coordinates multiple AI agents that work together to evaluate a candidate and generate a hiring recommendation.

This project was built for the Amazon Nova AI Hackathon and uses Amazon Nova models to power reasoning inside several parts of the workflow.

What NovaOps Agent Does

NovaOps takes a job description and candidate resume and runs them through a multi-step AI pipeline.

The system:

Reads the job description and generates an interview plan

Screens the candidate’s resume against the role

Simulates a technical screening interview

Evaluates the candidate’s answers

Simulates ATS workflow actions

Produces a final hiring recommendation

The result is a structured report that summarizes the candidate evaluation.

How the System Works

NovaOps is built as a multi-agent pipeline where each agent performs a specific task.

Workflow:

Planner Agent
   ↓
Screening Agent
   ↓
Interview Agent
   ↓
Evaluation Agent
   ↓
Execution Agent
   ↓
Verification Agent
Planner Agent

Analyzes the job description and generates:

required skills

interview questions

evaluation criteria

a workflow plan

Screening Agent

Evaluates the candidate’s resume against the job description and produces a screening score.

Interview Agent

Simulates a short technical screening interview based on the generated questions.

Evaluation Agent

Analyzes the candidate’s answers and calculates a final evaluation score.

Execution Agent

Simulates actions that might normally happen inside an Applicant Tracking System.

Verification Agent

Ensures the workflow completed successfully and returns the final structured result.

Example Output
{
  "candidate": "Alice Johnson",
  "overall_score": 78,
  "recommendation": "Shortlist",
  "status": "completed"
}
Technologies Used

Python

Amazon Bedrock

Amazon Nova 2 Lite

Boto3 (AWS SDK for Python)

JSON structured workflows

Running the Project

Clone the repository:

git clone https://github.com/LFGHcoder/NovaOps-Agent
cd NovaOps-Agent

Install dependencies:

pip install boto3

Configure AWS credentials:

aws configure

Then run the workflow:

python main.py
Future Work

The next step for NovaOps is building an AI Interviewer.

Instead of only analyzing resumes, the system will conduct a short automated interview with candidates. The first screening round will include three sections:

• Behavioral questions (5 minutes)
• Cultural fit questions (5 minutes)
• Technical questions (5 minutes)

The AI will evaluate how candidates structure their answers, how clearly they communicate their reasoning, and how well their responses match the job requirements.

This could automate the first round of interviews, helping companies screen candidates faster and more consistently.

{
  "steps": [
    {"action": "analyze_job"},
    {"action": "score_resumes"},
    {"action": "shortlist", "top_n": 3},
    {"action": "update_status", "candidate": "NAME", "status": "Shortlisted"},
    {"action": "send_email", "candidate": "NAME"}
  ]
}
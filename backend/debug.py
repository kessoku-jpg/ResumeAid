from resume_parser import parse_resume
from matching_engine import debug_match

resume = parse_resume("res_tech.pdf")
debug_match(resume, "Full-stack role requiring Python backend experience, SQL databases, and containerization with Docker. Bonus if you've used REST APIs.", job_title="Backend/Full-Stack Developer")

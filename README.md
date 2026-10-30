# ResumeAid

ResumeAid is a resume-to-job matching application designed for Filipino job seekers. It extracts information from a resume, checks the resume for common issues, searches supported job sources, and ranks available listings using semantic similarity.

> **Important:** ResumeAid is an experimental job-search aid. Match scores are estimates, not guarantees of qualification, hiring outcomes, or job availability. Always review the original job listing and verify the employer before applying.

## Features

- Upload a resume in **PDF** or **DOCX** format.
- Extract resume information such as contact details, skills, education, and experience.
- Run a basic resume health check and show suggested skills.
- Infer a likely occupation/search query from the resume.
- Search job listings from:
  - PhilJobNet
  - OnlineJobs.ph
  - Adzuna (optional API credentials)
- Rank job listings using Sentence-BERT (`all-MiniLM-L6-v2`) and additional matching signals.
- Filter and sort results by available listing details, such as match score, work setup, job type, source, salary, and posting date.
- Choose a country in **Settings**.
- Toggle dark mode.

## Requirements

The easiest setup is on **Windows 10 or Windows 11**.

- Python **3.10 or later** (Python 3.10–3.12 is recommended for compatibility with the ML dependencies).
- Internet connection for installing Python packages, downloading the language/model files on first run, and retrieving online job listings.
- A modern browser such as Chrome, Edge, or Firefox.
- Enough free disk space for Python dependencies and the machine-learning model. The first installation can be large and may take several minutes.

## Quick start (Windows)

1. Download or clone this repository.
2. Extract the project folder if you downloaded a ZIP.
3. Open the extracted `ResumeAid` folder.
4. Double-click **`start.bat`**.
5. On the first run, the launcher will:
   - Check that Python is available.
   - Create `backend\.venv` if needed.
   - Install the packages in `backend\requirements.txt`.
   - Download the spaCy English model (`en_core_web_sm`) if it is missing.
   - Create `backend\.env` from `backend\.env.example` if it does not exist.
   - Start the Flask API on `http://127.0.0.1:5000`.
   - Open `index.html` in your browser.
6. Wait for the backend to finish loading. The first run may take longer because dependencies and the Sentence-BERT model may need to be downloaded.

Keep the **ResumeAid API** terminal window open while using the application. Close that window when you are finished to stop the local backend.

## Manual setup

If you prefer to start the backend yourself, open a terminal in the project folder and run:

```powershell
cd backend
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m spacy download en_core_web_sm
Copy-Item .env.example .env
python api.py
```

If PowerShell blocks virtual-environment activation, you can run the environment's Python directly instead:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m spacy download en_core_web_sm
.\.venv\Scripts\python.exe api.py
```

With the backend running, open the project's `index.html` file in your browser.

## Using ResumeAid

1. **Open the Upload page.** Make sure the backend is running. The application checks the local API when it starts.
2. **Select a resume.** Drag a file into the upload area or click it to browse for a file.
   - Supported formats: `.pdf` and `.docx`
   - Maximum file size: **10 MB**
   - Legacy `.doc` files are not supported. Save them as PDF or DOCX first.
3. **Analyze the resume.** Click the analyze button and wait for parsing, job searching, and matching to finish. This may take some time, especially on the first run.
4. **Review the resume check.** If the application identifies possible issues or missing skills, review the suggestions. Continue to the matches page when ready.
5. **Review job matches.** The Matches page displays ranked listings. Select a listing to see its match details and use the listing link to visit the original job source.
6. **Refine the results.** Use the available filters and sorting controls to narrow the list. You can refresh listings when the refresh control is available.
7. **Change your country or appearance.** Open **Settings** to select a supported country or toggle dark mode. Your country and dark-mode preferences are stored in your browser's local storage.

### Supported country selections

Philippines, United States, United Kingdom, Australia, Canada, Singapore, New Zealand, India, Germany, France, Netherlands, Italy, Poland, Austria, South Africa, and Brazil.

Job-source coverage varies by country. PhilJobNet and OnlineJobs.ph are Philippines-oriented sources; Adzuna coverage depends on its supported markets and API access.

## Optional Adzuna API configuration

Adzuna credentials are optional. To configure them:

1. Open `backend\.env`.
2. Add your Adzuna application ID and key:

```dotenv
ADZUNA_APP_ID=your_app_id
ADZUNA_APP_KEY=your_app_key
```

3. Save the file and restart the backend.

You can obtain credentials from the Adzuna developer website: <https://developer.adzuna.com/>.

Do not commit your real `.env` file or publish API credentials. The repository includes `.env.example` as a template.

## How matching works (high level)

1. The backend extracts text from the uploaded PDF or DOCX resume.
2. It parses resume sections and extracts information such as skills, education, and work experience.
3. It derives a search query, generally based on an inferred occupation or detected skills.
4. The job-source modules retrieve listings. If no live or cached listings are available, the application may show clearly identified sample listings.
5. The matching engine compares resume information with job requirements using Sentence-BERT embeddings and other scoring signals.
6. The frontend displays the ranked results and available match explanations.

A higher score means the listing appears more similar to the resume according to the application's matching method. It should not be interpreted as a probability of getting hired.

## Configuration

The backend reads optional settings from `backend\.env`:

| Variable | Purpose | Default |
|---|---|---|
| `ADZUNA_APP_ID` | Adzuna application ID | Empty |
| `ADZUNA_APP_KEY` | Adzuna application key | Empty |
| `CORS_ORIGINS` | Comma-separated allowed frontend origins | `*` |
| `HOST` | Backend bind address | `127.0.0.1` |
| `PORT` | Backend port | `5000` |
| `FLASK_DEBUG` | Enables Flask debug mode for local development | `0` |

The frontend defaults to `http://localhost:5000` for its API. For local use, leave `HOST` as `127.0.0.1` and keep the API accessible only from your own computer.

## API endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/health` | Checks whether the backend is responding. |
| `POST` | `/api/analyze` | Accepts a resume upload and returns the parsed resume summary, resume health check, search information, and ranked job matches. |

The analyze endpoint expects a multipart form upload with a `resume` field. It also accepts optional `country`, `source`, `query`, and `refresh` fields.

## Troubleshooting

### The browser says it cannot reach the server

- Make sure the **ResumeAid API** terminal window is still open.
- Check whether the backend reports an error during startup.
- Confirm that `http://localhost:5000/api/health` responds with `{"status":"ok"}`.
- Check whether another application is already using port `5000`.

### The first launch takes a long time

The launcher may be installing PyTorch and Sentence-Transformers, downloading the spaCy language model, or downloading the Sentence-BERT model. Allow the first setup to finish and keep the internet connection active.

### Python is not found

Install Python from <https://www.python.org/downloads/> and enable **Add Python to PATH** during installation. Reopen the terminal and run `start.bat` again.

### No live job listings appear

- Check your internet connection.
- Try another country or run the search again later.
- Some job sites may be temporarily unavailable, change their page structure, or limit automated requests.
- If live and cached results are unavailable, ResumeAid may display sample listings instead. Check the search notice to distinguish sample results from live listings.
- Configure Adzuna credentials if you want to use that optional source.

### A resume cannot be processed

- Make sure the file is a valid PDF or DOCX.
- Make sure it is not empty and is no larger than 10 MB.
- If a PDF is scanned or image-only, text extraction may be limited. A text-based PDF is preferable.
- Try exporting the resume again from the application that created it.

## Privacy and security notes

- ResumeAid is designed for local use by default. The backend binds to `127.0.0.1` unless configured otherwise.
- The API writes an uploaded resume to a temporary directory for processing and attempts to delete that directory after the request finishes.
- The API returns extracted contact information and resume details to the browser as part of the analysis response.
- The job cache stores retrieved job listings, not uploaded resume files.
- Do not upload someone else's resume without their permission. Use dummy or consented sample resumes when demonstrating the project.
- Do not commit real resumes, personal documents, `.env` files, API keys, or other credentials to a public repository.
- The default CORS setting is permissive for development. **CORS is not authentication.** Do not expose this development API to the public internet as-is. A public deployment needs authentication or abuse controls, strict origin rules, HTTPS, secure file handling, privacy disclosures, and appropriate retention/deletion procedures.
- Treat uploaded resumes and job listings as untrusted input. Keep Python packages and model files from trusted sources.

## Project structure

```text
ResumeAid/
├── backend/
│   ├── api.py                         # Flask API
│   ├── pipeline.py                    # Resume-to-job processing flow
│   ├── resume_parser.py                # PDF/DOCX text extraction and parsing
│   ├── matching_engine.py              # Semantic matching and scoring
│   ├── resume_improver.py               # Resume health checks and suggestions
│   ├── job_scraper.py                   # Adzuna integration
│   ├── onlinejobs_scraper.py            # OnlineJobs.ph integration
│   ├── philjobnet_scraper.py            # PhilJobNet integration
│   ├── job_cache.py                    # SQLite job-listing cache
│   ├── requirements.txt
│   └── .env.example
├── fonts/                              # Locally hosted web fonts
├── app.js                              # Frontend behavior
├── index.html                          # Application interface
├── style.css                           # Application styles
├── start.bat                           # Windows launcher
└── README.md
```

## License

No license is specified in this repository. Unless a license is added, the default copyright rules apply; people may view the public source on GitHub but do not automatically receive permission to reuse, modify, or redistribute it. Add a license file if you want to grant those permissions.

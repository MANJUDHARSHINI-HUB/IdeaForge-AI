# 💡 IdeaForge AI

### AI-Powered Project Development Advisor

IdeaForge AI is an AI-powered project development advisor that helps students and developers transform a basic project idea into a structured, feasible, and professional project blueprint.

Instead of providing only a general AI response, IdeaForge focuses on the **project-development workflow** — from understanding the initial idea to defining users, features, technologies, improvements, and a final downloadable blueprint.

## 🚀 Live Demo

**Try IdeaForge AI:**
https://ideaforge-ai-a7aiugt57ktpn352ajhb2v.streamlit.app/

---

## 🎯 Problem

Students often have project ideas but face difficulties in converting those ideas into practical projects.

Common questions include:

* Is my project idea feasible?
* Who will use this project?
* What features should I include?
* Which technologies should I use?
* What improvements can I make?
* How difficult will the project be?
* How can I structure my project properly?

IdeaForge AI addresses these challenges by providing an interactive AI-assisted project development workflow.

---

## 💡 Solution

IdeaForge AI takes a user's project idea and generates a structured project analysis.

The system can provide:

* Project analysis
* Problem understanding
* Target users
* Core features
* Technology recommendations
* Difficulty assessment
* Project improvements
* Interactive project refinement
* Project blueprint
* Downloadable PDF

Users can also continue chatting with the system after the initial project is created and modify the project using natural-language instructions.

---

## ✨ Key Features

### 🔍 Project Analysis

Analyzes a project idea and provides structured recommendations.

### 👥 Target User Identification

Identifies the intended users of the proposed project.

### 🧩 Feature Planning

Creates and manages the core features of the project.

### 💬 Interactive Project Chat

Allows users to modify and refine their project using natural language.

For example:

> "Add playlist sharing."

The system identifies the request, generates an appropriate description, and updates the project.

### 🧠 RAG-Based Knowledge Retrieval

Uses Retrieval-Augmented Generation to retrieve relevant project knowledge before generating responses.

### 📋 Project Blueprint

Creates a structured final representation of the project.

### 📄 PDF Export

Generates a downloadable project blueprint using Python and ReportLab.

---

## 🏗️ System Architecture

```text
                 USER
                   ↓
            Streamlit UI
                   ↓
          Python Application
                   ↓
            User Request
                   ↓
           RAG Retrieval
                   ↓
       Sentence Transformers
        all-MiniLM-L6-v2
                   ↓
              ChromaDB
                   ↓
        Relevant Project Context
                   ↓
              Groq API
        openai/gpt-oss-120b
                   ↓
            AI Response
                   ↓
        Project State Update
                   ↓
       Project Blueprint
                   ↓
          ReportLab PDF
                   ↓
          Downloadable PDF
```

---

## 🧠 RAG Architecture

IdeaForge AI uses Retrieval-Augmented Generation (RAG).

The process is:

```text
User Query
    ↓
Create Query Embedding
    ↓
Search ChromaDB
    ↓
Retrieve Relevant Knowledge
    ↓
Combine Query + Retrieved Context
    ↓
Groq LLM
    ↓
Generated Response
```

The embedding model currently used by the application is:

**Sentence Transformers — all-MiniLM-L6-v2**

ChromaDB is used as the vector database for semantic retrieval.

---

## 🤖 Language Model

The deployed version uses the **Groq API** with:

```text
Model: openai/gpt-oss-120b
```

The LLM is responsible for understanding natural-language requests and generating project-related responses.

The Python application controls the project state and performs operations such as updating project features and generating the final blueprint.

---

## 🛠️ Technology Stack

| Component             | Technology                |
| --------------------- | ------------------------- |
| Programming Language  | Python                    |
| User Interface        | Streamlit                 |
| LLM Provider          | Groq                      |
| LLM Model             | openai/gpt-oss-120b       |
| Embedding Model       | all-MiniLM-L6-v2          |
| Embedding Framework   | Sentence Transformers     |
| Vector Database       | ChromaDB                  |
| PDF Generation        | ReportLab                 |
| Environment Variables | python-dotenv             |
| Version Control       | Git                       |
| Repository            | GitHub                    |
| Deployment            | Streamlit Community Cloud |

---

## 🔄 Project Workflow

```text
1. User enters project idea
        ↓
2. IdeaForge analyzes the idea
        ↓
3. Relevant knowledge is retrieved
        ↓
4. LLM generates project recommendations
        ↓
5. Project structure is created
        ↓
6. User can modify the project through chat
        ↓
7. Features and project details are updated
        ↓
8. User finalizes the project
        ↓
9. Project blueprint is generated
        ↓
10. Blueprint can be downloaded as PDF
```

---

## 💬 Example Interaction

### User

> I want to build a playlist manager.

### IdeaForge AI

The system can generate a structured project containing:

* Project description
* Target users
* Core features
* Technology recommendations
* Difficulty level
* Project improvements

The user can then continue the conversation.

### User

> Add playlist sharing.

### IdeaForge AI

The system processes the request and adds the feature with a generated description.

The updated feature is reflected in the project blueprint.

---

## 📄 PDF Generation

After the project is finalized, IdeaForge AI creates a project blueprint.

The application uses **ReportLab** to generate the PDF.

```text
Project Data
     ↓
Blueprint Content
     ↓
Python
     ↓
ReportLab
     ↓
PDF Document
     ↓
Streamlit Download
```

---

## 🔐 Security

The Groq API key is not stored in the GitHub repository.

During local development, environment variables are loaded using `.env`.

During deployment, the API key is stored securely using **Streamlit Secrets**.

The `.env` file should never be committed to GitHub.

---

## ☁️ Deployment

IdeaForge AI is deployed using:

**Streamlit Community Cloud**

The application is connected to the GitHub repository and automatically redeploys when changes are pushed to the configured branch.

### Live Application

https://ideaforge-ai-a7aiugt57ktpn352ajhb2v.streamlit.app/

---

## 🚀 Running Locally

### 1. Clone the repository

```bash
git clone https://github.com/MANJUDHARSHINI-HUB/IdeaForge-AI.git
cd IdeaForge-AI
```

### 2. Create and activate the environment

```bash
conda create -n ideaforge python=3.13
conda activate ideaforge
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure the API key

Create a `.env` file:

```text
GROQ_API_KEY=your_api_key_here
```

### 5. Run the application

```bash
streamlit run app.py
```

---

## 📁 Project Structure

```text
IdeaForge-AI/
│
├── app.py
├── requirements.txt
├── .gitignore
├── chroma_db/
├── README.md
└── .env
```

> `.env` should remain local and should not be committed to GitHub.

---

## 🎯 Use Cases

IdeaForge AI can be useful for:

* College students planning academic projects
* Beginners learning project development
* Developers exploring new project ideas
* Project mentors guiding students
* Teams brainstorming and refining project concepts

---

## 🌱 Future Enhancements

Potential future improvements include:

* User authentication
* Project history
* Multiple LLM providers
* Advanced feasibility analysis
* Personalized project recommendations
* Automatic GitHub project generation
* Automatic starter-code generation
* Team collaboration
* Cloud-based project storage

---

## 📌 Project Objective

The main objective of IdeaForge AI is to reduce the gap between:

**"I have an idea."**

and

**"I know how to build it."**

---

## 👩‍💻 Author

**S. Manju Dharshini**

B.Sc Computer Science with Artificial Intelligence

GitHub:
https://github.com/MANJUDHARSHINI-HUB

---

## ⭐ Project

If you find the project useful, consider giving the repository a star.

**IdeaForge AI — From an idea to a structured project blueprint.**

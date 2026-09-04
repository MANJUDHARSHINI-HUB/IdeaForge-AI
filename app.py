import streamlit as st
import chromadb
from groq import Groq
from dotenv import load_dotenv
import os
import json
import re
import io
from datetime import datetime
from sentence_transformers import SentenceTransformer

load_dotenv()

groq_client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

from reportlab.lib.pagesizes import A4
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle
)
from reportlab.lib import colors
from reportlab.lib.styles import (
    getSampleStyleSheet,
    ParagraphStyle
)
from reportlab.lib.enums import TA_CENTER


# ============================================================
# CONFIGURATION
# ============================================================

OLLAMA_MODEL = "llama3.2:3b"
EMBED_MODEL = "nomic-embed-text"
CHROMA_PATH = "chroma_db"


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="IdeaForge AI",
    page_icon="💡",
    layout="wide"
)


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
    <style>
    .main-title {
        font-size: 42px;
        font-weight: 800;
        margin-bottom: 0px;
    }

    .subtitle {
        font-size: 18px;
        color: #666;
        margin-bottom: 25px;
    }

    .project-card {
        padding: 18px;
        border-radius: 15px;
        border: 1px solid #ddd;
        background-color: #fafafa;
        margin-bottom: 15px;
    }

    .chat-user {
        padding: 14px;
        border-radius: 12px;
        background-color: #e8f0fe;
        margin-bottom: 10px;
    }

    .chat-ai {
        padding: 14px;
        border-radius: 12px;
        background-color: #f4f4f4;
        margin-bottom: 10px;
    }

    .locked {
        padding: 15px;
        border-radius: 12px;
        background-color: #e8f5e9;
        border: 1px solid #81c784;
        margin-bottom: 15px;
    }

    .warning-box {
        padding: 15px;
        border-radius: 12px;
        background-color: #fff8e1;
        border: 1px solid #ffcc80;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# PROJECT STATE
# ============================================================

def empty_project():

    return {
        "name": "",
        "domain": "",
        "problem": "",
        "solution": "",
        "target_users": "",
        "platform": "",
        "ai_features": [],
        "features": [],
        "technologies": [],
        "models": [],
        "rag": "",
        "database": "",
        "difficulty": "",
        "feasibility": "",
        "data_requirements": [],
        "modules": [],
        "workflow": [],
        "improvements": [],
        "alternatives": [],
        "skills": [],
        "reasoning": ""
    }


# ============================================================
# SESSION STATE
# ============================================================

if "project" not in st.session_state:
    st.session_state.project = empty_project()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "locked" not in st.session_state:
    st.session_state.locked = False

if "analysis_done" not in st.session_state:
    st.session_state.analysis_done = False

if "rag_sources" not in st.session_state:
    st.session_state.rag_sources = []

if "last_analysis" not in st.session_state:
    st.session_state.last_analysis = ""


# ============================================================
# GROQ CHAT
# ============================================================

def ollama_chat(prompt, system_prompt=None):

    try:

        messages = []

        if system_prompt:

            messages.append(
                {
                    "role": "system",
                    "content": system_prompt
                }
            )

        messages.append(
            {
                "role": "user",
                "content": prompt
            }
        )

        response = groq_client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=messages,
            temperature=0.2
        )

        return response.choices[0].message.content

    except Exception as e:

        return f"GROQ_ERROR: {str(e)}"


# ============================================================
# EMBEDDINGS
# ============================================================

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


def create_embedding(text):
    try:
        return embedding_model.encode(text).tolist()
    except Exception as e:
        st.warning(f"Embedding error: {e}")
        return None


# ============================================================
# CHROMADB
# ============================================================

@st.cache_resource
def load_chroma():

    try:

        client = chromadb.PersistentClient(
            path=CHROMA_PATH
        )

        collections = client.list_collections()

        if not collections:
            return None

        collection_names = [
            c.name for c in collections
        ]

        selected_collection = None

        for name in collection_names:

            if "project" in name.lower():

                selected_collection = name

                break

        if selected_collection is None:

            selected_collection = collection_names[0]

        return client.get_collection(
            selected_collection
        )

    except Exception as e:

        st.warning(
            f"Could not load ChromaDB: {e}"
        )

        return None


# ============================================================
# RAG RETRIEVAL
# ============================================================

def retrieve_context(query, top_k=5):

    collection = load_chroma()

    if collection is None:

        return [], []

    embedding = create_embedding(query)

    if embedding is None:

        return [], []

    try:

        results = collection.query(
            query_embeddings=[embedding],
            n_results=top_k
        )

        documents = results.get(
            "documents",
            [[]]
        )[0]

        metadatas = results.get(
            "metadatas",
            [[]]
        )[0]

        return documents, metadatas

    except Exception as e:

        st.warning(
            f"RAG retrieval failed: {e}"
        )

        return [], []


# ============================================================
# JSON EXTRACTION
# ============================================================

def extract_json(text):

    if not text:

        return None

    text = text.strip()

    text = re.sub(
        r"```json\s*",
        "",
        text,
        flags=re.IGNORECASE
    )

    text = re.sub(
        r"```\s*",
        "",
        text
    )

    start = text.find("{")

    end = text.rfind("}")

    if start == -1 or end == -1:

        return None

    json_text = text[
        start:end + 1
    ]

    try:

        return json.loads(
            json_text
        )

    except:

        return None


# ============================================================
# NORMALIZE PROJECT
# ============================================================

def normalize_project(data):

    text_fields = [

        "name",
        "domain",
        "problem",
        "solution",
        "target_users",
        "platform",
        "rag",
        "database",
        "difficulty",
        "feasibility",
        "reasoning"

    ]

    list_fields = [

        "ai_features",
        "features",
        "technologies",
        "models",
        "data_requirements",
        "modules",
        "workflow",
        "improvements",
        "alternatives",
        "skills"

    ]

    for field in text_fields:

        if field in data:

            value = data[field]

            if value is not None:

                st.session_state.project[field] = str(
                    value
                )

    for field in list_fields:

        if field not in data:

            continue

        value = data[field]

        if isinstance(value, list):

            cleaned = []

            for item in value:

                item = str(item).strip()

                if item and item not in cleaned:

                    cleaned.append(item)

            st.session_state.project[field] = cleaned

        elif isinstance(value, str):

            items = [

                x.strip()

                for x in value.split(",")

                if x.strip()

            ]

            st.session_state.project[field] = items


# ============================================================
# PROJECT → TEXT
# ============================================================

def project_to_text():

    p = st.session_state.project

    return f"""

PROJECT NAME:

{p["name"]}

DOMAIN:

{p["domain"]}

PROBLEM:

{p["problem"]}

SOLUTION:

{p["solution"]}

TARGET USERS:

{p["target_users"]}

PLATFORM:

{p["platform"]}

AI FEATURES:

{", ".join(p["ai_features"])}

CORE FEATURES:

{", ".join(p["features"])}

TECHNOLOGIES:

{", ".join(p["technologies"])}

AI MODELS:

{", ".join(p["models"])}

RAG:

{p["rag"]}

DATABASE:

{p["database"]}

DIFFICULTY:

{p["difficulty"]}

FEASIBILITY:

{p["feasibility"]}

DATA REQUIREMENTS:

{", ".join(p["data_requirements"])}

MODULES:

{", ".join(p["modules"])}

WORKFLOW:

{" -> ".join(p["workflow"])}

IMPROVEMENTS:

{", ".join(p["improvements"])}

ALTERNATIVES:

{", ".join(p["alternatives"])}

SKILLS:

{", ".join(p["skills"])}

REASONING:

{p["reasoning"]}

"""


# ============================================================
# INITIAL PROJECT ANALYSIS
# ============================================================

def analyze_project(
    idea,
    skills,
    interests,
    difficulty
):

    rag_query = f"""

Project idea:

{idea}

Skills:

{skills}

Interests:

{interests}

Difficulty:

{difficulty}

Find relevant project architecture,

AI approaches, technologies,

datasets, RAG approaches,

models and implementation ideas.

"""

    documents, metadatas = retrieve_context(
        rag_query,
        top_k=5
    )

    st.session_state.rag_sources = documents

    rag_context = "\n\n".join(
        documents
    )

    system_prompt = """

You are IdeaForge AI.

You are an expert AI project development advisor.

Understand the user's actual project idea.

Build the project around THAT idea.

Do not inject unrelated domains.

For example:

A music project should receive music-related
features.

A finance project should receive finance-related
features.

A healthcare project should receive healthcare-related
features.

Do not blindly add Generative AI or RAG.

Use them only when they actually make sense.

A playlist recommendation system is NOT automatically
a music-generation system.

Return ONLY valid JSON.

"""

    prompt = f"""

USER PROJECT IDEA:

{idea}

USER SKILLS:

{skills}

USER INTERESTS:

{interests}

PREFERRED DIFFICULTY:

{difficulty}

RETRIEVED KNOWLEDGE:

{rag_context}

Create a realistic student-level professional
project blueprint.

Return EXACTLY:

{{
    "name": "",
    "domain": "",
    "problem": "",
    "solution": "",
    "target_users": "",
    "platform": "",
    "ai_features": [],
    "features": [],
    "technologies": [],
    "models": [],
    "rag": "",
    "database": "",
    "difficulty": "",
    "feasibility": "",
    "data_requirements": [],
    "modules": [],
    "workflow": [],
    "improvements": [],
    "alternatives": [],
    "skills": [],
    "reasoning": ""
}}

IMPORTANT:

- Preserve the user's original idea.

- Features must belong to the domain.

- AI must have a real purpose.

- RAG must have a real purpose.

- Recommend realistic models.

- Do not claim the system generates audio if it only
  recommends existing music.

- Workflow must be logical.

- Do not create duplicate modules.

- Keep the project feasible for a student.

"""

    result = ollama_chat(
        prompt,
        system_prompt
    )

    data = extract_json(
        result
    )

    if data:

        normalize_project(
            data
        )

        st.session_state.analysis_done = True

        st.session_state.last_analysis = result

        return True

    st.error(
        "The AI returned an invalid project structure."
    )

    st.write(result)

    return False


# ============================================================
# RECENT CHAT TEXT
# ============================================================

def recent_chat_text():

    recent = (
        st.session_state.chat_history[-10:]
    )

    if not recent:

        return "No previous conversation."

    return json.dumps(
        recent,
        indent=2
    )


# ============================================================
# UNDERSTAND USER INTENT
# ============================================================

def understand_chat(user_message):

    system_prompt = """

You are the intent-understanding engine of IdeaForge AI.

The user is discussing ONE current project.

Use BOTH the current project and the conversation history.

Possible intents:

question

add_feature

remove_feature

modify_feature

improve

alternative

technology

ai_model

rag_question

feasibility

workflow

project_change

finalize

reject

general


IMPORTANT:

If the user says:

"Add that feature"

look at the previous conversation and identify
the feature they are referring to.

If the user says:

"Remove that"

look at the previous conversation and identify
the feature.

If the user says:

"I don't like that feature"

identify the feature from conversation history.


IMPORTANT FOR ADD FEATURE:

If the user asks to add a feature but does NOT
specify the feature name, the intent must still be:

{
    "intent": "add_feature",
    "target": "",
    "value": "",
    "confidence": 1.0
}

Examples:

"Add a new feature"

"Add an additional feature"

"I want to add a feature"

"Can you add something useful?"

"Think of a new feature"

All of these should return:

{
    "intent": "add_feature",
    "target": "",
    "value": "",
    "confidence": 1.0
}


For add_feature:

target = the feature to add.

If the user did not specify a feature,
target must be an empty string.


For remove_feature:

target = the feature to remove.


For modify_feature:

target = existing feature.

value = new version.


For project_change:

target = the project field being changed.

value = the new value.


Examples:

User:

"Change the target audience to college students."

Return:

{
    "intent": "project_change",
    "target": "target_users",
    "value": "College students",
    "confidence": 1.0
}


User:

"Change the platform to a web application."

Return:

{
    "intent": "project_change",
    "target": "platform",
    "value": "Web application",
    "confidence": 1.0
}


User:

"Change the difficulty to Advanced."

Return:

{
    "intent": "project_change",
    "target": "difficulty",
    "value": "Advanced",
    "confidence": 1.0
}


User:

"Change the domain to education."

Return:

{
    "intent": "project_change",
    "target": "domain",
    "value": "Education",
    "confidence": 1.0
}


For finalize:

intent = finalize.


Do not invent a feature that does not exist in the
conversation when resolving references.

"""

    prompt = f"""

CURRENT PROJECT:

{project_to_text()}

RECENT CONVERSATION:

{recent_chat_text()}

CURRENT USER MESSAGE:

{user_message}

Determine the user's intent.

"""

    result = ollama_chat(
        prompt,
        system_prompt
    )

    data = extract_json(
        result
    )

    if data:

        return data

    return {

        "intent": "general",

        "target": "",

        "value": "",

        "confidence": 0.0

    }


# ============================================================
# FIND FEATURE
# ============================================================

def find_feature(feature_text):

    if not feature_text:

        return None

    feature_text = feature_text.strip().lower()

    for feature in st.session_state.project["features"]:

        existing = feature.lower()

        if (
            feature_text == existing
            or feature_text in existing
        ):

            return feature

        feature_name = existing.split(
            "—",
            1
        )[0].strip()

        if (
            feature_text == feature_name
            or feature_text in feature_name
            or feature_name in feature_text
        ):

            return feature

    return None


# ============================================================
# ADD FEATURE
# ============================================================

def add_feature(feature):

    feature = feature.strip()

    # ========================================================
    # AUTOMATIC FEATURE GENERATION
    # ========================================================

    if not feature:

        prompt = f"""

CURRENT PROJECT:

{project_to_text()}

The user wants to add a new feature but did not
specify what the feature should be.

Think like a professional project advisor.

Suggest ONE useful new feature that:

1. Fits the current project's domain.

2. Fits the current target users.

3. Is genuinely useful.

4. Does not duplicate an existing feature.

5. Is realistic for a student project.

6. Does not change the project's main purpose.

Return ONLY the feature name.

Do not provide an explanation.

Do not provide a numbered list.

Do not provide multiple options.

"""

        generated_feature = ollama_chat(
            prompt,
            """

You are an expert project architecture advisor.

Choose one useful feature for the current project.

The feature must be specific to the project,
not a generic feature.

Return ONLY the feature name.

"""
        )

        if generated_feature.startswith(
            "GROQ_ERROR:"
        ):

            return None

        feature = generated_feature.strip()

        # Remove markdown formatting if the model adds it
        feature = re.sub(
            r"^[#*\-\d.\s]+",
            "",
            feature
        ).strip()

        # Remove accidental explanation after newline
        feature = feature.split(
            "\n",
            1
        )[0].strip()

    # ========================================================
    # CHECK DUPLICATE
    # ========================================================

    existing = find_feature(
        feature
    )

    if existing:

        return None

    # ========================================================
    # GENERATE FEATURE DESCRIPTION
    # ========================================================

    prompt = f"""

CURRENT PROJECT:

{project_to_text()}

NEW FEATURE:

{feature}

Create a concise professional description for this feature.

Explain:

1. What the feature does.
2. How it works in this project.
3. Why it is useful for the target users.

Return ONLY the feature description.

Keep it student-friendly and specific to this project.

Do not introduce unrelated functionality.

"""

    description = ollama_chat(
        prompt,
        """

You are a project architecture advisor.

Describe the requested feature specifically for
the current project.

Do not change the project domain.
Do not invent unrelated functionality.

Return only a concise feature description.

"""
    )

    if description.startswith(
        "GROQ_ERROR:"
    ):

        description = (
            "Provides useful functionality "
            "that supports the project's main goal."
        )

    description = description.strip()

    feature_entry = (
        f"{feature} — {description}"
    )

    st.session_state.project[
        "features"
    ].append(
        feature_entry
    )

    return feature_entry


# ============================================================
# REMOVE FEATURE
# ============================================================

def remove_feature(feature):

    existing = find_feature(
        feature
    )

    if existing:

        st.session_state.project[
            "features"
        ].remove(existing)

        return existing

    return None


# ============================================================
# PROCESS CHAT
# ============================================================

def process_chat(user_message):

    intent_data = understand_chat(
        user_message
    )

    intent = intent_data.get(
        "intent",
        "general"
    )

    target = str(
        intent_data.get(
            "target",
            ""
        )
    ).strip()

    value = str(
        intent_data.get(
            "value",
            ""
        )
    ).strip()


    # ========================================================
    # FINALIZE
    # ========================================================

    if intent == "finalize":

        st.session_state.locked = True

        return """

Your project has been finalized and locked. 🔒

The latest project state will now be used for
the final blueprint and PDF.

"""


    # ========================================================
    # BLOCK MODIFICATION AFTER LOCK
    # ========================================================

    modification_intents = [

        "add_feature",
        "remove_feature",
        "modify_feature",
        "project_change"

    ]

    if (
        st.session_state.locked
        and intent in modification_intents
    ):

        return """

🔒 This project is currently locked.

Unlock the project before making changes.

"""


    # ========================================================
    # ADD FEATURE
    # ========================================================

    if intent == "add_feature":

        # If user did not specify a feature,
        # automatically generate one.
        if not target:

            added = add_feature(
                ""
            )

            if added:

                feature_name = added.split(
                    "—",
                    1
                )[0].strip()

                description = added.split(
                    "—",
                    1
                )[1].strip()

                return (
                    f"💡 I thought of a useful feature for your project.\n\n"
                    f"✅ Added **{feature_name}** to the project.\n\n"
                    f"**What it does:** {description}\n\n"
                    "The Core Features section in the "
                    "Blueprint has also been updated."
                )

            return """

I couldn't generate a suitable new feature right now.

Please try again.

"""

        # User specified the feature
        added = add_feature(
            target
        )

        if added:

            description = added.split(
                "—",
                1
            )[1].strip()

            return (
                f"✅ Added **{target}** to the project.\n\n"
                f"**What it does:** {description}\n\n"
                "The Core Features section in the "
                "Blueprint has also been updated."
            )

        return (
            f"That feature already exists in the project: "
            f"**{target}**."
        )


    # ========================================================
    # REMOVE FEATURE
    # ========================================================

    if intent == "remove_feature":

        if not target:

            return """

I couldn't identify which feature you want to remove.

"""

        removed = remove_feature(
            target
        )

        if removed:

            return (
                f"✅ Removed **{removed}** from the project.\n\n"
                "No unrelated project information was changed."
            )

        return (
            f"I couldn't find **{target}** "
            "in the current project features."
        )


    # ========================================================
    # PROJECT FIELD CHANGE
    # ========================================================

    if intent == "project_change":

        target_lower = target.lower()


        # ----------------------------------------------------
        # TARGET USERS
        # ----------------------------------------------------

        if (
            target_lower in [
                "target_users",
                "target users",
                "target audience",
                "audience",
                "users",
                "user"
            ]
            or "target" in target_lower
            or "audience" in target_lower
        ):

            if not value:

                return """

Please tell me who the new target audience should be.

"""

            st.session_state.project[
                "target_users"
            ] = value

            return (
                "✅ Target audience updated successfully.\n\n"
                f"**New Target Users:** {value}\n\n"
                "The Project Blueprint has been updated."
            )


        # ----------------------------------------------------
        # DOMAIN
        # ----------------------------------------------------

        if (
            target_lower == "domain"
            or "domain" in target_lower
        ):

            if not value:

                return """

Please tell me the new project domain.

"""

            st.session_state.project[
                "domain"
            ] = value

            return (
                f"✅ Domain updated to **{value}**.\n\n"
                "The Project Blueprint has been updated."
            )


        # ----------------------------------------------------
        # PLATFORM
        # ----------------------------------------------------

        if (
            target_lower == "platform"
            or "platform" in target_lower
        ):

            if not value:

                return """

Please tell me the new platform.

"""

            st.session_state.project[
                "platform"
            ] = value

            return (
                f"✅ Platform updated to **{value}**.\n\n"
                "The Project Blueprint has been updated."
            )


        # ----------------------------------------------------
        # DIFFICULTY
        # ----------------------------------------------------

        if (
            target_lower == "difficulty"
            or "difficulty" in target_lower
        ):

            if not value:

                return """

Please tell me the new difficulty level.

"""

            st.session_state.project[
                "difficulty"
            ] = value

            return (
                f"✅ Difficulty updated to **{value}**.\n\n"
                "The Project Blueprint has been updated."
            )


        # ----------------------------------------------------
        # SOLUTION
        # ----------------------------------------------------

        if (
            target_lower == "solution"
            or "solution" in target_lower
        ):

            if not value:

                return """

Please tell me the new solution.

"""

            st.session_state.project[
                "solution"
            ] = value

            return (
                "✅ Project solution updated successfully.\n\n"
                "The Project Blueprint has been updated."
            )


        # ----------------------------------------------------
        # PROBLEM
        # ----------------------------------------------------

        if (
            target_lower == "problem"
            or "problem" in target_lower
        ):

            if not value:

                return """

Please tell me the new problem statement.

"""

            st.session_state.project[
                "problem"
            ] = value

            return (
                "✅ Problem statement updated successfully.\n\n"
                "The Project Blueprint has been updated."
            )


        return """

I understood that you want to change the project,

but I couldn't identify which project field should change.

Try:

"Change the target audience to college students."

or

"Change the platform to web application."

or

"Change the difficulty to Advanced."

"""


    # ========================================================
    # MODIFY FEATURE
    # ========================================================

    if intent == "modify_feature":

        if not target or not value:

            return """

Please tell me which feature you want to change
and what you want to change it to.

"""

        existing = find_feature(
            target
        )

        if existing:

            index = st.session_state.project[
                "features"
            ].index(existing)

            # Generate explanation for modified feature
            prompt = f"""

CURRENT PROJECT:

{project_to_text()}

UPDATED FEATURE:

{value}

Write a concise description explaining:

1. What the updated feature does.

2. How it works in this project.

3. Why it is useful.

Return only the description.

"""

            description = ollama_chat(
                prompt,
                """

You are a project architecture advisor.

Describe the modified feature specifically
for the current project.

Return only the feature description.

"""
            )

            if description.startswith(
                "GROQ_ERROR:"
            ):

                description = (
                    "Provides the requested functionality "
                    "within the project."
                )

            description = description.strip()

            new_feature = (
                f"{value} — {description}"
            )

            st.session_state.project[
                "features"
            ][index] = new_feature

            return (
                f"✅ Changed **{existing}** "
                f"to **{value}**.\n\n"
                f"**What it does:** {description}\n\n"
                "The Blueprint has been updated."
            )

        return (
            f"I couldn't find **{target}** "
            "in the current project."
        )


    # ========================================================
    # REJECT / DON'T LIKE FEATURE
    # ========================================================

    if intent == "reject":

        if target:

            return (
                f"I understand that you don't like "
                f"**{target}**.\n\n"
                "I haven't removed it yet. "
                "If you want, say **'remove it'** or "
                "**'replace it with something better'**."
            )

        return """

I understand. Tell me which feature you don't like,
and I can remove or replace it.

"""


    # ========================================================
    # RAG CONTEXT
    # ========================================================

    documents, metadatas = retrieve_context(
        f"""

{user_message}

Current project:

{project_to_text()}

""",
        top_k=4
    )

    rag_context = "\n\n".join(
        documents
    )


    # ========================================================
    # NORMAL PROJECT CHAT
    # ========================================================

    system_prompt = """

You are IdeaForge AI.

You are having a continuous conversation about ONE
specific project.

Always use the current project as your context.

Do NOT change the project during a normal question.

Do NOT invent unrelated features.

If the project is a music playlist system,
talk about music playlists.

If the project is financial,
talk about financial use cases.

If the user asks about RAG,
explain how RAG works specifically for THIS project.

If the user asks about models,
recommend models specifically suitable for THIS project.

If the user asks for improvements,
suggest improvements relevant to THIS project.

If the user asks for alternatives,
give genuinely different alternatives related
to THIS project's domain.

Be technically accurate and student-friendly.

"""

    prompt = f"""

CURRENT PROJECT:

{project_to_text()}

RECENT CONVERSATION:

{recent_chat_text()}

RELEVANT RAG KNOWLEDGE:

{rag_context}

USER MESSAGE:

{user_message}

INTENT:

{intent}

Answer the user's question clearly.

Do not rewrite the entire project unless the user
specifically asks for the full structure.

"""

    return ollama_chat(
        prompt,
        system_prompt
    )


# ============================================================
# PDF GENERATION
# ============================================================

def generate_pdf():

    project = st.session_state.project

    buffer = io.BytesIO()

    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "CustomTitle",
        parent=styles["Title"],
        alignment=TA_CENTER,
        fontSize=22,
        spaceAfter=20
    )

    heading_style = ParagraphStyle(
        "CustomHeading",
        parent=styles["Heading2"],
        fontSize=15,
        spaceBefore=14,
        spaceAfter=8
    )

    normal_style = ParagraphStyle(
        "CustomNormal",
        parent=styles["BodyText"],
        fontSize=10,
        leading=15
    )

    story = []

    story.append(
        Paragraph(
            "IdeaForge AI — Project Blueprint",
            title_style
        )
    )

    story.append(
        Paragraph(
            "Generated on "
            + datetime.now().strftime(
                "%d %B %Y"
            ),
            normal_style
        )
    )

    story.append(
        Spacer(1, 15)
    )

    sections = [

        (
            "Project Name",
            project["name"]
        ),

        (
            "Domain",
            project["domain"]
        ),

        (
            "Problem Statement",
            project["problem"]
        ),

        (
            "Proposed Solution",
            project["solution"]
        ),

        (
            "Target Users",
            project["target_users"]
        ),

        (
            "Platform",
            project["platform"]
        ),

        (
            "Difficulty",
            project["difficulty"]
        ),

        (
            "Feasibility",
            project["feasibility"]
        ),

        (
            "RAG",
            project["rag"]
        ),

        (
            "Database",
            project["database"]
        ),

        (
            "Reasoning",
            project["reasoning"]
        )

    ]

    for heading, content in sections:

        story.append(
            Paragraph(
                heading,
                heading_style
            )
        )

        safe_content = str(
            content
        ).replace(
            "\n",
            "<br/>"
        )

        story.append(
            Paragraph(
                safe_content,
                normal_style
            )
        )

    list_sections = [

        (
            "AI Features",
            project["ai_features"]
        ),

        (
            "Core Features",
            project["features"]
        ),

        (
            "Technologies",
            project["technologies"]
        ),

        (
            "AI Models",
            project["models"]
        ),

        (
            "Data Requirements",
            project["data_requirements"]
        ),

        (
            "Modules",
            project["modules"]
        ),

        (
            "Workflow",
            project["workflow"]
        ),

        (
            "Suggested Improvements",
            project["improvements"]
        ),

        (
            "Alternative Ideas",
            project["alternatives"]
        ),

        (
            "Required Skills",
            project["skills"]
        )

    ]

    for heading, items in list_sections:

        story.append(
            Paragraph(
                heading,
                heading_style
            )
        )

        if items:

            table_data = []

            for index, item in enumerate(
                items,
                1
            ):

                table_data.append(
                    [
                        str(index),
                        str(item)
                    ]
                )

            table = Table(
                table_data,
                colWidths=[
                    35,
                    470
                ]
            )

            table.setStyle(
                TableStyle(
                    [

                        (
                            "GRID",
                            (0, 0),
                            (-1, -1),
                            0.5,
                            colors.grey
                        ),

                        (
                            "VALIGN",
                            (0, 0),
                            (-1, -1),
                            "TOP"
                        ),

                        (
                            "FONT_SIZE",
                            (0, 0),
                            (-1, -1),
                            9
                        ),

                        (
                            "TOPPADDING",
                            (0, 0),
                            (-1, -1),
                            6
                        ),

                        (
                            "BOTTOMPADDING",
                            (0, 0),
                            (-1, -1),
                            6
                        )

                    ]
                )
            )

            story.append(
                table
            )

    story.append(
        Spacer(1, 20)
    )

    story.append(
        Paragraph(
            "Generated by IdeaForge AI",
            normal_style
        )
    )

    document.build(
        story
    )

    buffer.seek(0)

    return buffer


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">💡 IdeaForge AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Your AI project development advisor — '
    'turn your idea into a useful, feasible and professional project.'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ Project Controls")


    # ========================================================
    # NEW PROJECT
    # ========================================================

    if st.button(
        "🆕 New Project",
        use_container_width=True
    ):

        st.session_state.project = empty_project()

        st.session_state.chat_history = []

        st.session_state.locked = False

        st.session_state.analysis_done = False

        st.session_state.rag_sources = []

        st.session_state.last_analysis = ""

        st.rerun()


    st.divider()


    # ========================================================
    # LOCK / UNLOCK
    # ========================================================

    if st.session_state.locked:

        st.success(
            "🔒 Project Locked"
        )

        if st.button(
            "🔓 Unlock Project",
            use_container_width=True
        ):

            st.session_state.locked = False

            st.rerun()

    else:

        st.info(
            "✏️ Project Editable"
        )


    st.divider()

    st.caption(
        "LLM: Groq — openai/gpt-oss-120b"
    )

    st.caption(
        f"Embeddings: {EMBED_MODEL}"
    )

    st.caption(
        "RAG: ChromaDB"
    )


# ============================================================
# TABS
# ============================================================

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "💡 Find a Project",
        "💬 Project Chat",
        "📋 Project Blueprint",
        "📄 Final PDF"
    ]
)


# ============================================================
# TAB 1
# ============================================================

with tab1:

    st.header(
        "🚀 Find a Project That Fits You"
    )

    skills = st.text_input(
        "Your skills",
        placeholder=(
            "Python, ML, NLP, Streamlit..."
        )
    )

    interests = st.text_input(
        "Your interests",
        placeholder=(
            "AI, music, finance, education..."
        )
    )

    difficulty = st.selectbox(
        "Preferred difficulty",
        [
            "Beginner",
            "Intermediate",
            "Advanced"
        ]
    )

    idea = st.text_area(
        "💡 Your project idea",
        height=150,
        placeholder=(
            "Example: I want to build an AI-powered "
            "music playlist manager..."
        )
    )

    if st.button(
        "🔍 Analyze My Idea",
        type="primary",
        use_container_width=True
    ):

        if not idea.strip():

            st.warning(
                "Please enter your project idea."
            )

        else:

            with st.spinner(
                "Understanding your idea..."
            ):

                success = analyze_project(
                    idea,
                    skills,
                    interests,
                    difficulty
                )

            if success:

                st.session_state.chat_history = []

                st.session_state.locked = False

                st.success(
                    "Project analyzed successfully! 🎉"
                )

                st.rerun()


# ============================================================
# TAB 2 — CHAT
# ============================================================

with tab2:

    st.header(
        "💬 Talk to IdeaForge"
    )

    if not st.session_state.analysis_done:

        st.info(
            "Analyze a project first. "
            "Then you can continuously discuss it with IdeaForge."
        )

    else:

        if st.session_state.locked:

            st.markdown(
                '<div class="locked">'
                '🔒 <b>This project is finalized.</b><br>'
                'The project state is locked.'
                '</div>',
                unsafe_allow_html=True
            )

        else:

            st.markdown(
                '<div class="project-card">'
                '<b>Current Project:</b> '
                + st.session_state.project["name"]
                + '</div>',
                unsafe_allow_html=True
            )


        # ====================================================
        # CHAT HISTORY
        # ====================================================

        for message in st.session_state.chat_history:

            if message["role"] == "user":

                st.markdown(
                    f"""
                    <div class="chat-user">
                    👤 <b>You:</b><br>
                    {message["content"]}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            else:

                st.markdown(
                    f"""
                    <div class="chat-ai">
                    💡 <b>IdeaForge:</b><br>
                    {message["content"]}
                    </div>
                    """,
                    unsafe_allow_html=True
                )


        # ====================================================
        # CHAT INPUT
        # ====================================================

        user_message = st.chat_input(
            "Ask anything about your project..."
        )

        if user_message:

            st.session_state.chat_history.append(
                {
                    "role": "user",
                    "content": user_message
                }
            )

            with st.spinner(
                "IdeaForge is thinking..."
            ):

                response = process_chat(
                    user_message
                )

            st.session_state.chat_history.append(
                {
                    "role": "assistant",
                    "content": response
                }
            )

            st.rerun()


# ============================================================
# TAB 3 — BLUEPRINT
# ============================================================

with tab3:

    st.header(
        "📋 Your Project Blueprint"
    )

    if not st.session_state.analysis_done:

        st.info(
            "Analyze a project first."
        )

    else:

        project = st.session_state.project

        st.subheader(
            f"💡 {project['name']}"
        )

        st.write(
            f"**Domain:** {project['domain']}"
        )

        st.write(
            f"**Difficulty:** {project['difficulty']}"
        )

        st.write(
            f"**Feasibility:** {project['feasibility']}"
        )

        st.divider()


        # ====================================================
        # PROBLEM
        # ====================================================

        st.subheader(
            "🎯 Problem"
        )

        st.write(
            project["problem"]
        )


        # ====================================================
        # SOLUTION
        # ====================================================

        st.subheader(
            "💡 Solution"
        )

        st.write(
            project["solution"]
        )


        # ====================================================
        # TARGET USERS
        # ====================================================

        st.subheader("👥 Target Users")
        st.write(st.session_state.project["target_users"])

        # ====================================================
        # AI FEATURES
        # ====================================================

        st.subheader(
            "🤖 AI Features"
        )

        for item in project["ai_features"]:

            st.markdown(
                f"- {item}"
            )


        # ====================================================
        # CORE FEATURES
        # ====================================================

        st.subheader(
            "✨ Core Features"
        )

        for item in project["features"]:

            st.markdown(
                f"- {item}"
            )


        # ====================================================
        # AI MODELS
        # ====================================================

        st.subheader(
            "🧠 AI Models"
        )

        for item in project["models"]:

            st.markdown(
                f"- {item}"
            )


        # ====================================================
        # RAG
        # ====================================================

        st.subheader(
            "🔎 RAG"
        )

        st.write(
            project["rag"]
        )


        # ====================================================
        # DATABASE
        # ====================================================

        st.subheader(
            "🗄️ Database"
        )

        st.write(
            project["database"]
        )


        # ====================================================
        # TECHNOLOGY
        # ====================================================

        st.subheader(
            "🛠️ Technology Stack"
        )

        for item in project["technologies"]:

            st.markdown(
                f"- {item}"
            )


        # ====================================================
        # MODULES
        # ====================================================

        st.subheader(
            "📦 Modules"
        )

        for item in project["modules"]:

            st.markdown(
                f"- {item}"
            )


        # ====================================================
        # WORKFLOW
        # ====================================================

        st.subheader(
            "🔄 Workflow"
        )

        if project["workflow"]:

            st.code(
                " → ".join(
                    project["workflow"]
                )
            )


        # ====================================================
        # IMPROVEMENTS
        # ====================================================

        st.subheader(
            "📈 Suggested Improvements"
        )

        for item in project["improvements"]:

            st.markdown(
                f"- {item}"
            )


        # ====================================================
        # ALTERNATIVES
        # ====================================================

        st.subheader(
            "🔀 Alternatives"
        )

        for item in project["alternatives"]:

            st.markdown(
                f"- {item}"
            )


        st.divider()


        # ====================================================
        # FINALIZE
        # ====================================================

        if not st.session_state.locked:

            if st.button(
                "🔒 Finalize & Lock Project",
                type="primary",
                use_container_width=True
            ):

                st.session_state.locked = True

                st.success(
                    "Project finalized and locked!"
                )

                st.rerun()

        else:

            st.success(
                "🔒 This project is finalized."
            )


# ============================================================
# TAB 4 — PDF
# ============================================================

with tab4:

    st.header(
        "📄 Final Professional Blueprint"
    )

    if not st.session_state.analysis_done:

        st.info(
            "Analyze a project first."
        )

    elif not st.session_state.locked:

        st.warning(
            "Please finalize and lock the project "
            "before generating the PDF."
        )

    else:

        st.success(
            "Your project is finalized and ready for export. ✅"
        )

        if st.button(
            "📄 Generate PDF",
            type="primary",
            use_container_width=True
        ):

            with st.spinner(
                "Generating professional PDF..."
            ):

                pdf_file = generate_pdf()

            st.download_button(
                label=(
                    "⬇️ Download Project Blueprint PDF"
                ),
                data=pdf_file,
                file_name=(
                    "IdeaForge_Project_Blueprint.pdf"
                ),
                mime="application/pdf",
                use_container_width=True
            )
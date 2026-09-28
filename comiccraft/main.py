import os
import time
import requests

from concurrent.futures import ThreadPoolExecutor, as_completed

from fpdf import FPDF
from pydantic import BaseModel
from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from google import genai


# =========================
# DATA MODELS
# =========================

class Panel(BaseModel):
    panel: int
    description: str
    narration: str
    dialogue: str


class ComicStory(BaseModel):
    character_profile: str
    panels: list[Panel]


# =========================
# APP SETUP
# =========================

app = FastAPI()

templates = Jinja2Templates(
    directory="templates"
)


# =========================
# STATIC FILES
# =========================

os.makedirs(
    "static/generated",
    exist_ok=True
)

app.mount(
    "/static",
    StaticFiles(directory="static"),
    name="static"
)


# =========================
# GEMINI STORY GENERATION
# =========================

story_api_key = os.getenv(
    "GEMINI_API_KEY"
)

if not story_api_key:
    raise RuntimeError(
        "GEMINI_API_KEY is missing"
    )

story_client = genai.Client(
    api_key=story_api_key
)


def generate_story(
    prompt: str,
    character: str,
    setting: str,
    tone: str,
    style: str
):

    comic_prompt = f"""
Create a coherent 5-panel comic story.

USER INPUT
==========

Story idea:
{prompt}

Main character:
{character}

Setting:
{setting}

Story tone:
{tone}

Art style:
{style}


CHARACTER CONSISTENCY
=====================

First, create a fixed character profile for the main
character.

The character profile must describe stable visual features
that should remain unchanged throughout the entire comic.

Include things such as:
- approximate age
- hair
- face
- clothing
- colors
- body proportions
- distinctive features
- important accessories

Do NOT change the character's appearance between panels
unless the story explicitly requires a change.

The character profile will be reused when generating every
image.


STORY CONTINUITY
================

Create exactly 5 panels.

The panels must form ONE continuous story.

Panel 2 must logically follow Panel 1.
Panel 3 must logically follow Panel 2.
Panel 4 must logically follow Panel 3.
Panel 5 must logically conclude the story.

Maintain continuity of:

- character appearance
- character identity
- location
- important objects
- events
- actions
- time progression
- cause and effect

Do not introduce unexplained changes.

If an object or event is introduced in an earlier panel,
remember it in later panels when it is relevant.

Do not make each panel feel like an unrelated scene.


PANEL CONTENT
=============

For each panel provide:

- Panel description
- Narration
- Dialogue

The panel description should describe exactly what should
be visible in that panel.

Narration and dialogue should support the actual events
shown in the panel.

Keep the story understandable from Panel 1 through 5.


IMPORTANT
=========

Return exactly 5 panels.

Do not create additional panels.

Do not change the main character's identity.

Do not contradict events from earlier panels.
"""


    interaction = story_client.interactions.create(
    model="gemini-3.5-flash",
    input=comic_prompt,
    generation_config={
        "thinking_level": "minimal"
    },
    response_format={
        "type": "text",
        "mime_type": "application/json",
        "schema": ComicStory.model_json_schema()
    }
)

    comic_story = ComicStory.model_validate_json(
        interaction.output_text
    )

    return comic_story


# =========================
# POLLINATIONS IMAGE GENERATION
# =========================

def generate_image(prompt: str):

    api_key = os.getenv(
        "POLLINATIONS_API_KEY"
    )

    if not api_key:
        raise RuntimeError(
            "POLLINATIONS_API_KEY is missing"
        )


    url = (
        "https://gen.pollinations.ai/image/"
        + requests.utils.quote(prompt)
    )


    last_error = None


    for attempt in range(3):

        try:

            response = requests.get(
                url,

                headers={
                    "Authorization":
                        f"Bearer {api_key}"
                },

                params={
                    "model": "flux",
                    "width": 768,
                    "height": 768
                },

                timeout=120
            )


            response.raise_for_status()

            return response.content


        except (
            requests.exceptions.ConnectionError,
            requests.exceptions.Timeout,
            requests.exceptions.SSLError,
            requests.exceptions.HTTPError
        ) as error:

            last_error = error

            if attempt < 2:
                time.sleep(5)


    raise RuntimeError(
        "Pollinations image generation failed "
        f"after 3 attempts: {last_error}"
    )


# =========================
# PDF GENERATION
# =========================

def clean_pdf_text(text):

    return (
        str(text)
        .encode(
            "latin-1",
            "replace"
        )
        .decode("latin-1")
    )


def create_pdf(panels, pdf_path):

    pdf = FPDF()

    pdf.set_auto_page_break(
        auto=True,
        margin=15
    )


    for panel in panels:

        pdf.add_page()


        # -------------------------
        # PANEL TITLE
        # -------------------------

        pdf.set_font(
            "Arial",
            "B",
            20
        )

        pdf.cell(
            0,
            12,
            f"Panel {panel['panel']}",
            ln=True,
            align="C"
        )

        pdf.ln(4)


        # -------------------------
        # COMIC IMAGE
        # -------------------------

        pdf.image(
            panel["filepath"],
            x=20,
            y=28,
            w=170
        )


        pdf.set_y(203)


        # -------------------------
        # NARRATION
        # -------------------------

        pdf.set_font(
            "Arial",
            "B",
            14
        )

        pdf.cell(
            0,
            9,
            "NARRATION",
            ln=True
        )


        pdf.set_font(
            "Arial",
            "",
            13
        )


        narration = clean_pdf_text(
            panel["narration"]
        )


        pdf.multi_cell(
            0,
            7,
            narration
        )


        pdf.ln(5)


        # -------------------------
        # DIALOGUE
        # -------------------------

        pdf.set_font(
            "Arial",
            "B",
            14
        )

        pdf.cell(
            0,
            9,
            "DIALOGUE",
            ln=True
        )


        pdf.set_font(
            "Arial",
            "",
            13
        )


        dialogue = clean_pdf_text(
            panel["dialogue"]
        )


        pdf.multi_cell(
            0,
            7,
            dialogue
        )


    pdf.output(
        pdf_path
    )


# =========================
# HOME PAGE
# =========================

@app.get("/")
def home(request: Request):

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={}
    )


# =========================
# IMAGE TASK
# =========================

def generate_panel_image(
    panel,
    character_profile,
    setting,
    style
):

    # -------------------------
    # STYLE INSTRUCTION
    # -------------------------

    if style.lower() in [
        "realistic",
        "realistic style",
        "photorealistic"
    ]:

        style_instruction = """
Photorealistic cinematic photography,
natural realistic lighting,
realistic textures and materials,
natural human proportions,
detailed real-world environment,
highly realistic visual appearance.
"""

    else:

        style_instruction = style


    # -------------------------
    # IMAGE PROMPT
    # -------------------------

    image_prompt = f"""
Create a single comic panel image.

VISUAL STYLE
============

{style_instruction}


CHARACTER CONSISTENCY
=====================

The following is the FIXED character profile.

Use this same character appearance in every panel:

{character_profile}

The character must remain visually consistent.

Do not randomly change:

- face
- hair
- hairstyle
- clothing
- clothing colors
- body proportions
- age
- accessories
- distinctive features

The character's appearance should match the character
profile exactly as closely as possible.


SETTING
=======

{setting}


CURRENT SCENE
=============

{panel.description}


STORY CONTEXT
=============

This is panel {panel.panel} of a 5-panel continuous story.

The scene must logically fit the previous and following
events of the story.

Do not introduce unrelated characters, objects or locations
unless they are required by the scene.


IMAGE REQUIREMENTS
==================

Create only the visual scene.

Do not add:

- speech bubbles
- captions
- narration text
- written dialogue
- watermarks
- random text

The image should visually represent the current scene.
"""


    print(
        f"Starting image generation for panel {panel.panel}..."
    )


    image = generate_image(
        image_prompt
    )


    filename = (
        f"panel_{panel.panel}.png"
    )


    filepath = os.path.join(
        "static",
        "generated",
        filename
    )


    with open(
        filepath,
        "wb"
    ) as file:

        file.write(image)


    print(
        f"Panel {panel.panel} image completed."
    )


    return {

        "panel":
            panel.panel,

        "description":
            panel.description,

        "narration":
            panel.narration,

        "dialogue":
            panel.dialogue,

        "image":
            f"/static/generated/{filename}",

        "filepath":
            filepath

    }


# =========================
# COMIC GENERATION
# =========================

@app.post("/generate")
def create_comic(

    prompt: str = Form(...),

    character: str = Form(...),

    setting: str = Form(...),

    tone: str = Form(...),

    style: str = Form(...)

):


    # =========================
    # STEP 1
    # GENERATE STORY
    # =========================

    print("Generating story with Gemini...")

    story = generate_story(
        prompt,
        character,
        setting,
        tone,
        style
    )

    print("Story generated successfully.")


    # Fixed character profile generated by Gemini.

    character_profile = (
        story.character_profile
    )


    # =========================
    # STEP 2
    # PREPARE IMAGE TASKS
    # =========================

    print(
        "Starting concurrent image generation..."
    )


    panels = []


    # =========================
    # STEP 3
    # GENERATE IMAGES
    # =========================

    # Three images are generated at the
    # same time instead of one by one.

    with ThreadPoolExecutor(
        max_workers=3
    ) as executor:

        futures = [

            executor.submit(
                generate_panel_image,
                panel,
                character_profile,
                setting,
                style
            )

            for panel in story.panels
        ]


        for future in as_completed(futures):

            panel_result = future.result()

            panels.append(
                panel_result
            )


    # as_completed() returns panels in the
    # order they finish, not panel number.
    #
    # Sort them back into comic order.

    panels.sort(
        key=lambda panel:
            panel["panel"]
    )


    print(
        "All images generated successfully."
    )


    # =========================
    # STEP 4
    # CREATE PDF
    # =========================

    pdf_path = os.path.join(
        "static",
        "generated",
        "comic.pdf"
    )


    create_pdf(
        panels,
        pdf_path
    )


    print(
        "PDF created successfully."
    )


    # =========================
    # STEP 5
    # RETURN RESULT
    # =========================

    return {

        "comic": {
            "panels": panels
        },

        "panels":
            panels,

        "pdf":
            "/static/generated/comic.pdf"

    }

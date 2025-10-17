# app.py  -- fixed: persistent generated-image rendering so Edit button works after generation
import os
import re
import uuid
import datetime
from io import BytesIO
import streamlit as st
from PIL import Image

# Lazy-import VertexAI so the app doesn't break at import time if SDK isn't installed.
try:
    import vertexai
    from vertexai.preview.vision_models import ImageGenerationModel
    from vertexai.generative_models import GenerativeModel, Part
    from google.oauth2 import service_account
    VERTEX_AVAILABLE = True
except Exception:
    VERTEX_AVAILABLE = False

# ---------------- Page config ----------------
st.set_page_config(page_title="AI Image Generator + Editor", layout="wide")
st.title("AI Image Generator + Editor")

# ---------------- Safe session initialization ----------------
def safe_init_session():
    try:
        _ = st.session_state
    except RuntimeError:
        return False
    st.session_state.setdefault("generated_images", [])   # list of {"filename","content","key"}
    st.session_state.setdefault("edited_images", [])      # list of {"original","edited","prompt","filename"}
    st.session_state.setdefault("edit_image_bytes", None) # currently loaded image bytes for editing
    st.session_state.setdefault("edit_image_name", "")
    return True

safe_init_session()

# ---------------- Prompt templates and style map ----------------
PROMPT_TEMPLATES = {

    "None": """
Dont make any changes in the user's prompt.Follow it as it is
User’s raw prompt:
"{USER_PROMPT}"

Refined general image prompt:
""",

    "General": """
You are an expert AI prompt engineer specialized in creating vivid and descriptive image prompts.

Your job:
- Expand the user’s input into a detailed, clear prompt for an image generation model.
- Add missing details such as:
  • Background and setting
  • Lighting and mood
  • Style and realism level
  • Perspective and composition

Rules:
- Stay true to the user’s intent.
- Keep language concise, descriptive, and expressive.
- Output only the final refined image prompt.

User’s raw prompt:
"{USER_PROMPT}"

Refined general image prompt:
""",

    "Design": """
You are a senior AI prompt engineer supporting a creative design team.

Your job:
- Expand raw input into a visually inspiring, design-oriented image prompt.
- Add imaginative details about:
  • Artistic styles (minimalist, abstract, futuristic, flat, 3D render, watercolor, digital illustration)
  • Color schemes, palettes, textures, and patterns
  • Composition and balance (symmetry, negative space, creative framing)
  • Lighting and atmosphere (soft glow, vibrant contrast, surreal shading)
  • Perspective (isometric, top-down, wide shot, close-up)

Rules:
- Keep fidelity to the idea but make it highly creative and visually unique.
- Output only the final refined image prompt.

User’s raw prompt:
"{USER_PROMPT}"

Refined design image prompt:
""",
    "Marketing": """
You are a senior AI prompt engineer creating polished prompts for marketing and advertising visuals.

Task:
- Take the user’s raw input and turn it into a polished, professional, campaign-ready image prompt.
- Expand the idea with rich marketing-oriented details that make it visually persuasive.

When refining, include:
- Background & setting (modern, lifestyle, commercial, aspirational)
- Lighting & atmosphere (studio lights, golden hour, cinematic)
- Style (photorealistic, cinematic, product photography, lifestyle branding)
- Perspective & composition (wide shot, close-up, dramatic angles)
- Mood, tone & branding suitability (premium, sleek, aspirational)

Special Brand Rule:
- If the user asks for an image related to a specific brand, seamlessly add the brand’s tagline into the final image prompt.
- For **Dr. Reddy’s**, the correct tagline is: “Good Health Can’t Wait.”

Rules:
- Stay faithful to the user’s idea but elevate it for use in ads, social media, or presentations.
- Output **only** the final refined image prompt (no explanations, no extra text).

User raw input:
{USER_PROMPT}


Refined marketing image prompt:
""",
    "DPEX": """
You are a senior AI prompt engineer creating refined prompts for IT and technology-related visuals.

Your job:
- Transform the raw input into a detailed, professional, and technology-focused image prompt.
- Expand with contextual details about:
  • Technology environments (server rooms, data centers, cloud systems, coding workspaces)
  • Digital elements (network diagrams, futuristic UIs, holograms, cybersecurity visuals)
  • People in IT roles (developers, engineers, admins, tech support, collaboration)
  • Tone (innovative, technical, futuristic, professional)
  • Composition (screens, servers, code on monitors, abstract digital patterns)
  • Lighting and effects (LED glow, cyberpunk tones, neon highlights, modern tech ambiance)

Rules:
- Ensure images are suitable for IT presentations, product demos, training, technical documentation, and digital transformation campaigns.
- Stay true to the user’s intent but emphasize a technological and innovative look.
- Output only the final refined image prompt.

User’s raw prompt:
"{USER_PROMPT}"

Refined DPEX image prompt:
""",

    "HR": """
You are a senior AI prompt engineer creating refined prompts for human resources and workplace-related visuals.

Your job:
- Transform the raw input into a detailed, professional, and HR-focused image prompt.
- Expand with contextual details about:
  • Workplace settings (modern office, meeting rooms, open workspaces, onboarding sessions)
  • People interactions (interviews, teamwork, training, collaboration, diversity and inclusion)
  • Themes (employee engagement, professional growth, recruitment, performance evaluation)
  • Composition (groups in discussion, managers mentoring, collaborative workshops)
  • Lighting and tone (bright, welcoming, professional, inclusive)

Rules:
- Ensure images are suitable for HR presentations, recruitment campaigns, internal training, or employee engagement material.
- Stay true to the user’s intent but emphasize people, culture, and workplace positivity.
- Output only the final refined image prompt.

User’s raw prompt:
"{USER_PROMPT}"

Refined HR image prompt:
""",

    "Business": """
You are a senior AI prompt engineer creating refined prompts for business and corporate visuals.

Your job:
- Transform the raw input into a detailed, professional, and business-oriented image prompt.
- Expand with contextual details about:
  • Corporate settings (boardrooms, skyscrapers, modern offices, networking events)
  • Business activities (presentations, negotiations, brainstorming sessions, teamwork)
  • People (executives, entrepreneurs, consultants, diverse teams, global collaboration)
  • Tone (professional, ambitious, strategic, innovative)
  • Composition (formal meetings, handshake deals, conference tables, city skyline backgrounds)
  • Lighting and atmosphere (clean, modern, premium, professional)

Rules:
- Ensure images are suitable for corporate branding, investor decks, strategy sessions, or professional reports.
- Stay true to the user’s intent but emphasize professionalism, ambition, and success.
- Output only the final refined image prompt.

User’s raw prompt:
"{USER_PROMPT}"

Refined business image prompt:
"""
}


STYLE_DESCRIPTIONS = {
    "None": "No special styling — keep the image natural, faithful to the user’s idea.",
    "Smart": "A clean, balanced, and polished look. Professional yet neutral, visually appealing without strong artistic bias.",
    "Cinematic": "Film-style composition with professional lighting. Wide dynamic range, dramatic highlights, storytelling feel.",
    "Creative": "Playful, imaginative, and experimental. Bold artistic choices, unexpected elements, and expressive color use.",
    "Bokeh": "Photography style with shallow depth of field. Subject in sharp focus with soft, dreamy, blurred backgrounds.",
    "Macro": "Extreme close-up photography. High detail, textures visible, shallow focus highlighting minute features.",
    "Illustration": "Hand-drawn or digitally illustrated style. Clear outlines, stylized shading, expressive and artistic.",
    "3D Render": "Photorealistic or stylized CGI. Crisp geometry, depth, shadows, and reflective surfaces with realistic rendering.",
    "Fashion": "High-end editorial photography. Stylish, glamorous poses, bold makeup, controlled lighting, and modern aesthetic.",
    "Minimalist": "Simple and uncluttered. Few elements, large negative space, flat or muted color palette, clean composition.",
    "Moody": "Dark, atmospheric, and emotional. Strong shadows, high contrast, deep tones, cinematic ambiance.",
    "Portrait": "Focus on the subject. Natural skin tones, shallow depth of field, close-up or waist-up framing, studio or natural lighting.",
    "Stock Photo": "Professional, commercial-quality photo. Neutral subject matter, polished composition, business-friendly aesthetic.",
    "Vibrant": "Bold, saturated colors. High contrast, energetic mood, eye-catching and lively presentation.",
    "Pop Art": "Comic-book and pop-art inspired. Bold outlines, halftone patterns, flat vivid colors, high contrast, playful tone.",
    "Vector": "Flat vector graphics. Smooth shapes, sharp edges, solid fills, and clean scalable style like logos or icons.",

    "Watercolor": "Soft, fluid strokes with delicate blending and washed-out textures. Artistic and dreamy.",
    "Oil Painting": "Rich, textured brushstrokes. Classic fine art look with deep color blending.",
    "Charcoal": "Rough, sketchy textures with dark shading. Artistic, raw, dramatic effect.",
    "Line Art": "Minimal monochrome outlines with clean, bold strokes. No shading, focus on form.",

    "Anime": "Japanese animation style with vibrant colors, clean outlines, expressive features, and stylized proportions.",
    "Cartoon": "Playful, exaggerated features, simplified shapes, bold outlines, and bright colors.",
    "Pixel Art": "Retro digital art style. Small, pixel-based visuals resembling old-school video games.",

    "Fantasy Art": "Epic fantasy scenes. Magical elements, mythical creatures, enchanted landscapes.",
    "Surreal": "Dreamlike, bizarre imagery. Juxtaposes unexpected elements, bending reality.",
    "Concept Art": "Imaginative, detailed artwork for games or films. Often moody and cinematic.",

    "Cyberpunk": "Futuristic neon city vibes. High contrast, glowing lights, dark tones, sci-fi feel.",
    "Steampunk": "Retro-futuristic style with gears, brass, Victorian aesthetics, and industrial design.",
    "Neon Glow": "Bright neon outlines and glowing highlights. Futuristic, nightlife aesthetic.",
    "Low Poly": "Simplified 3D style using flat geometric shapes and polygons.",
    "Isometric": "3D look with isometric perspective. Often used for architecture, games, and diagrams.",

    "Vintage": "Old-school, retro tones. Faded colors, film grain, sepia, or retro print feel.",
    "Graffiti": "Urban street art style with bold colors, spray paint textures, and rebellious tone."
}

# ---------------- Helpers ----------------
def sanitize_prompt(text: str) -> str:
    """Strip headings, numbered options and labels often produced by a prompt refiner."""
    if not text:
        return text
    lines = []
    for line in text.splitlines():
        ln = line.strip()
        if not ln:
            continue
        # Skip lines that start with 'Option', 'Key', 'Apply', 'Specificity', 'Keywords', etc.
        if re.match(r'^(Option|Key|Apply|Specificity|Keywords)\b', ln, re.I):
            continue
        if re.match(r'^\d+[\.\)]\s*', ln):
            continue
        # Skip short header-like lines that end with colon (e.g., "Key Improvements:")
        if len(ln) < 80 and ln.endswith(':'):
            continue
        # Skip lines that look like list bullets (e.g., starting with '-')
        if ln.startswith('-') or ln.startswith('*'):
            continue
        lines.append(ln)
    cleaned = ' '.join(lines)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned or text

def safe_get_enhanced_text(resp):
    if resp is None:
        return ""
    if hasattr(resp, "text") and resp.text:
        return resp.text
    if hasattr(resp, "candidates") and resp.candidates:
        try:
            return resp.candidates[0].content.parts[0].text
        except Exception:
            pass
    return str(resp)

def get_image_bytes_from_genobj(gen_obj):
    if isinstance(gen_obj, (bytes, bytearray)):
        return bytes(gen_obj)
    for attr in ("image_bytes", "_image_bytes"):
        if hasattr(gen_obj, attr):
            return getattr(gen_obj, attr)
    if hasattr(gen_obj, "image") and gen_obj.image:
        for attr in ("image_bytes", "_image_bytes"):
            if hasattr(gen_obj.image, attr):
                return getattr(gen_obj.image, attr)
    return None

def show_image_safe(image_source, caption="Image"):
    try:
        if isinstance(image_source, (bytes, bytearray)):
            st.image(image_source, caption=caption, use_container_width=True)
        elif isinstance(image_source, Image.Image):
            st.image(image_source, caption=caption, use_container_width=True)
        else:
            st.image(Image.open(BytesIO(image_source)), caption=caption, use_container_width=True)
    except TypeError:
        # fallback for older Streamlit
        if isinstance(image_source, (bytes, bytearray)):
            st.image(image_source, caption=caption, use_column_width=True)
        else:
            st.image(Image.open(BytesIO(image_source)), caption=caption, use_column_width=True)
    except Exception as e:
        st.error(f"Failed to display image: {e}")

# ---------------- Vertex lazy loaders ----------------
MODEL_CACHE = {"imagen": None, "nano": None, "text": None}

def init_vertex(project_id, credentials_info, location="us-central1"):
    if not VERTEX_AVAILABLE:
        return False
    try:
        if getattr(vertexai, "_initialized", False):
            return True
    except Exception:
        pass
    try:
        creds = service_account.Credentials.from_service_account_info(dict(credentials_info))
        vertexai.init(project=project_id, location=location, credentials=creds)
        setattr(vertexai, "_initialized", True)
        return True
    except Exception as e:
        st.error(f"Vertex init failed: {e}")
        return False

def get_imagen_model():
    if MODEL_CACHE["imagen"]:
        return MODEL_CACHE["imagen"]
    if not VERTEX_AVAILABLE:
        return None
    try:
        MODEL_CACHE["imagen"] = ImageGenerationModel.from_pretrained("imagen-4.0-generate-001")
        return MODEL_CACHE["imagen"]
    except Exception as e:
        st.error(f"Failed to load Imagen model: {e}")
        return None

def get_nano_banana_model():
    if MODEL_CACHE["nano"]:
        return MODEL_CACHE["nano"]
    if not VERTEX_AVAILABLE:
        return None
    try:
        MODEL_CACHE["nano"] = GenerativeModel("gemini-2.5-flash-image")
        return MODEL_CACHE["nano"]
    except Exception as e:
        st.error(f"Failed to load Nano Banana model: {e}")
        return None

def get_text_model():
    if MODEL_CACHE["text"]:
        return MODEL_CACHE["text"]
    if not VERTEX_AVAILABLE:
        return None
    try:
        MODEL_CACHE["text"] = GenerativeModel("gemini-2.0-flash")
        return MODEL_CACHE["text"]
    except Exception as e:
        st.error(f"Failed to load text model: {e}")
        return None

# ---------------- Core flows ----------------
def generate_images_from_prompt(prompt, dept="None", style_desc="", n_images=1):
    """Generate images with Imagen. Refine prompt only when dept != 'None' and sanitize result."""
    if not VERTEX_AVAILABLE:
        st.error("VertexAI SDK not available in this environment.")
        return []

    creds = st.secrets.get("gcp_service_account")
    if not creds or not creds.get("project_id"):
        st.error("Missing GCP credentials in Streamlit secrets: 'gcp_service_account'")
        return []

    if not init_vertex(creds["project_id"], creds):
        st.error("Failed to initialize VertexAI.")
        return []

    imagen = get_imagen_model()
    if imagen is None:
        st.error("Imagen model unavailable.")
        return []

    # Only refine when department selected
    enhanced_prompt = prompt
    if dept and dept != "None":
        text_model = get_text_model()
        if text_model:
            try:
                template = PROMPT_TEMPLATES.get(dept, PROMPT_TEMPLATES["General"])
                refinement_input = template.replace("{USER_PROMPT}", prompt)
                if style_desc:
                    refinement_input += f"\n\nApply style: {style_desc}"
                text_resp = text_model.generate_content(refinement_input)
                maybe = safe_get_enhanced_text(text_resp).strip()
                cleaned = sanitize_prompt(maybe)
                if cleaned:
                    enhanced_prompt = cleaned
            except Exception as e:
                # fallback to original prompt
                st.warning(f"Prompt refinement failed, using raw prompt. ({e})")
                enhanced_prompt = prompt

    # call Imagen
    try:
        resp = imagen.generate_images(prompt=enhanced_prompt, number_of_images=n_images)
    except Exception as e:
        st.error(f"Imagen generate_images failed: {e}")
        return []

    out = []
    for i in range(min(n_images, len(resp.images))):
        gen_obj = resp.images[i]
        b = get_image_bytes_from_genobj(gen_obj)
        if b:
            out.append(b)
    return out

def run_edit_flow(edit_prompt, base_bytes):
    """Edit image bytes using Gemini Nano Banana. Returns edited bytes or None."""
    if not VERTEX_AVAILABLE:
        st.error("VertexAI SDK not available in this environment.")
        return None

    creds = st.secrets.get("gcp_service_account")
    if not creds or not creds.get("project_id"):
        st.error("Missing GCP credentials in Streamlit secrets: 'gcp_service_account'")
        return None

    if not init_vertex(creds["project_id"], creds):
        st.error("Failed to initialize VertexAI.")
        return None

    nano = get_nano_banana_model()
    if nano is None:
        st.error("Editor model unavailable.")
        return None

    input_image = Part.from_data(mime_type="image/png", data=base_bytes)
    edit_instruction = f"""
You are a professional AI image editor.
Instructions:
- Take the provided image.
- Apply these edits: {edit_prompt}
- Return the final edited image inline (PNG).
- Do not include any extra text or captions.
"""
    try:
        response = nano.generate_content([edit_instruction, input_image])
    except Exception as e:
        st.error(f"Nano Banana call failed: {e}")
        return None

    for candidate in getattr(response, "candidates", []):
        for part in getattr(candidate.content, "parts", []):
            if hasattr(part, "inline_data") and getattr(part.inline_data, "data", None):
                return part.inline_data.data

    if hasattr(response, "text") and response.text:
        st.warning("Editor returned text instead of an image. See message above.")
    else:
        st.error("Editor returned no inline image.")
    return None

# ---------------- UI (single-flow) ----------------
left_col, right_col = st.columns([3, 1])

with left_col:
   
    # Department selector (re-added)
    dept = st.selectbox("🏢 Department", list(PROMPT_TEMPLATES.keys()), index=0)

    # Style
    style = st.selectbox("🎨 Style ", list(STYLE_DESCRIPTIONS.keys()), index=0)
    style_desc = "" if style == "None" else STYLE_DESCRIPTIONS.get(style, "")

    # Upload an image (optional)
    uploaded_file = st.file_uploader(" ", type=["png","jpg","jpeg","webp"])
    if uploaded_file:
        raw = uploaded_file.read()
        pil = Image.open(BytesIO(raw)).convert("RGB")
        buf = BytesIO()
        pil.save(buf, format="PNG")
        buf_bytes = buf.getvalue()
        st.session_state["edit_image_bytes"] = buf_bytes
        st.session_state["edit_image_name"] = getattr(uploaded_file, "name", f"uploaded_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        show_image_safe(buf_bytes, caption=f"Uploaded: {st.session_state['edit_image_name']}")
    else:
        if st.session_state.get("edit_image_bytes"):
            show_image_safe(st.session_state["edit_image_bytes"], caption=f"Editor loaded: {st.session_state.get('edit_image_name','Selected Image')}")

    # Prompt and number of images
    prompt = st.text_area("Enter prompt ", key="main_prompt", height=140, placeholder="Enter prompt")
    num_images = 1

    # Run button (generation happens here and results are appended to session_state)
    if st.button("Run"):
        prompt_text = (prompt or "").strip()
        if not prompt_text:
            st.warning("Please enter a prompt.")
        else:
            base_image = st.session_state.get("edit_image_bytes")
            if base_image:
                # EDIT FLOW (uploaded image -> Nano Banana)
                with st.spinner("Editing image with Nano Banana..."):
                    edited = run_edit_flow(prompt_text, base_image)
                    if edited:
                        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                        out_fn = f"outputs/edited/edited_{ts}_{uuid.uuid4().hex[:6]}.png"
                        os.makedirs(os.path.dirname(out_fn), exist_ok=True)
                        with open(out_fn, "wb") as f:
                            f.write(edited)
                        st.success("Edited image created below.")
                        show_image_safe(edited, caption=f"Edited ({ts})")
                        st.download_button("⬇️ Download Edited", data=edited, file_name=os.path.basename(out_fn), mime="image/png", key=f"dl_edit_{uuid.uuid4().hex}")
                        st.session_state.edited_images.append({
                            "original": base_image,
                            "edited": edited,
                            "prompt": prompt_text,
                            "filename": out_fn
                        })
                        # load the edited image into editor for potential re-edit
                        st.session_state["edit_image_bytes"] = edited
                        st.session_state["edit_image_name"] = os.path.basename(out_fn)
                    else:
                        st.error("Editing failed or returned no image.")
            else:
                # GENERATION FLOW (Imagen) - append images to session state
                with st.spinner("Generating images with Imagen..."):
                    generated = generate_images_from_prompt(prompt_text, dept=dept, style_desc=style_desc, n_images=num_images)
                    if generated:
                        st.success(f"Generated {len(generated)} image(s).")
                        for i, b in enumerate(generated):
                            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                            fname = f"outputs/generated/gen_{ts}_{i}.png"
                            os.makedirs(os.path.dirname(fname), exist_ok=True)
                            with open(fname, "wb") as f:
                                f.write(b)

                            # create a stable short key for this image
                            short = os.path.basename(fname) + str(i)
                            key_hash = uuid.uuid5(uuid.NAMESPACE_DNS, short).hex[:8]

                            # store generated image with metadata (persistent)
                            entry = {"filename": fname, "content": b, "key": key_hash}
                            st.session_state.generated_images.append(entry)
                    else:
                        st.error("Generation failed or returned no images.")

    # -------------------------
    # Render generated images persistently (outside Run block so buttons survive reruns)
    # -------------------------
    if st.session_state.get("generated_images"):
        st.markdown("### Recently Generated")
        # show the most recent N generated images in the main column
        for entry in reversed(st.session_state.generated_images[-12:]):
            fname = entry.get("filename")
            b = entry.get("content")
            key_hash = entry.get("key") or uuid.uuid5(uuid.NAMESPACE_DNS, os.path.basename(fname)).hex[:8]

            show_image_safe(b, caption=os.path.basename(fname))
            col_dl, col_edit = st.columns([1, 1])
            with col_dl:
                st.download_button(
                    "⬇️ Download",
                    data=b,
                    file_name=os.path.basename(fname),
                    mime="image/png",
                    key=f"dl_gen_{key_hash}"
                )
            with col_edit:
                # this Edit button now reliably works because it's rendered on every run
                if st.button("✏️ Edit this image", key=f"edit_gen_{key_hash}"):
                    st.session_state["edit_image_bytes"] = b
                    st.session_state["edit_image_name"] = os.path.basename(fname)
                    # rerun so the left editor area updates to show the loaded image
                    st.experimental_rerun()


# ---------------- Right column: history + inline editing + re-edit ----------------
with right_col:
    st.subheader("📂 History")

    # Generated images (history column)
    if st.session_state.get("generated_images"):
        st.markdown("### Generated Images")
        for idx, entry in enumerate(reversed(st.session_state.generated_images[-20:])):
            name = os.path.basename(entry.get("filename", f"gen_{idx}.png"))
            content = entry.get("content")
            key_hash = entry.get("key") or uuid.uuid5(uuid.NAMESPACE_DNS, name + str(idx)).hex[:8]
            with st.expander(name):
                show_image_safe(content, caption=name)
                st.download_button("⬇️ Download", data=content, file_name=name, mime="image/png", key=f"hist_dl_{key_hash}")
                if st.button("✏️ Edit this image (load into editor)", key=f"hist_edit_{key_hash}"):
                    st.session_state["edit_image_bytes"] = content
                    st.session_state["edit_image_name"] = name
                    st.experimental_rerun()

                # Inline quick edit
                inline_key = f"inline_prompt_{key_hash}"
                if inline_key not in st.session_state:
                    st.session_state[inline_key] = ""
                st.text_area("Quick edit instructions (inline)", key=inline_key, value=st.session_state[inline_key], height=80)
                if st.button("Edit Inline", key=f"inline_btn_{key_hash}"):
                    ptxt = st.session_state.get(inline_key, "").strip()
                    if not ptxt:
                        st.warning("Enter edit instructions.")
                    else:
                        with st.spinner("Editing image inline..."):
                            edited_bytes = run_edit_flow(ptxt, content)
                            if edited_bytes:
                                ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                                outfn = f"outputs/edited/edited_{ts}_{key_hash}.png"
                                os.makedirs(os.path.dirname(outfn), exist_ok=True)
                                with open(outfn, "wb") as f:
                                    f.write(edited_bytes)
                                st.success("Edited image created.")
                                show_image_safe(edited_bytes, caption=f"Edited {name}")
                                st.download_button("⬇️ Download Edited", data=edited_bytes, file_name=os.path.basename(outfn), mime="image/png", key=f"inline_dl_{uuid.uuid4().hex}")
                                st.session_state.edited_images.append({
                                    "original": content,
                                    "edited": edited_bytes,
                                    "prompt": ptxt,
                                    "filename": outfn
                                })
                            else:
                                st.error("Edit returned no image.")

    # Edited images (re-edit chain)
    if st.session_state.get("edited_images"):
        st.markdown("### Edited Images (re-editable)")
        for idx, entry in enumerate(reversed(st.session_state.edited_images[-40:])):
            fn = entry.get("filename", f"edited_{idx}.png")
            name = os.path.basename(fn)
            edited_bytes = entry.get("edited")
            prompt_prev = entry.get("prompt", "")
            key_hash = uuid.uuid5(uuid.NAMESPACE_DNS, name + str(idx)).hex[:8]
            with st.expander(f"{name} — {prompt_prev[:60]}"):
                col1, col2 = st.columns(2)
                with col1:
                    show_image_safe(entry.get("original"), caption="Original (before this edit)")
                with col2:
                    show_image_safe(edited_bytes, caption="Edited result")

                # re-edit textarea (pre-populate with previous prompt to tweak)
                reedit_key = f"reedit_prompt_{key_hash}"
                if reedit_key not in st.session_state:
                    st.session_state[reedit_key] = prompt_prev or ""
                st.text_area("Re-edit instructions (tweak previous):", key=reedit_key, value=st.session_state[reedit_key], height=100)

                if st.button("Re-Edit Inline", key=f"reedit_btn_{key_hash}"):
                    retext = st.session_state.get(reedit_key, "").strip()
                    if not retext:
                        st.warning("Enter re-edit instructions.")
                    else:
                        with st.spinner("Re-editing..."):
                            new_edited = run_edit_flow(retext, edited_bytes)
                            if new_edited:
                                ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                                outfn = f"outputs/edited/reedited_{ts}_{key_hash}.png"
                                os.makedirs(os.path.dirname(outfn), exist_ok=True)
                                with open(outfn, "wb") as f:
                                    f.write(new_edited)
                                st.success("Re-edited image created.")
                                show_image_safe(new_edited, caption=f"Re-Edited {name}")
                                st.download_button("⬇️ Download Re-Edited", data=new_edited, file_name=os.path.basename(outfn), mime="image/png", key=f"reedit_dl_{uuid.uuid4().hex}")
                                # append as new edited entry so you can chain again
                                st.session_state.edited_images.append({
                                    "original": edited_bytes,
                                    "edited": new_edited,
                                    "prompt": retext,
                                    "filename": outfn
                                })
                            else:
                                st.error("Re-edit returned no image.")

st.markdown("---")

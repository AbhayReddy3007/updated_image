# app.py
import os
import re
import uuid
import datetime
import hashlib
from io import BytesIO

import streamlit as st
from PIL import Image

# Lazy-import VertexAI so the app doesn't crash if SDK is not installed on dev machine.
try:
    import vertexai
    from vertexai.preview.vision_models import ImageGenerationModel
    from vertexai.generative_models import GenerativeModel, Part
    from google.oauth2 import service_account
    VERTEX_AVAILABLE = True
except Exception:
    VERTEX_AVAILABLE = False

# ---------------- Page config ----------------
st.set_page_config(page_title="AI Image Generator + Iterative Editor", layout="wide")
st.title("AI Image Generator + Iterative Editor")

# ---------------- Session initialization ----------------
def safe_init_session():
    try:
        _ = st.session_state
    except RuntimeError:
        return False
    st.session_state.setdefault("generated_images", [])   # list of {"filename","content","key","enhanced_prompt"}
    st.session_state.setdefault("edited_images", [])      # list of {"original","edited","prompt","filename","ts"}
    st.session_state.setdefault("edit_image_bytes", None) # bytes of the image currently loaded in the left editor
    st.session_state.setdefault("edit_image_name", "")
    st.session_state.setdefault("edit_iterations", 0)
    st.session_state.setdefault("max_edit_iterations", 20) # configurable cap
    return True

safe_init_session()

# ---------------- Prompt templates and style map ----------------
PROMPT_TEMPLATES = {
    "None": """
Don't change the user's prompt. Use it as-is.
User's raw prompt:
"{USER_PROMPT}"

Refined prompt:
""",
    "General": """
You are an expert image prompt engineer. Expand the user's short prompt into a clear, concise, and vivid prompt suitable for photoreal image generation.
User's raw prompt:
"{USER_PROMPT}"

Refined prompt:
""",
    "Design": """
You are a senior prompt engineer focused on design visuals. Expand the user's prompt into a design-oriented image prompt including composition, color palette, textures, and style.
User's raw prompt:
"{USER_PROMPT}"

Refined prompt:
""",
    "Marketing": """
You are a senior prompt engineer for marketing imagery. Turn the user's raw prompt into an ad-ready, campaign-friendly, professional image prompt. If the user references a brand, include that brand's tone in the prompt.
User's raw prompt:
"{USER_PROMPT}"

Refined prompt:
""",
    "DPEX": """
You are a prompt engineer creating technology/IT visuals. Expand the user's prompt for an IT/tech context (data center, futuristic UI, holograms) when appropriate.
User's raw prompt:
"{USER_PROMPT}"

Refined prompt:
""",
    "HR": """
You are a prompt engineer creating workplace and HR visuals. Expand the user's prompt into an inclusive, professional workplace scene if relevant.
User's raw prompt:
"{USER_PROMPT}"

Refined prompt:
""",
    "Business": """
You are a prompt engineer creating corporate visuals. Expand the user's prompt into a polished, business-oriented photo (boardrooms, pitch decks).
User's raw prompt:
"{USER_PROMPT}"

Refined prompt:
"""
}

STYLE_DESCRIPTIONS = {
    "None": "No special styling — keep it natural and faithful to the user's idea.",
    "Smart": "Clean, balanced, professional look.",
    "Cinematic": "Film-like composition and dramatic lighting.",
    "Vibrant": "Bold, saturated colors and high contrast.",
    "Photorealistic": "Highly realistic, lifelike photography style.",
}

# ---------------- Helpers ----------------
def sanitize_prompt(text: str) -> str:
    if not text:
        return text
    lines = []
    for line in text.splitlines():
        ln = line.strip()
        if not ln:
            continue
        if re.match(r'^(Option|Key|Apply|Specificity|Keywords)\b', ln, re.I):
            continue
        if re.match(r'^\d+[\.\)]\s*', ln):
            continue
        if len(ln) < 80 and ln.endswith(':'):
            continue
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
    """
    Returns (list_of_image_bytes, enhanced_prompt_str)
    """
    enhanced_prompt = prompt  # default

    if not VERTEX_AVAILABLE:
        st.warning("VertexAI SDK not available — generation disabled in this environment.")
        return [], enhanced_prompt

    creds = st.secrets.get("gcp_service_account")
    if not creds or not creds.get("project_id"):
        st.warning("Missing GCP credentials in Streamlit secrets: 'gcp_service_account'. Generation disabled.")
        return [], enhanced_prompt

    if not init_vertex(creds["project_id"], creds):
        st.warning("Failed to initialize VertexAI.")
        return [], enhanced_prompt

    # attempt text refinement when dept is selected
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
                st.warning(f"Prompt refinement failed, using raw prompt. ({e})")
                enhanced_prompt = prompt

    imagen = get_imagen_model()
    if imagen is None:
        st.warning("Imagen model unavailable.")
        return [], enhanced_prompt

    try:
        resp = imagen.generate_images(prompt=enhanced_prompt, number_of_images=n_images)
    except Exception as e:
        st.error(f"Imagen generate_images failed: {e}")
        return [], enhanced_prompt

    out = []
    for i in range(min(n_images, len(resp.images))):
        gen_obj = resp.images[i]
        b = get_image_bytes_from_genobj(gen_obj)
        if b:
            out.append(b)
    return out, enhanced_prompt

def run_edit_flow(edit_prompt, base_bytes):
    """
    Use Nano Banana (Gemini image gen) to apply edits to base_bytes.
    Returns edited bytes or None.
    """
    if not VERTEX_AVAILABLE:
        st.warning("VertexAI SDK not available — editing disabled.")
        return None

    creds = st.secrets.get("gcp_service_account")
    if not creds or not creds.get("project_id"):
        st.warning("Missing GCP credentials in Streamlit secrets: 'gcp_service_account'. Editing disabled.")
        return None

    if not init_vertex(creds["project_id"], creds):
        st.warning("Failed to initialize VertexAI.")
        return None

    nano = get_nano_banana_model()
    if nano is None:
        st.warning("Nano Banana editor model unavailable.")
        return None

    # Build parts: inline image + text instruction
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
        st.warning("Editor returned text instead of an image. Check response.")
    else:
        st.warning("Editor returned no inline image.")
    return None

# ---------------- UI ----------------
left_col, right_col = st.columns([3,1])

with left_col:
    st.subheader("Create or Edit — single flow")
    st.markdown("Upload an image to edit repeatedly, or generate new images and iterate edits on them until you download the final image.")

    # Controls
    dept = st.selectbox("🏢 Department (prompt refinement)", list(PROMPT_TEMPLATES.keys()), index=0)
    style = st.selectbox("🎨 Style (optional)", list(STYLE_DESCRIPTIONS.keys()), index=0)
    style_desc = "" if style == "None" else STYLE_DESCRIPTIONS.get(style, "")

    uploaded_file = st.file_uploader("Upload an image to edit (optional)", type=["png","jpg","jpeg","webp"])
    if uploaded_file:
        raw = uploaded_file.read()
        pil = Image.open(BytesIO(raw)).convert("RGB")
        buf = BytesIO()
        pil.save(buf, format="PNG")
        buf_bytes = buf.getvalue()
        # immediately load uploaded image into editor
        st.session_state["edit_image_bytes"] = buf_bytes
        st.session_state["edit_image_name"] = getattr(uploaded_file, "name", f"uploaded_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        st.session_state["edit_iterations"] = 0
        show_image_safe(buf_bytes, caption=f"Uploaded: {st.session_state['edit_image_name']}")
    else:
        if st.session_state.get("edit_image_bytes"):
            show_image_safe(st.session_state["edit_image_bytes"], caption=f"Editor loaded: {st.session_state.get('edit_image_name','Selected Image')}")

    prompt = st.text_area("Enter prompt (for generation or editing)", key="main_prompt", height=140,
                         placeholder="If an image is loaded this will edit it; otherwise it will generate new images.")
    num_images = st.slider("Number of images to generate (when generating)", min_value=1, max_value=4, value=1, key="num_images_slider")

    # Run button: either Edit (if edit image loaded) or Generate
    if st.button("Run"):
        prompt_text = (prompt or "").strip()
        if not prompt_text:
            st.warning("Please enter a prompt before running.")
        else:
            base_image = st.session_state.get("edit_image_bytes")
            if base_image:
                # EDIT flow: edit the loaded image and make result the new loaded image
                with st.spinner("Editing image..."):
                    edited = run_edit_flow(prompt_text, base_image)
                    if edited:
                        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                        out_fn = f"outputs/edited/edited_{ts}_{uuid.uuid4().hex[:6]}.png"
                        os.makedirs(os.path.dirname(out_fn), exist_ok=True)
                        with open(out_fn, "wb") as f:
                            f.write(edited)

                        st.success("Edited image created.")
                        show_image_safe(edited, caption=f"Edited ({ts})")

                        # --------------------------
                        # NEW: 3-column controls for current edited image: Download | Edit (load into editor) | Clear
                        # --------------------------
                        current_name = os.path.basename(out_fn)
                        safe_key = hashlib.sha1(current_name.encode()).hexdigest()[:12]

                        col_dl, col_edit, col_clear = st.columns([1,1,1])
                        with col_dl:
                            st.download_button(
                                "⬇️ Download Edited (current)",
                                data=edited,
                                file_name=current_name,
                                mime="image/png",
                                key=f"dl_edit_{safe_key}"
                            )
                        with col_edit:
                            if st.button("✏️ Edit this image (load into editor)", key=f"edit_current_{safe_key}"):
                                # put the current edited bytes into the editor slot (so the next Run will edit this image)
                                st.session_state["edit_image_bytes"] = edited
                                st.session_state["edit_image_name"] = current_name
                                st.session_state["edit_iterations"] = 0
                                st.experimental_rerun()
                        with col_clear:
                            if st.button("Clear editor (switch to generation)", key=f"clear_editor_{safe_key}"):
                                st.session_state["edit_image_bytes"] = None
                                st.session_state["edit_image_name"] = ""
                                st.session_state["edit_iterations"] = 0
                                st.experimental_rerun()
                        # --------------------------

                        # Replace the editor image with the freshly edited bytes so user can re-edit
                        st.session_state["edit_image_bytes"] = edited
                        st.session_state["edit_image_name"] = current_name

                        # increment iteration counter and append to edited history chain
                        st.session_state["edit_iterations"] = st.session_state.get("edit_iterations", 0) + 1
                        st.session_state.edited_images.append({
                            "original": base_image,
                            "edited": edited,
                            "prompt": prompt_text,
                            "filename": out_fn,
                            "ts": ts
                        })

                        # optional guard
                        if st.session_state["edit_iterations"] >= st.session_state.get("max_edit_iterations", 20):
                            st.warning(f"Reached {st.session_state['edit_iterations']} edits. Please finalize or reset to avoid runaway costs.")
                    else:
                        st.error("Editing failed or returned no image.")
            else:
                # GENERATION flow
                with st.spinner("Generating images..."):
                    generated, enhanced = generate_images_from_prompt(prompt_text, dept=dept, style_desc=style_desc, n_images=num_images)
                    if generated:
                        st.success(f"Generated {len(generated)} image(s).")
                        for i, b in enumerate(generated):
                            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                            fname = f"outputs/generated/gen_{ts}_{i}.png"
                            os.makedirs(os.path.dirname(fname), exist_ok=True)
                            with open(fname, "wb") as f:
                                f.write(b)

                            short = os.path.basename(fname) + str(i)
                            key_hash = uuid.uuid5(uuid.NAMESPACE_DNS, short).hex[:8]
                            entry = {"filename": fname, "content": b, "key": key_hash, "enhanced_prompt": enhanced}
                            st.session_state.generated_images.append(entry)
                    else:
                        st.error("Generation failed or returned no images.")

    # Option to clear editor (go back to generate-mode)
    if st.button("Clear editor (switch to generation)"):
        st.session_state["edit_image_bytes"] = None
        st.session_state["edit_image_name"] = ""
        st.session_state["edit_iterations"] = 0

    st.markdown("---")

    # -------------------------
    # Render Recently Generated (persistent, outside Run block)
    # -------------------------
    if st.session_state.get("generated_images"):
        st.markdown("### Recently Generated")
        for entry in reversed(st.session_state.generated_images[-12:]):
            fname = entry.get("filename")
            b = entry.get("content")
            key_hash = entry.get("key") or uuid.uuid5(uuid.NAMESPACE_DNS, os.path.basename(fname)).hex[:8]
            enhanced_prompt = entry.get("enhanced_prompt", "")

            show_image_safe(b, caption=os.path.basename(fname))

            if enhanced_prompt:
                with st.expander("Enhanced prompt (refined)"):
                    st.code(enhanced_prompt)

            col_dl, col_edit = st.columns([1,1])
            with col_dl:
                # stable download key per image
                st.download_button(
                    "⬇️ Download",
                    data=b,
                    file_name=os.path.basename(fname),
                    mime="image/png",
                    key=f"dl_gen_{key_hash}"
                )
            with col_edit:
                # stable edit button - loads the image into the editor so it becomes re-editable
                if st.button("✏️ Continue editing (load into editor)", key=f"edit_gen_{key_hash}"):
                    st.session_state["edit_image_bytes"] = b
                    st.session_state["edit_image_name"] = os.path.basename(fname)
                    st.session_state["edit_iterations"] = 0
                    st.experimental_rerun()

    # -------------------------
    # Render Edited History (allow picking any previous edited version to continue)
    # -------------------------
    if st.session_state.get("edited_images"):
        st.markdown("### Edited History (chain)")
        for idx, entry in enumerate(reversed(st.session_state.edited_images[-40:])):
            name = os.path.basename(entry.get("filename", f"edited_{idx}.png"))
            orig = entry.get("original")
            edited_bytes = entry.get("edited")
            prompt_prev = entry.get("prompt", "")
            ts = entry.get("ts", "")
            # uniqueish key for widgets in this loop
            hash_k = hashlib.sha1((name + ts + str(idx)).encode()).hexdigest()[:12]

            with st.expander(f"{name} — {prompt_prev[:80]}"):
                col1, col2 = st.columns(2)
                with col1:
                    if orig:
                        show_image_safe(orig, caption="Before")
                with col2:
                    show_image_safe(edited_bytes, caption="After")

                # download and continue-edit buttons side-by-side
                col_dl, col_edit = st.columns([1,1])
                with col_dl:
                    st.download_button("⬇️ Download Edited", data=edited_bytes, file_name=name, mime="image/png", key=f"hist_dl_{hash_k}")
                with col_edit:
                    if st.button("✏️ Continue editing this version", key=f"hist_edit_{hash_k}"):
                        st.session_state["edit_image_bytes"] = edited_bytes
                        st.session_state["edit_image_name"] = name
                        st.session_state["edit_iterations"] = 0
                        st.experimental_rerun()

# ---------------- Right column: smaller history + controls ----------------
with right_col:
    st.subheader("Controls & Chain")

    st.markdown(f"- Current editor image: **{st.session_state.get('edit_image_name') or 'None'}**")
    st.markdown(f"- Edit iterations performed this session: **{st.session_state.get('edit_iterations', 0)}**")
    max_it = st.number_input("Max edit iterations (safety cap)", min_value=1, max_value=100, value=st.session_state.get("max_edit_iterations", 20), step=1, key="ui_max_iters")
    st.session_state["max_edit_iterations"] = int(max_it)

    if st.button("Reset everything (clear session)"):
        # careful: this wipes history in session only
        st.session_state.generated_images = []
        st.session_state.edited_images = []
        st.session_state.edit_image_bytes = None
        st.session_state.edit_image_name = ""
        st.session_state.edit_iterations = 0
        st.experimental_rerun()

    st.markdown("---")
    st.markdown("### Quick history")
    if st.session_state.get("generated_images"):
        for entry in reversed(st.session_state.generated_images[-8:]):
            name = os.path.basename(entry.get("filename"))
            st.caption(name)

    st.markdown("### Edited chain (most recent)")
    if st.session_state.get("edited_images"):
        for entry in reversed(st.session_state.edited_images[-8:]):
            ts = entry.get("ts", "")
            nm = os.path.basename(entry.get("filename", "edited.png"))
            st.caption(f"{nm} — {ts}")

st.markdown("---")
st.caption("Note: each edit calls an external model (cost/time). Use the iteration cap to avoid runaway usage.")

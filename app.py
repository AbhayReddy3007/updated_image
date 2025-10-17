# streamlit_image_single_flow_with_dept_iterative_feedback.py
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
st.set_page_config(page_title="AI Image Generator + Editor (with Department)", layout="wide")
st.title("AI Image Generator + Editor (with Department)")

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
    """Strip headings, numbered options and labels often produced by a prompt refiner."""
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
    st.subheader("Create or Edit — single flow")
    st.markdown("**How it works:** If you upload an image (or load one from history into the editor), the prompt edits that image (Nano Banana). If you do not upload an image, the prompt generates new images (Imagen).")

    # Department selector (re-added)
    dept = st.selectbox("🏢 Department (controls prompt refinement)", list(PROMPT_TEMPLATES.keys()), index=0)

    # Style
    style = st.selectbox("🎨 Style (optional)", list(STYLE_DESCRIPTIONS.keys()), index=0)
    style_desc = "" if style == "None" else STYLE_DESCRIPTIONS.get(style, "")

    # Upload an image (optional)
    uploaded_file = st.file_uploader("Upload an image to edit (optional) — if present the prompt will edit this image", type=["png","jpg","jpeg","webp"])
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
    prompt = st.text_area("Enter prompt (for generation or editing)", key="main_prompt", height=140, placeholder="If you uploaded an image this will edit that image; otherwise it will generate new images.")
    num_images = st.slider("Number of images to generate (when generating)", min_value=1, max_value=4, value=1, key="num_images_slider")

    # Run button
    if st.button("Run"):
        prompt_text = (prompt or "").strip()
        if not prompt_text:
            st.warning("Please enter a prompt.")
        else:
            base_image = st.session_state.get("edit_image_bytes")
            if base_image:
                # EDIT FLOW
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
                # GENERATION FLOW
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

                            # store generated image with metadata
                            entry = {"filename": fname, "content": b, "key": key_hash}
                            st.session_state.generated_images.append(entry)

                            # display image and iterative-feedback UI
                            show_image_safe(b, caption=os.path.basename(fname))

                            # feedback text area (iterative editing)
                            fb_key = f"feedback_{key_hash}"
                            if fb_key not in st.session_state:
                                st.session_state[fb_key] = ""

                            finalized_key = f"finalized_{key_hash}"
                            if finalized_key not in st.session_state:
                                st.session_state[finalized_key] = False

                            col_a, col_b, col_c = st.columns([3, 2, 1])
                            with col_a:
                                st.text_area("Give feedback to edit this image (iterations allowed):", key=fb_key, value=st.session_state[fb_key], height=80)
                            with col_b:
                                if st.button("Edit with feedback", key=f"edit_fb_{key_hash}"):
                                    feedback_text = st.session_state.get(fb_key, "").strip()
                                    if not feedback_text:
                                        st.warning("Enter feedback for editing.")
                                    else:
                                        with st.spinner("Applying feedback edits..."):
                                            new_bytes = run_edit_flow(feedback_text, st.session_state.generated_images[-1]["content"]) if st.session_state.generated_images and st.session_state.generated_images[-1]["key"] == key_hash else run_edit_flow(feedback_text, b)
                                            if new_bytes:
                                                ts2 = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                                                outfn = f"outputs/edited/edited_{ts2}_{key_hash}.png"
                                                os.makedirs(os.path.dirname(outfn), exist_ok=True)
                                                with open(outfn, "wb") as f:
                                                    f.write(new_bytes)
                                                st.success("Edited image created from feedback.")
                                                # replace the stored content for this image so further edits use latest
                                                # find and update the matching generated image in session_state
                                                for idx, gi in enumerate(st.session_state.generated_images[::-1]):
                                                    # iterate reversed to find most recent matching key
                                                    if gi.get("key") == key_hash:
                                                        real_idx = len(st.session_state.generated_images) - 1 - idx
                                                        st.session_state.generated_images[real_idx]["content"] = new_bytes
                                                        st.session_state.generated_images[real_idx]["filename"] = outfn
                                                        break
                                                # also load into editor slot for convenience
                                                st.session_state["edit_image_bytes"] = new_bytes
                                                st.session_state["edit_image_name"] = os.path.basename(outfn)
                                                # show the new edited image immediately
                                                show_image_safe(new_bytes, caption=f"Edited ({ts2})")
                                            else:
                                                st.error("Feedback edit failed or returned no image.")
                                if st.button("Finalize & Prepare Download", key=f"finalize_{key_hash}"):
                                    st.session_state[finalized_key] = True
                                    st.success("Image finalized. Use the download button that just appeared to save the final image.")
                            with col_c:
                                # only show download when finalized
                                if st.session_state.get(finalized_key):
                                    # find the stored content for this key
                                    found_bytes = None
                                    for gi in st.session_state.generated_images[::-1]:
                                        if gi.get("key") == key_hash:
                                            found_bytes = gi.get("content")
                                            break
                                    if found_bytes:
                                        st.download_button("⬇️ Download Final Image", data=found_bytes, file_name=os.path.basename(fname), mime="image/png", key=f"dl_final_{key_hash}")
                                    else:
                                        st.error("No image data available for download.")

                    else:
                        st.error("Generation failed or returned no images.")

    # Clear editor button (switch to generation)
    if st.button("Clear editor (switch to generation)"):
        st.session_state["edit_image_bytes"] = None
        st.session_state["edit_image_name"] = ""

# ---------------- Right column: history + inline editing + re-edit ----------------
with right_col:
    st.subheader("📂 History")

    # Generated images
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

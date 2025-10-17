# streamlit_image_editor_switchable.py
import os
import datetime
import uuid
from io import BytesIO
import streamlit as st
from PIL import Image

# Optional: Vertex AI imports kept lazy to avoid running network/SDK calls at import time
try:
    import vertexai
    from vertexai.preview.vision_models import ImageGenerationModel
    from vertexai.generative_models import GenerativeModel, Part
    from google.oauth2 import service_account
    VERTEX_AVAILABLE = True
except Exception:
    VERTEX_AVAILABLE = False

# ---------------- STREAMLIT CONFIG ----------------
st.set_page_config(page_title="AI Image Generator + Editor", layout="wide")
st.title("AI Image Generator + Editor")

# ---------------- SAFE SESSION INIT ----------------
def init_session_state():
    """Initialize session state safely. Call this after set_page_config/title."""
    try:
        _ = st.session_state  # will raise RuntimeError if session isn't ready
    except RuntimeError:
        return False

    st.session_state.setdefault("generated_images", [])
    st.session_state.setdefault("edited_images", [])
    st.session_state.setdefault("edit_image_bytes", None)
    st.session_state.setdefault("edit_image_name", "")
    st.session_state.setdefault("active_tab", "generate")
    return True

init_session_state()

# ---------------- LAZY MODEL GETTERS ----------------
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
        credentials = service_account.Credentials.from_service_account_info(dict(credentials_info))
        vertexai.init(project=project_id, location=location, credentials=credentials)
        setattr(vertexai, "_initialized", True)
        return True
    except Exception as e:
        st.error(f"VertexAI init failed: {e}")
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
        st.error(f"Failed to load Nano Banana (editor) model: {e}")
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

# ---------------- IMAGE DISPLAY WRAPPER ----------------
def show_image_safe(image_source, caption="Image"):
    try:
        if isinstance(image_source, (bytes, bytearray)):
            st.image(image_source, caption=caption, use_container_width=True)
        elif isinstance(image_source, Image.Image):
            st.image(image_source, caption=caption, use_container_width=True)
        else:
            st.image(Image.open(BytesIO(image_source)), caption=caption, use_container_width=True)
    except TypeError:
        try:
            if isinstance(image_source, (bytes, bytearray)):
                st.image(image_source, caption=caption, use_column_width=True)
            else:
                st.image(Image.open(BytesIO(image_source)), caption=caption, use_column_width=True)
        except Exception as e:
            st.error(f"Failed to display image: {e}")
    except Exception as e:
        st.error(f"Failed to display image: {e}")

# ---------------- HELPERS ----------------
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
    for attr in ["image_bytes", "_image_bytes"]:
        if hasattr(gen_obj, attr):
            return getattr(gen_obj, attr)
    if hasattr(gen_obj, "image") and gen_obj.image:
        for attr in ["image_bytes", "_image_bytes"]:
            if hasattr(gen_obj.image, attr):
                return getattr(gen_obj.image, attr)
    return None

# ---------------- EDIT FLOW ----------------
def run_edit_flow(edit_prompt, base_bytes):
    nano = get_nano_banana_model()
    if nano is None:
        st.error("Editor model not available. Make sure VertexAI SDK is installed and configured in Streamlit secrets.")
        return None

    input_image = Part.from_data(mime_type="image/png", data=base_bytes)

    edit_instruction = f"""
You are a professional AI image editor.

Instructions:
- Take the provided image.
- Apply these edits: {edit_prompt}.
- Return the final edited image inline (PNG).
- Do not include any extra text or captions unless mentioned.
"""

    try:
        response = nano.generate_content([edit_instruction, input_image])
        for candidate in getattr(response, "candidates", []):
            for part in getattr(candidate.content, "parts", []):
                if hasattr(part, "inline_data") and getattr(part.inline_data, "data", None):
                    return part.inline_data.data
        if hasattr(response, "text") and response.text:
            st.warning(f"⚠️ Gemini returned text instead of an image:\n\n{response.text}")
        else:
            st.error("⚠️ No inline image returned by Nano Banana.")
        return None
    except Exception as e:
        st.error(f"❌ Error while editing: {e}")
        return None

# ---------------- TRANSFER TO EDIT VIEW ----------------
def select_image_for_edit(img_bytes, filename):
    st.session_state["edit_image_bytes"] = img_bytes
    st.session_state["edit_image_name"] = filename
    st.session_state["active_tab"] = "edit"
    # Force a rerun so the UI shows Edit view
    st.experimental_rerun()

# ---------------- PROMPT TEMPLATES & STYLE (trimmed) ----------------
PROMPT_TEMPLATES = {
    "None": """
Dont make any changes in the user's prompt.Follow it as it is
User’s raw prompt:
"{USER_PROMPT}"

Refined general image prompt:
""",
    "General": """
You are an expert AI prompt engineer specialized in creating vivid and descriptive image prompts.

User’s raw prompt:
"{USER_PROMPT}"

Refined general image prompt:
"""
}
STYLE_DESCRIPTIONS = {
    "None": "No special styling — keep the image natural, faithful to the user’s idea.",
    "Smart": "A clean, balanced, and polished look.",
    "Cinematic": "Film-style composition with dramatic lighting.",
}

# ---------------- CREATE OUTPUT FOLDERS ----------------
os.makedirs("outputs/generated", exist_ok=True)
os.makedirs("outputs/edited", exist_ok=True)

# ---------------- UI LAYOUT (radio-controlled pages) ----------------
col_left, col_right = st.columns([3, 1])

# map session_state active to radio index
active_tab = st.session_state.get("active_tab", "generate")
radio_index = 0 if active_tab == "generate" else 1

with col_left:
    page_choice = st.radio("Choose view", ("Generate Images", "Edit Images"),
                           index=radio_index, key="main_page_radio")
    # sync session_state when user manually changes radio
    if page_choice == "Generate Images":
        st.session_state["active_tab"] = "generate"
    else:
        st.session_state["active_tab"] = "edit"

    # ---------------- GENERATE VIEW ----------------
    if page_choice == "Generate Images":
        st.header(" Generate Images ")
        dept = st.selectbox("🏢 Department", list(PROMPT_TEMPLATES.keys()), index=0)
        style = st.selectbox("🎨 Style", list(STYLE_DESCRIPTIONS.keys()), index=0)
        user_prompt = st.text_area("Enter your prompt", height=120)
        num_images = st.slider("Number of images", min_value=1, max_value=4, value=1)

        if st.button(" Generate Images"):
            if not user_prompt.strip():
                st.warning("Please enter a prompt.")
            else:
                if not VERTEX_AVAILABLE:
                    st.error("VertexAI SDK not available in this environment. Install and configure it to use model features.")
                else:
                    creds_info = st.secrets.get("gcp_service_account")
                    if not creds_info or not creds_info.get("project_id"):
                        st.error("Missing GCP credentials in Streamlit secrets. Add 'gcp_service_account' JSON.")
                    else:
                        if not init_vertex(creds_info["project_id"], creds_info):
                            st.error("Failed to initialize VertexAI. Check logs.")
                        else:
                            with st.spinner("Refining prompt with Gemini..."):
                                text_model = get_text_model()
                                refinement_prompt = PROMPT_TEMPLATES.get(dept, PROMPT_TEMPLATES["General"]).replace("{USER_PROMPT}", user_prompt)
                                if style != "None":
                                    refinement_prompt += f"\n\nApply style: {STYLE_DESCRIPTIONS.get(style, '')}"
                                try:
                                    if text_model:
                                        text_resp = text_model.generate_content(refinement_prompt)
                                        enhanced_prompt = safe_get_enhanced_text(text_resp).strip()
                                    else:
                                        enhanced_prompt = refinement_prompt
                                except Exception as e:
                                    st.error(f"Prompt refinement failed: {e}")
                                    enhanced_prompt = refinement_prompt

                                if enhanced_prompt:
                                    st.info(f"🔮 Enhanced Prompt:\n\n{enhanced_prompt}")

                            with st.spinner("Generating images with Imagen 4..."):
                                imagen = get_imagen_model()
                                if imagen is None:
                                    st.error("Imagen model unavailable. Check Vertex setup.")
                                else:
                                    try:
                                        resp = imagen.generate_images(prompt=enhanced_prompt, number_of_images=num_images)
                                    except Exception as e:
                                        st.error(f"⚠️ Imagen error: {e}")
                                        resp = None

                                    if resp is None:
                                        st.stop()

                                    for i in range(num_images):
                                        try:
                                            gen_obj = resp.images[i]
                                            img_bytes = get_image_bytes_from_genobj(gen_obj)
                                            if not img_bytes:
                                                st.warning(f"No bytes for generated image index {i}.")
                                                continue

                                            filename = f"outputs/generated/{dept.lower()}_{style.lower()}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{i}.png"
                                            with open(filename, "wb") as f:
                                                f.write(img_bytes)

                                            st.session_state.generated_images.append({"filename": filename, "content": img_bytes})
                                            show_image_safe(img_bytes, caption=os.path.basename(filename))

                                            btn_key = f"dl_gen_{i}_{uuid.uuid4().hex}"
                                            st.download_button("⬇️ Download", data=img_bytes, file_name=os.path.basename(filename), mime="image/png", key=btn_key)
                                        except Exception as e:
                                            st.error(f"⚠️ Failed to display image {i}: {e}")

    # ---------------- EDIT VIEW ----------------
    else:
        st.header("Edit Images")

        uploaded_file = st.file_uploader("📤 Upload an image", type=["png", "jpg", "jpeg", "webp"])
        base_image = None

        # Priority: session_state edit image (sent from history) -> uploaded file
        if st.session_state.get("edit_image_bytes"):
            base_image = st.session_state.get("edit_image_bytes")
            show_image_safe(base_image, caption=f"Editing: {st.session_state.get('edit_image_name','Selected Image')}")
        elif uploaded_file:
            image_bytes = uploaded_file.read()
            img = Image.open(BytesIO(image_bytes)).convert("RGB")
            buf = BytesIO()
            img.save(buf, format="PNG")
            base_image = buf.getvalue()
            show_image_safe(base_image, caption="Uploaded Image")

        edit_prompt = st.text_area("Enter your edit instruction", height=120)
        num_edits = st.slider("How many edited variations to create", min_value=1, max_value=3, value=1)

        if st.button(" Edit image"):
            if not base_image or not edit_prompt.strip():
                st.warning("Please upload/select an image and enter instructions.")
            else:
                with st.spinner("Editing with Nano Banana..."):
                    edited_versions = []
                    for _ in range(num_edits):
                        edited = run_edit_flow(edit_prompt, base_image)
                        if edited:
                            edited_versions.append(edited)

                    if edited_versions:
                        for i, out_bytes in enumerate(edited_versions):
                            filename = f"outputs/edited/edited_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{i}.png"
                            with open(filename, "wb") as f:
                                f.write(out_bytes)

                            show_image_safe(out_bytes, caption=f"Edited Version {i+1}")
                            dl_key = f"edit_dl_{i}_{uuid.uuid4().hex}"
                            st.download_button(f"⬇️ Download Edited {i+1}", data=out_bytes, file_name=os.path.basename(filename), mime="image/png", key=dl_key)

                            st.session_state.edited_images.append({
                                "original": base_image,
                                "edited": out_bytes,
                                "prompt": edit_prompt,
                                "filename": filename,
                            })
                    else:
                        st.error("❌ No edited image returned by Nano Banana.")

# ---------------- HISTORY (right column) ----------------
with col_right:
    st.subheader("📂 History")

    if st.session_state.generated_images:
        st.markdown("### Generated Images")
        for i, img in enumerate(reversed(st.session_state.generated_images[-20:])):
            name = os.path.basename(img.get('filename', 'Unnamed Image'))
            with st.expander(f"{i+1}. {name}"):
                content = img.get("content")
                show_image_safe(content, caption=name)
                dl_key = f"gen_dl_hist_{i}_{uuid.uuid4().hex}"
                st.download_button("⬇️ Download Again", data=content, file_name=name, mime="image/png", key=dl_key)

                # Offer quick-send-to-edit
                if st.button("✏️ Edit this image", key=f"send_edit_{i}_{uuid.uuid4().hex}"):
                    select_image_for_edit(content, name)

    if st.session_state.edited_images:
        st.markdown("### Edited Images")
        for i, entry in enumerate(reversed(st.session_state.edited_images[-20:])):
            prompt_preview = entry.get("prompt", "")[:60]
            with st.expander(f"Edited {i+1}: {prompt_preview}"):
                col1, col2 = st.columns(2)
                with col1:
                    orig_bytes = entry.get("original")
                    show_image_safe(orig_bytes, caption="Original")
                with col2:
                    edited_bytes = entry.get("edited")
                    show_image_safe(edited_bytes, caption="Edited")
                    dl_key = f"edit_dl_hist_{i}_{uuid.uuid4().hex}"
                    st.download_button("⬇️ Download Edited", data=edited_bytes, file_name=os.path.basename(entry.get("filename", f"edited_{i}.png")), mime="image/png", key=dl_key)

# -------------- Usage tip --------------
st.markdown("---")
st.caption("Tip: Use the 'Edit this image' button in the history panel. The app will rerun and the left view selector will switch to Edit with that image preloaded.")

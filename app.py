# streamlit_image_single_flow.py
import os
import datetime
import uuid
import hashlib
from io import BytesIO
import streamlit as st
from PIL import Image

# Lazy Vertex imports (so app doesn't try to load SDK at import time)
try:
    import vertexai
    from vertexai.preview.vision_models import ImageGenerationModel
    from vertexai.generative_models import GenerativeModel, Part
    from google.oauth2 import service_account
    VERTEX_AVAILABLE = True
except Exception:
    VERTEX_AVAILABLE = False

# --------------------------------------------------
# Page config + safe session init
# --------------------------------------------------
st.set_page_config(page_title="AI Image Generator / Editor (Single Flow)", layout="wide")
st.title("AI Image Generator + Editor — single flow")

def safe_init_session():
    try:
        _ = st.session_state
    except RuntimeError:
        return False
    st.session_state.setdefault("generated_images", [])   # list of {"filename","content"}
    st.session_state.setdefault("edited_images", [])      # list of {"original","edited","prompt","filename"}
    st.session_state.setdefault("edit_image_bytes", None) # image currently loaded into editor
    st.session_state.setdefault("edit_image_name", "")
    return True

safe_init_session()

# --------------------------------------------------
# Model lazy initializers
# --------------------------------------------------
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

# --------------------------------------------------
# Utilities for image bytes and display
# --------------------------------------------------
def show_image_safe(image_source, caption="Image"):
    try:
        if isinstance(image_source, (bytes, bytearray)):
            st.image(image_source, caption=caption, use_container_width=True)
        elif isinstance(image_source, Image.Image):
            st.image(image_source, caption=caption, use_container_width=True)
        else:
            st.image(Image.open(BytesIO(image_source)), caption=caption, use_container_width=True)
    except TypeError:
        # Fallback for older Streamlit
        if isinstance(image_source, (bytes, bytearray)):
            st.image(image_source, caption=caption, use_column_width=True)
        else:
            st.image(Image.open(BytesIO(image_source)), caption=caption, use_column_width=True)

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

# --------------------------------------------------
# Core flows: generate & edit
# --------------------------------------------------
def generate_images_from_prompt(prompt, style_desc="", n_images=1):
    """Use Imagen to generate images from prompt. Returns list of bytes."""
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

    # optional refinement with text model (if available)
    text_model = get_text_model()
    enhanced_prompt = prompt
    if text_model:
        try:
            refinement = f"Refine the prompt for image generation:\n\n{prompt}\n\nApply style: {style_desc}"
            text_resp = text_model.generate_content(refinement)
            maybe = safe_get_enhanced_text(text_resp).strip()
            if maybe:
                enhanced_prompt = maybe
        except Exception:
            # ignore refinement errors; keep original prompt
            enhanced_prompt = prompt

    try:
        resp = imagen.generate_images(prompt=enhanced_prompt, number_of_images=n_images)
    except Exception as e:
        st.error(f"Imagen generate_images failed: {e}")
        return []

    out_bytes = []
    for i in range(min(n_images, len(resp.images))):
        gen_obj = resp.images[i]
        b = get_image_bytes_from_genobj(gen_obj)
        if b:
            out_bytes.append(b)
    return out_bytes

def run_edit_flow(edit_prompt, base_bytes):
    """Use Gemini Nano Banana to edit an existing image bytes. Returns edited bytes or None."""
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
        st.error(f"Nano Banana generate failed: {e}")
        return None

    for candidate in getattr(response, "candidates", []):
        for part in getattr(candidate.content, "parts", []):
            if hasattr(part, "inline_data") and getattr(part.inline_data, "data", None):
                return part.inline_data.data

    if hasattr(response, "text") and response.text:
        st.warning(f"Editor returned text instead of an image:\n\n{response.text}")
    else:
        st.error("Editor returned no inline image.")
    return None

# --------------------------------------------------
# UI: single view where action decided by whether there is an uploaded image
# --------------------------------------------------
left_col, right_col = st.columns([3, 1])

with left_col:
    st.subheader("Create or Edit — single flow")
    st.markdown("**How it works:** If you upload an image, the prompt will edit that image (Nano Banana). If you don't upload an image, the prompt will generate new images (Imagen).")

    uploaded_file = st.file_uploader("Upload an image to edit (optional)", type=["png","jpg","jpeg","webp"])
    if uploaded_file:
        # convert to PNG bytes
        raw = uploaded_file.read()
        pil = Image.open(BytesIO(raw)).convert("RGB")
        buf = BytesIO()
        pil.save(buf, format="PNG")
        buf_bytes = buf.getvalue()
        st.session_state["edit_image_bytes"] = buf_bytes
        st.session_state["edit_image_name"] = getattr(uploaded_file, "name", f"uploaded_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        show_image_safe(buf_bytes, caption=f"Uploaded: {st.session_state['edit_image_name']}")
    else:
        # if no uploaded file but session has previously set edit_image_bytes, show it (user may have used History->Edit)
        if st.session_state.get("edit_image_bytes"):
            show_image_safe(st.session_state["edit_image_bytes"], caption=f"Editor loaded: {st.session_state.get('edit_image_name','Selected Image')}")

    st.text_area("Enter prompt (for generation or editing)", key="main_prompt", height=140, placeholder="If you uploaded an image this will edit that image; otherwise it will generate new images.")
    # number of images only relevant for generation (but harmless to show)
    num_images = st.slider("Number of images to generate (when generating)", min_value=1, max_value=4, value=1, key="num_images_slider")
    style = st.selectbox("Style (optional)", ["None", "Smart", "Cinematic", "Vibrant"], index=0, key="style_choice")

    # action button
    if st.button("Run"):
        prompt_text = (st.session_state.get("main_prompt") or "").strip()
        if not prompt_text:
            st.warning("Please enter a prompt.")
        else:
            # Decide flow: edit if an image is present in session_state; otherwise generate
            base_image = st.session_state.get("edit_image_bytes")
            if base_image:
                # EDIT FLOW
                with st.spinner("Editing image with Nano Banana..."):
                    edited = run_edit_flow(prompt_text, base_image)
                    if edited:
                        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                        out_fn = f"outputs/edited/edited_{timestamp}_{uuid.uuid4().hex[:6]}.png"
                        with open(out_fn, "wb") as f:
                            f.write(edited)
                        # show result and add to history
                        st.success("Edited image created below.")
                        show_image_safe(edited, caption=f"Edited ({timestamp})")
                        st.download_button("⬇️ Download Edited", data=edited, file_name=os.path.basename(out_fn), mime="image/png", key=f"dl_edit_{uuid.uuid4().hex}")
                        st.session_state.edited_images.append({
                            "original": base_image,
                            "edited": edited,
                            "prompt": prompt_text,
                            "filename": out_fn
                        })
                        # load the edited image into editor for potential re-editing
                        st.session_state["edit_image_bytes"] = edited
                        st.session_state["edit_image_name"] = os.path.basename(out_fn)
                    else:
                        st.error("Editing failed or returned no image.")
            else:
                # GENERATION FLOW
                with st.spinner("Generating images with Imagen..."):
                    style_desc = "" if style == "None" else style
                    generated_list = generate_images_from_prompt(prompt_text, style_desc=style_desc, n_images=num_images)
                    if generated_list:
                        st.success(f"Generated {len(generated_list)} image(s).")
                        for i, b in enumerate(generated_list):
                            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                            fname = f"outputs/generated/gen_{ts}_{i}.png"
                            with open(fname, "wb") as f:
                                f.write(b)
                            st.session_state.generated_images.append({"filename": fname, "content": b})
                            show_image_safe(b, caption=os.path.basename(fname))
                            st.download_button("⬇️ Download", data=b, file_name=os.path.basename(fname), mime="image/png", key=f"dl_gen_{uuid.uuid4().hex}")
                    else:
                        st.error("Generation failed or returned no images.")

    # Provide a button to clear the editor image (so next prompt generates)
    if st.button("Clear editor (switch to generation)"):
        st.session_state["edit_image_bytes"] = None
        st.session_state["edit_image_name"] = ""

# --------------------------------------------------
# Right column: history + inline edits + re-edit
# --------------------------------------------------
with right_col:
    st.subheader("📂 History")

    # Generated history
    if st.session_state.get("generated_images"):
        st.markdown("#### Generated Images")
        for idx, entry in enumerate(reversed(st.session_state.generated_images[-20:])):
            name = os.path.basename(entry.get("filename", f"gen_{idx}.png"))
            content = entry.get("content")
            short_hash = hashlib.md5(name.encode()).hexdigest()[:8]
            with st.expander(name):
                show_image_safe(content, caption=name)
                st.download_button("⬇️ Download", data=content, file_name=name, mime="image/png", key=f"hist_dl_{short_hash}")
                if st.button("✏️ Edit this image", key=f"hist_edit_{short_hash}"):
                    # put into editor and reload page so left UI shows it
                    st.session_state["edit_image_bytes"] = content
                    st.session_state["edit_image_name"] = name
                    st.experimental_rerun()
                st.markdown("---")
                st.write("Edit inline (quick):")
                inline_key = f"inline_prompt_{short_hash}"
                st.text_area("Edit instructions", key=inline_key, value=st.session_state.get(inline_key,""), height=80)
                if st.button("Edit Inline", key=f"inline_btn_{short_hash}"):
                    prompt_val = st.session_state.get(inline_key,"").strip()
                    if not prompt_val:
                        st.warning("Enter edit instructions.")
                    else:
                        with st.spinner("Editing image inline..."):
                            edited_bytes = run_edit_flow(prompt_val, content)
                            if edited_bytes:
                                ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                                outfn = f"outputs/edited/edited_{ts}_{short_hash}.png"
                                with open(outfn, "wb") as f:
                                    f.write(edited_bytes)
                                st.success("Edited image created.")
                                show_image_safe(edited_bytes, caption=f"Edited {name}")
                                st.download_button("⬇️ Download Edited", data=edited_bytes, file_name=os.path.basename(outfn), mime="image/png", key=f"inline_dl_{uuid.uuid4().hex}")
                                st.session_state.edited_images.append({
                                    "original": content,
                                    "edited": edited_bytes,
                                    "prompt": prompt_val,
                                    "filename": outfn
                                })
                            else:
                                st.error("Edit returned no image.")

    # Edited history with re-edit (chain)
    if st.session_state.get("edited_images"):
        st.markdown("#### Edited Images (re-editable)")
        for idx, entry in enumerate(reversed(st.session_state.edited_images[-40:])):
            name = os.path.basename(entry.get("filename", f"edited_{idx}.png"))
            short_hash = hashlib.md5(name.encode()).hexdigest()[:8]
            with st.expander(f"{name} — {entry.get('prompt','')}"):
                col1, col2 = st.columns(2)
                with col1:
                    show_image_safe(entry.get("original"), caption="Original")
                with col2:
                    show_image_safe(entry.get("edited"), caption="Edited result")
                st.markdown("---")
                rekey = f"reedit_prompt_{short_hash}"
                st.text_area("Re-edit instructions", key=rekey, value=st.session_state.get(rekey, entry.get("prompt","")), height=90)
                if st.button("Re-Edit Inline", key=f"reedit_btn_{short_hash}"):
                    re_prompt = st.session_state.get(rekey, "").strip()
                    if not re_prompt:
                        st.warning("Please enter re-edit instructions.")
                    else:
                        with st.spinner("Re-editing..."):
                            new_edited = run_edit_flow(re_prompt, entry.get("edited"))
                            if new_edited:
                                ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                                outfn = f"outputs/edited/reedited_{ts}_{short_hash}.png"
                                with open(outfn, "wb") as f:
                                    f.write(new_edited)
                                st.success("Re-edited image created.")
                                show_image_safe(new_edited, caption=f"Re-Edited {name}")
                                st.download_button("⬇️ Download Re-Edited", data=new_edited, file_name=os.path.basename(outfn), mime="image/png", key=f"reedit_dl_{uuid.uuid4().hex}")
                                # append chain: previous edited becomes original
                                st.session_state.edited_images.append({
                                    "original": entry.get("edited"),
                                    "edited": new_edited,
                                    "prompt": re_prompt,
                                    "filename": outfn
                                })
                            else:
                                st.error("Re-edit returned no image.")

st.markdown("---")
st.caption("Notes: • If you upload an image, the prompt will edit that image. • If you clear the editor (Clear editor), the prompt will generate new images. • Make sure your Vertex credentials are set in Streamlit secrets to use Imagen/Nano Banana.")


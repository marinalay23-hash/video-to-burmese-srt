import streamlit as st
import streamlit.components.v1 as components
import tempfile
import os
import subprocess
import json
import pandas as pd
from faster_whisper import WhisperModel

st.set_page_config(page_title="Video to Burmese SRT Pro (Cloud)", layout="wide")
st.title("🎬 Video to Burmese SRT Generator")

# ==========================================
# Helper Functions
# ==========================================

def format_time(seconds: float) -> str:
    millis = int((seconds - int(seconds)) * 1000)
    secs = int(seconds) % 60
    mins = (int(seconds) // 60) % 60
    hours = int(seconds) // 3600
    return f"{hours:02d}:{mins:02d}:{secs:02d},{millis:03d}"

def extract_audio_ffmpeg(input_video_path: str) -> str:
    """FFmpeg သုံး၍ Video မှ Audio (.wav) သီးသန့် ထုတ်ယူခြင်း"""
    temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    temp_audio.close()
    
    cmd = [
        "ffmpeg", "-y", "-i", input_video_path,
        "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
        temp_audio.name
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    return temp_audio.name

def wrap_burmese_text(text: str, max_chars_per_line: int = 18) -> str:
    """မြန်မာစာတန်းထိုး ရှည်ပါက ၂ ကြောင်း အလိုအလျောက် ခွဲပေးခြင်း"""
    text = text.strip()
    if len(text) <= max_chars_per_line:
        return text

    midpoint = len(text) // 2
    best_split_idx = -1
    min_dist = len(text)

    split_chars = ["၊", "။", " "]
    for i, char in enumerate(text):
        if char in split_chars and 4 < i < len(text) - 4:
            dist = abs(i - midpoint)
            if dist < min_dist:
                min_dist = dist
                best_split_idx = i + 1

    if best_split_idx == -1:
        best_split_idx = midpoint

    line1 = text[:best_split_idx].strip()
    line2 = text[best_split_idx:].strip()
    return f"{line1}\n{line2}"

def reset_all():
    """Session State ရှင်းလင်းခြင်း"""
    st.session_state.segments = []
    st.session_state.translations = []
    if "user_json_input" in st.session_state:
        st.session_state.user_json_input = ""
    st.rerun()

# ==========================================
# Session State Init & Sidebar
# ==========================================
if "segments" not in st.session_state:
    st.session_state.segments = []
if "translations" not in st.session_state:
    st.session_state.translations = []

with st.sidebar:
    st.header("⚙️ Settings & Controls")
    auto_wrap = st.checkbox("စာကြောင်းရှည်ပါက ၂ ကြောင်း အလိုအလျောက်ခွဲမည်", value=True)
    chars_per_line = st.slider("တစ်ကြောင်းလျှင် အများဆုံး စာလုံးရေ:", min_value=10, max_value=30, value=16)
    
    st.divider()
    if st.button("🔄 Clear All (အသစ်ပြန်စရန်)", type="secondary", use_container_width=True):
        reset_all()

# ==========================================
# Step 1: Upload Video or Audio
# ==========================================
st.subheader("Step 1: Video သို့မဟုတ် Audio တင်ပါ")
uploaded_file = st.file_uploader("MP4, MKV, MP3, WAV ဖိုင်ရွေးပါ", type=["mp4", "mkv", "mp3", "wav"])

col1, col2, col3 = st.columns(3)
with col1:
    lang_choice = st.selectbox("အသံ ဘာသာစကား:", ["zh (တရုတ်)", "en (အင်္ဂလိပ်)", "auto (အလိုအလျောက်)"], index=0)
with col2:
    model_choice = st.selectbox("Whisper Model:", ["small (အကြံပြုချက်)", "base (အမြန်သုံး)"], index=0)
with col3:
    max_chars = st.slider("Whisper Segments ခွဲမည့် စာလုံးရေ:", min_value=6, max_value=20, value=12)

if uploaded_file and st.button("အသံဖမ်းယူမည် (Extract & Transcribe)", type="primary"):
    with st.spinner("ဖိုင်ကို လုပ်ဆောင်နေပါသည် (Cloud ပေါ်တွင် ခေတ္တစောင့်ပါ)..."):
        file_ext = os.path.splitext(uploaded_file.name)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_in:
            tmp_in.write(uploaded_file.read())
            input_path = tmp_in.name

        audio_path = input_path
        if file_ext.lower() in [".mp4", ".mkv", ".mov", ".avi"]:
            with st.spinner("FFmpeg ဖြင့် Audio သီးသန့် ထုတ်ယူနေပါသည်..."):
                audio_path = extract_audio_ffmpeg(input_path)

        with st.spinner("Whisper AI ဖြင့် အသံဖမ်းယူနေပါသည်..."):
            selected_model = "small" if "small" in model_choice else "base"
            selected_lang = lang_choice.split(" ")[0]

            model = WhisperModel(selected_model, device="cpu", compute_type="int8")
            transcribe_args = {"beam_size": 5, "word_timestamps": True}
            
            if selected_lang != "auto":
                transcribe_args["language"] = selected_lang
                if selected_lang == "zh":
                    transcribe_args["initial_prompt"] = "这是一段普通话对话，包含日常交流、情感与生活用语。"

            segments_raw, _ = model.transcribe(audio_path, **transcribe_args)

            capcut_segments = []
            for seg in segments_raw:
                if not seg.words:
                    capcut_segments.append({"start": seg.start, "end": seg.end, "text": seg.text.strip()})
                    continue

                curr_text, curr_start, curr_end = "", None, None
                for w in seg.words:
                    if curr_start is None:
                        curr_start = w.start
                    curr_text += w.word
                    curr_end = w.end

                    if len(curr_text) >= max_chars or any(p in w.word for p in "，。！？,.?!"):
                        clean_t = curr_text.strip("，。！？,.?! ")
                        if clean_t:
                            capcut_segments.append({"start": curr_start, "end": curr_end, "text": clean_t})
                        curr_text, curr_start = "", None

                if curr_text.strip("，。！？,.?! "):
                    capcut_segments.append({"start": curr_start, "end": curr_end, "text": curr_text.strip("，。！？,.?! ")})

            st.session_state.segments = capcut_segments

        try:
            if os.path.exists(input_path):
                os.remove(input_path)
            if audio_path != input_path and os.path.exists(audio_path):
                os.remove(audio_path)
        except Exception:
            pass

    st.success(f"စာကြောင်းပေါင်း {len(st.session_state.segments)} ကြောင်း အောင်မြင်စွာ ထုတ်ယူပြီးပါပြီ!")

# ==========================================
# Step 2: Copy Prompt for Gemini
# ==========================================
st.divider()
st.subheader("Step 2: မူရင်းစာသားနှင့် Gemini Prompt ရယူပါ")
total_count = len(st.session_state.segments)

if st.session_state.segments:
    lines_only = [item["text"] for item in st.session_state.segments]
    input_display = json.dumps(lines_only, ensure_ascii=False, indent=2)
else:
    input_display = '[\n  "Video တင်ပြီးပါက ဤနေရာတွင် စာကြောင်းများ အလိုအလျောက် ပေါ်လာပါမည်..."\n]'

gemini_prompt = f"""You are a master film dialogue translator and localizer specializing in Chinese-to-Burmese video adaptation.
Translate the following short subtitle clips into natural, conversational, spoken Burmese (မြန်မာစကားပြော စာတန်းထိုး အသုံးအနှုန်းစစ်စစ်).

CRITICAL CONTEXT & FLOW RULES:
1. READ THE WHOLE STORY FIRST: Read the entire sequence to understand plot, context, tone, and character dynamics.
2. Maintain Sentence Continuity Across Splits: Translate connected clauses smoothly so they link naturally in Burmese.
3. Subtitle Style: Spoken Myanmar markers only ("တယ်", "မယ်", "ပါ", "ဘူး", "လား", "နော်", "လေ", "ပေါ့"). Never use formal literary markers ("သည်", "၏", "သဖြင့်").
4. STRICT 1-TO-1 ARRAY MAPPING: The input has EXACTLY {total_count} lines. Return EXACTLY {total_count} items in the same sequence.
5. Format: Myanmar script ONLY. No English/Chinese.

Return STRICTLY as a valid JSON object:
{{
  "translations": [
    "အဓိပ္ပာယ်ဆက်စပ်သော စာတန်းတို ၁",
    "ရှေ့စကားနှင့် အချိတ်အဆက်မိသော စာတန်းတို ၂"
  ]
}}

Input dialogue lines ({total_count} items):
{input_display}"""

js_text = json.dumps(gemini_prompt)
copy_component_html = f"""
<div style="display: flex; justify-content: flex-end; margin-bottom: 8px;">
    <button id="copy-btn" onclick="copyPrompt()" style="
        background-color: #ffffff; color: #31333F; border: 1px solid #d3d3d3;
        padding: 8px 16px; border-radius: 6px; cursor: pointer; font-size: 14px; font-weight: 500;
    ">📋 Copy Full Prompt</button>
</div>
<script>
function copyPrompt() {{
    const text = {js_text};
    navigator.clipboard.writeText(text).then(() => {{
        const btn = document.getElementById("copy-btn");
        btn.innerText = "✅ Copied!";
        btn.style.backgroundColor = "#e6f4ea";
        setTimeout(() => {{ btn.innerText = "📋 Copy Full Prompt"; btn.style.backgroundColor = "#ffffff"; }}, 2000);
    }}).catch(err => {{ alert("Copy မရပါ: " + err); }});
}}
</script>
"""
components.html(copy_component_html, height=45)
st.text_area("Prompt အစမ်းကြည့်ရှုရန်:", value=gemini_prompt, height=180)
st.markdown("[👉 Gemini App သို့ တိုက်ရိုက်သွားရန် နှိပ်ပါ (gemini.google.com)](https://gemini.google.com/app)")

# ==========================================
# Step 3: Paste JSON from Gemini & Edit Table
# ==========================================
st.divider()
st.subheader("Step 3: Gemini မှ ရရှိသော JSON စာသားကို ထည့်ပါ")
user_json = st.text_area(
    "Gemini ပေးလိုက်သော JSON ကို ဤနေရာတွင် Paste ချပါ:",
    height=160,
    placeholder='{\n  "translations": [\n    "မြန်မာဘာသာပြန် စာကြောင်း ၁",\n    "မြန်မာဘာသာပြန် စာကြောင်း ၂"\n  ]\n}',
    key="user_json_input"
)

if user_json.strip() and st.session_state.segments:
    try:
        parsed_data = json.loads(user_json)
        raw_translations = parsed_data.get("translations", [])
        
        table_data = []
        for i, seg in enumerate(st.session_state.segments):
            orig = seg["text"]
            burmese = raw_translations[i] if i < len(raw_translations) else ""
            time_str = f"{format_time(seg['start'])} --> {format_time(seg['end'])}"
            table_data.append({
                "Index": i + 1,
                "Timecode": time_str,
                "Original": orig,
                "Burmese Translation": burmese
            })

        st.markdown("#### 📝 Subtitle Editor Table (ဘာသာပြန်ထားသော စာသားများအား ဤဇယားတွင် တိုက်ရိုက် ပြင်ဆင်နိုင်သည်)")
        df = pd.DataFrame(table_data)
        
        edited_df = st.data_editor(
            df,
            column_config={
                "Index": st.column_config.NumberColumn(width="small", disabled=True),
                "Timecode": st.column_config.TextColumn(width="medium", disabled=True),
                "Original": st.column_config.TextColumn(width="medium", disabled=True),
                "Burmese Translation": st.column_config.TextColumn(width="large", disabled=False)
            },
            hide_index=True,
            use_container_width=True
        )

        st.session_state.translations = edited_df["Burmese Translation"].tolist()

    except Exception as e:
        st.error(f"JSON ဖတ်ရာတွင် အမှားရှိနေပါသည်: {e}")

# ==========================================
# Step 4: SRT Output & Download
# ==========================================
st.divider()
st.subheader("Step 4: SRT ဖိုင် အမည်ပေးပြီး ဒေါင်းလုဒ်ဆွဲပါ")

col_name, col_down_btn = st.columns([2, 1])
with col_name:
    output_filename = st.text_input("သိမ်းဆည်းမည့် ဖိုင်အမည် (Filename):", value="subtitles.srt")
    if not output_filename.endswith(".srt"):
        output_filename += ".srt"

if st.session_state.segments and st.session_state.translations:
    srt_output = []
    for idx, item in enumerate(st.session_state.segments):
        burmese_text = st.session_state.translations[idx] if idx < len(st.session_state.translations) else item["text"]
        
        if auto_wrap:
            burmese_text = wrap_burmese_text(burmese_text, max_chars_per_line=chars_per_line)
            
        start_str = format_time(item["start"])
        end_str = format_time(item["end"])
        srt_output.append(f"{idx + 1}\n{start_str} --> {end_str}\n{burmese_text}\n")

    final_srt_content = "\n".join(srt_output)

    with col_down_btn:
        st.write("") # spacing
        st.write("")
        st.download_button(
            label=f"📥 Download ({output_filename})",
            data=final_srt_content.encode("utf-8"),
            file_name=output_filename,
            mime="text/plain",
            type="primary",
            use_container_width=True
        )
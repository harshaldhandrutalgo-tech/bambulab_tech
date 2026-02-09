"""
Bambu Lab Smart Print Profile Generator (LLM-Powered)
=====================================================
Upload an STL, describe what you're printing in plain English,
and get safe, optimized Bambu Lab print profiles.

Install:
    pip install streamlit google-genai numpy python-dotenv

Run:
    streamlit run app.py

Create a .env file:
    GOOGLE_API_KEY=your_gemini_key
"""

import streamlit as st
import numpy as np
import struct
import json
import io
import re
import zipfile
import os
import tempfile
import traceback
import subprocess
import platform
import urllib.parse
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

st.set_page_config(
    page_title="MakerSpace Print Advisor",
    page_icon="🖨️",
    layout="wide",
)


# ─────────────────────────────────────────────
# STL Parsing
# ─────────────────────────────────────────────

def parse_stl_binary(data: bytes):
    num_triangles = struct.unpack('<I', data[80:84])[0]
    triangles = []
    offset = 84
    for _ in range(num_triangles):
        normal = struct.unpack('<3f', data[offset:offset + 12])
        v1 = struct.unpack('<3f', data[offset + 12:offset + 24])
        v2 = struct.unpack('<3f', data[offset + 24:offset + 36])
        v3 = struct.unpack('<3f', data[offset + 36:offset + 48])
        offset += 50
        triangles.append((normal, v1, v2, v3))
    return triangles


def parse_stl_ascii(text: str):
    triangles = []
    normal = None
    verts = []
    for line in text.strip().split('\n'):
        line = line.strip()
        if line.startswith('facet normal'):
            parts = line.split()
            normal = (float(parts[2]), float(parts[3]), float(parts[4]))
            verts = []
        elif line.startswith('vertex'):
            parts = line.split()
            verts.append((float(parts[1]), float(parts[2]), float(parts[3])))
        elif line.startswith('endfacet'):
            if normal and len(verts) == 3:
                triangles.append((normal, verts[0], verts[1], verts[2]))
    return triangles


def analyze_stl(file_bytes: bytes):
    try:
        text = file_bytes.decode('ascii')
        if text.strip().lower().startswith('solid') and 'facet' in text.lower():
            triangles = parse_stl_ascii(text)
        else:
            triangles = parse_stl_binary(file_bytes)
    except (UnicodeDecodeError, ValueError):
        triangles = parse_stl_binary(file_bytes)

    if not triangles:
        return None

    all_verts = []
    for t in triangles:
        all_verts.extend([t[1], t[2], t[3]])
    verts = np.array(all_verts)

    min_c = verts.min(axis=0)
    max_c = verts.max(axis=0)
    dims = max_c - min_c

    volume = 0.0
    surface_area = 0.0
    for t in triangles:
        v1, v2, v3 = np.array(t[1]), np.array(t[2]), np.array(t[3])
        volume += np.dot(v1, np.cross(v2, v3)) / 6.0
        surface_area += np.linalg.norm(np.cross(v2 - v1, v3 - v1)) / 2.0
    volume = abs(volume)

    sorted_dims = sorted(dims)
    bbox_vol = float(dims[0] * dims[1] * dims[2]) if all(d > 0 for d in dims) else 1.0

    return {
        "dimensions_mm": {
            "x": round(float(dims[0]), 2),
            "y": round(float(dims[1]), 2),
            "z": round(float(dims[2]), 2),
        },
        "volume_cm3": round(float(volume / 1000.0), 2),
        "surface_area_mm2": round(float(surface_area), 2),
        "triangle_count": len(triangles),
        "max_aspect_ratio": round(float(sorted_dims[2] / max(sorted_dims[0], 0.01)), 2),
        "is_tall": bool(dims[2] > max(dims[0], dims[1]) * 2.5),
        "is_flat": bool(dims[2] < min(dims[0], dims[1]) * 0.15),
        "fill_ratio": round(float(volume / bbox_vol), 3) if bbox_vol > 0 else 0,
        "fits_a1_mini": bool(all(d <= 180 for d in dims)),
        "fits_a1": bool(all(d <= 256 for d in dims)),
        "fits_x1c": bool(all(d <= 256 for d in dims)),
        "fits_p1s": bool(all(d <= 256 for d in dims)),
    }


# ─────────────────────────────────────────────
# .3MF Generation
# ─────────────────────────────────────────────

def generate_3mf(stl_bytes: bytes, profile: dict) -> bytes:
    try:
        text = stl_bytes.decode('ascii')
        if text.strip().lower().startswith('solid') and 'facet' in text.lower():
            triangles = parse_stl_ascii(text)
        else:
            triangles = parse_stl_binary(stl_bytes)
    except (UnicodeDecodeError, ValueError):
        triangles = parse_stl_binary(stl_bytes)

    vertices = []
    tri_indices = []
    vmap = {}
    idx = 0
    for t in triangles:
        face = []
        for v in [t[1], t[2], t[3]]:
            key = (round(v[0], 6), round(v[1], 6), round(v[2], 6))
            if key not in vmap:
                vmap[key] = idx
                vertices.append(key)
                idx += 1
            face.append(vmap[key])
        tri_indices.append(tuple(face))

    verts_xml = "\n".join(
        f'          <vertex x="{v[0]}" y="{v[1]}" z="{v[2]}" />'
        for v in vertices
    )
    tris_xml = "\n".join(
        f'          <triangle v1="{t[0]}" v2="{t[1]}" v3="{t[2]}" />'
        for t in tri_indices
    )

    pn = profile.get("profile_name", "MakerSpace_Profile")
    pr = profile.get("printer", "X1C")
    q = profile.get("quality", {})
    s = profile.get("strength", {})
    sp = profile.get("speed", {})
    m = profile.get("material", {})
    a = profile.get("adhesion", {})
    sup = profile.get("support", {})
    brim = "auto_brim" if a.get("brim", False) else "no_brim"

    model_xml = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<model unit="millimeter" xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02"\n'
        '       xmlns:slic3rpe="http://schemas.slic3r.org/3mf/2017/06">\n'
        f'  <metadata name="Application">BambuStudio</metadata>\n'
        f'  <metadata name="BambuStudio:ProfileName">{pn}</metadata>\n'
        '  <resources>\n'
        '    <object id="1" type="model">\n'
        '      <mesh>\n'
        f'        <vertices>\n{verts_xml}\n        </vertices>\n'
        f'        <triangles>\n{tris_xml}\n        </triangles>\n'
        '      </mesh>\n'
        '    </object>\n'
        '  </resources>\n'
        '  <build>\n    <item objectid="1" />\n  </build>\n'
        '</model>'
    )

    config_ini = (
        f"; Generated by MakerSpace Print Advisor\n; Profile: {pn}\n; Printer: {pr}\n\n"
        f"[print]\nlayer_height = {q.get('layer_height', 0.2)}\n"
        f"first_layer_height = {q.get('first_layer_height', 0.2)}\n"
        f"wall_loops = {s.get('wall_loops', 3)}\n"
        f"top_shell_layers = {s.get('top_shell_layers', 4)}\n"
        f"bottom_shell_layers = {s.get('bottom_shell_layers', 4)}\n"
        f"sparse_infill_density = {s.get('infill_density', 15)}%\n"
        f"sparse_infill_pattern = {s.get('infill_pattern', 'grid')}\n"
        f"outer_wall_speed = {sp.get('outer_wall_speed', 150)}\n"
        f"inner_wall_speed = {sp.get('inner_wall_speed', 180)}\n"
        f"sparse_infill_speed = {sp.get('infill_speed', 250)}\n"
        f"travel_speed = {sp.get('travel_speed', 350)}\n"
        f"initial_layer_speed = {sp.get('first_layer_speed', 50)}\n"
        f"brim_type = {brim}\nbrim_width = {a.get('brim_width', 0)}\n"
        f"support_type = {sup.get('type', 'none')}\n"
        f"line_width = {q.get('line_width', 0.42)}\n"
        f"nozzle_diameter = {q.get('nozzle_diameter', 0.4)}\n\n"
        f"[filament]\nfilament_type = {m.get('type', 'PLA')}\n"
        f"nozzle_temperature = {m.get('nozzle_temperature', 220)}\n"
        f"bed_temperature = {m.get('bed_temperature', 55)}\n\n"
        f"[printer]\nprinter_model = {pr.replace(' ', '')}\n"
    )

    content_types = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">\n'
        '  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml" />\n'
        '  <Default Extension="model" ContentType="application/vnd.ms-package.3dmanufacturing-3dmodel+xml" />\n'
        '</Types>'
    )

    rels = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">\n'
        '  <Relationship Target="/3D/3dmodel.model" Id="rel0"\n'
        '    Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel" />\n'
        '</Relationships>'
    )

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", content_types)
        zf.writestr("_rels/.rels", rels)
        zf.writestr("3D/3dmodel.model", model_xml)
        zf.writestr("Metadata/plate_1.config", config_ini)
        zf.writestr("3D/model.stl", stl_bytes)
    return buf.getvalue()


# ─────────────────────────────────────────────
# Open in Bambu Studio
# ─────────────────────────────────────────────

def save_3mf_temp(data: bytes, name: str) -> str:
    path = os.path.join(tempfile.gettempdir(), name)
    with open(path, 'wb') as f:
        f.write(data)
    return path


# def open_in_bambu_studio(filepath: str) -> tuple:
#     """Open .3mf in Bambu Studio. Returns (success, message)."""
#     sys_name = platform.system()
#     try:
#         if sys_name == "Windows":
#             os.startfile(filepath)
#         elif sys_name == "Darwin":
#             subprocess.run(["open", filepath], check=True)
#         else:
#             subprocess.run(["xdg-open", filepath], check=True)
#         return True, "Bambu Studio should open with your model and settings loaded."
#     except FileNotFoundError:
#         return False, "Could not open file. Make sure Bambu Studio is installed and .3mf files are associated with it."
#     except Exception as e:
#         return False, f"Failed to open: {e}"


def open_in_bambu_studio(filepath: str) -> tuple:
    """Explicitly open .3mf in Bambu Studio"""
    sys_name = platform.system()

    try:
        if sys_name == "Windows":
            bambu_path = r"C:\Program Files\Bambu Studio\bambu-studio.exe"

            if not os.path.exists(bambu_path):
                return False, "Bambu Studio not found at default install location."

            subprocess.Popen([bambu_path, filepath])

        elif sys_name == "Darwin":
            subprocess.run([
                "open", "-a", "Bambu Studio", filepath
            ], check=True)

        else:
            subprocess.run([
                "bambu-studio", filepath
            ], check=True)

        return True, "Opened in Bambu Studio successfully."

    except FileNotFoundError:
        return False, "Bambu Studio executable not found."
    except Exception as e:
        return False, f"Failed to open in Bambu Studio: {e}"

# ─────────────────────────────────────────────
# LLM
# ─────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are MakerSpace Print Advisor, an expert 3D printing consultant specialized in "
    "Bambu Lab printers (A1 mini, A1, P1S, X1C). You work in a college makerspace helping "
    "beginners get successful prints.\n\n"
    "Your job: Given STL geometry metadata and a plain-English description, generate a "
    "SAFE, KNOWN-GOOD print profile. Prioritize reliability over pushing limits.\n\n"
    "## Printers\n"
    "- A1 mini: 180x180x180mm, no enclosure, 0.4mm nozzle, max 300mm/s safe. No ABS.\n"
    "- A1: 256x256x256mm, no enclosure, 0.4mm nozzle, max 300mm/s safe. No ABS.\n"
    "- P1S: 256x256x256mm, enclosed, 0.4/0.6/0.8mm nozzle, max 350mm/s safe. ABS OK.\n"
    "- X1C: 256x256x256mm, enclosed, lidar, 0.4/0.6/0.8mm nozzle, max 400mm/s safe.\n\n"
    "## Materials\n"
    "- PLA (220C/55C): Default. Easy, good quality. Not heat resistant.\n"
    "- PETG (245C/70C): Stronger, heat resistant. Slight stringing.\n"
    "- PLA-CF (230C/55C): Carbon fiber PLA. Very stiff. Needs hardened nozzle.\n"
    "- TPU 95A (230C/50C): Flexible. Slow only. Max 80mm/s.\n"
    "- ABS (260C/100C): Strong. NEEDS enclosure (P1S/X1C only).\n\n"
    "## Safety Rules\n"
    "1. NEVER recommend ABS on A1 or A1 mini\n"
    "2. NEVER exceed safe speed limits\n"
    "3. TPU max 80mm/s outer wall\n"
    "4. Tall prints need brim\n"
    "5. Large footprint >150mm gets brim\n"
    "6. First layer speed always <=50mm/s\n"
    "7. When in doubt, safer/slower\n\n"
    "## Response: ONLY valid JSON, no markdown, no backticks.\n"
    "All strings on SINGLE LINE. Keep reasoning/warnings/tips to one sentence each.\n\n"
    "Schema:\n"
    '{"profile_name":"str","printer":"str","reasoning":"str",'
    '"material":{"name":"str","type":"PLA|PETG|PLA-CF|TPU|ABS",'
    '"nozzle_temperature":int,"bed_temperature":int},'
    '"quality":{"layer_height":float,"first_layer_height":float,'
    '"nozzle_diameter":float,"line_width":float},'
    '"strength":{"wall_loops":int,"top_shell_layers":int,"bottom_shell_layers":int,'
    '"infill_density":int,"infill_pattern":"str"},'
    '"speed":{"outer_wall_speed":int,"inner_wall_speed":int,"infill_speed":int,'
    '"travel_speed":int,"first_layer_speed":int},'
    '"support":{"type":"none|normal(auto)","notes":"str"},'
    '"adhesion":{"brim":bool,"brim_width":int},'
    '"warnings":["str"],"tips":["str"]}'
)


def clean_json_string(raw: str) -> str:
    text = raw.strip()
    text = re.sub(r'^```(?:json)?\s*\n?', '', text)
    text = re.sub(r'\n?```\s*$', '', text)
    text = text.strip()

    start = text.find('{')
    if start == -1:
        raise ValueError("No JSON object found")

    depth = 0
    in_string = False
    escape_next = False
    end = -1
    for i in range(start, len(text)):
        c = text[i]
        if escape_next:
            escape_next = False
            continue
        if c == '\\' and in_string:
            escape_next = True
            continue
        if c == '"' and not escape_next:
            in_string = not in_string
            continue
        if in_string:
            continue
        if c == '{':
            depth += 1
        elif c == '}':
            depth -= 1
            if depth == 0:
                end = i
                break

    if end == -1:
        raise ValueError("No matching closing brace")

    json_str = text[start:end + 1]
    json_str = re.sub(r',\s*([}\]])', r'\1', json_str)

    result = []
    in_str = False
    esc = False
    for ch in json_str:
        if esc:
            result.append(ch)
            esc = False
            continue
        if ch == '\\' and in_str:
            result.append(ch)
            esc = True
            continue
        if ch == '"':
            in_str = not in_str
            result.append(ch)
            continue
        if in_str and ch in ('\n', '\r', '\t'):
            result.append(' ')
            continue
        result.append(ch)
    return ''.join(result)


def call_llm(stl_info: dict, description: str, printer: str) -> dict:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not found in .env file.")

    client = genai.Client(api_key=api_key)

    user_msg = (
        f"STL Analysis:\n{json.dumps(stl_info, indent=2)}\n\n"
        f"Printer: {printer}\n\n"
        f"Description: \"{description}\"\n\n"
        f"Return ONLY valid JSON."
    )

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=user_msg,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0.1,
            max_output_tokens=2000,
            response_mime_type="application/json",
        ),
    )

    raw = response.text
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return json.loads(clean_json_string(raw))


# ─────────────────────────────────────────────
# Display Profile
# ─────────────────────────────────────────────

def display_profile(profile: dict, stl_bytes: bytes, filename: str):
    mat = profile.get("material", {})
    qual = profile.get("quality", {})
    stren = profile.get("strength", {})
    spd = profile.get("speed", {})

    reasoning = profile.get("reasoning", "")
    if reasoning:
        st.markdown(
            f'<div class="success-box"><strong>Why these settings:</strong> {reasoning}</div>',
            unsafe_allow_html=True,
        )

    # Metrics row 1
    st.markdown("#### Settings Overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Material", mat.get("type", "PLA"))
    c2.metric("Layer Height", f"{qual.get('layer_height', 0.2)} mm")
    c3.metric("Infill", f"{stren.get('infill_density', 15)}%")
    c4.metric("Walls", f"{stren.get('wall_loops', 3)} loops")

    # Metrics row 2
    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Nozzle Temp", f"{mat.get('nozzle_temperature', 220)}°C")
    c6.metric("Bed Temp", f"{mat.get('bed_temperature', 55)}°C")
    c7.metric("Wall Speed", f"{spd.get('outer_wall_speed', 150)} mm/s")
    c8.metric("Infill Pattern", stren.get("infill_pattern", "grid"))

    # Expandable full detail
    with st.expander("🔍 All Settings", expanded=False):
        dc1, dc2 = st.columns(2)
        with dc1:
            st.markdown("**Quality**")
            for k, v in qual.items():
                st.text(f"  {k}: {v}")
            st.markdown("**Strength**")
            for k, v in stren.items():
                st.text(f"  {k}: {v}")
        with dc2:
            st.markdown("**Speed**")
            for k, v in spd.items():
                st.text(f"  {k}: {v}")
            st.markdown("**Support**")
            for k, v in profile.get("support", {}).items():
                st.text(f"  {k}: {v}")
            st.markdown("**Adhesion**")
            for k, v in profile.get("adhesion", {}).items():
                st.text(f"  {k}: {v}")

    # Warnings & Tips
    for w in profile.get("warnings", []):
        st.markdown(f'<div class="warning-box">{w}</div>', unsafe_allow_html=True)
    for t in profile.get("tips", []):
        st.markdown(f'<div class="tip-box">{t}</div>', unsafe_allow_html=True)

    # ── Generate .3MF ──
    tmf_bytes = None
    base = os.path.splitext(filename)[0]
    tmf_name = f"{base}_{profile.get('profile_name', 'profile')}.3mf"

    if stl_bytes:
        try:
            tmf_bytes = generate_3mf(stl_bytes, profile)
        except Exception as e:
            st.error(f"3MF generation failed: {e}")

    # ── Open in Bambu Studio ──
    st.markdown("#### 🖨️ Open in Bambu Studio")
    st.caption(
        "Saves the .3mf file and opens it in Bambu Studio with your "
        "model and recommended settings pre-loaded."
    )
    if tmf_bytes:
        if st.button(
            "🚀 Open in Bambu Studio",
            type="primary",
            use_container_width=True,
            key="btn_open_studio",
        ):
            filepath = save_3mf_temp(tmf_bytes, tmf_name)
            ok, msg = open_in_bambu_studio(filepath)
            if ok:
                st.success(msg)
            else:
                st.warning(msg)
                st.info(f"File saved at: `{filepath}` — you can open it manually.")

    # ── Downloads ──
    st.markdown("#### 📥 Downloads")
    dl1, dl2, dl3 = st.columns(3)

    dl1.download_button(
        label="⬇️ JSON Profile",
        data=json.dumps(profile, indent=2),
        file_name=f"{profile.get('profile_name', 'profile')}.json",
        mime="application/json",
        use_container_width=True,
    )

    if tmf_bytes:
        dl2.download_button(
            label="⬇️ .3MF File",
            data=tmf_bytes,
            file_name=tmf_name,
            mime="application/vnd.ms-package.3dmanufacturing-3dmodel+xml",
            use_container_width=True,
        )

    config_snip = (
        f"# Profile: {profile.get('profile_name', '')}\n\n"
        f"layer_height = {qual.get('layer_height', 0.2)}\n"
        f"first_layer_height = {qual.get('first_layer_height', 0.2)}\n"
        f"wall_loops = {stren.get('wall_loops', 3)}\n"
        f"top_shell_layers = {stren.get('top_shell_layers', 4)}\n"
        f"bottom_shell_layers = {stren.get('bottom_shell_layers', 4)}\n"
        f"sparse_infill_density = {stren.get('infill_density', 15)}%\n"
        f"sparse_infill_pattern = {stren.get('infill_pattern', 'grid')}\n"
        f"outer_wall_speed = {spd.get('outer_wall_speed', 150)}\n"
        f"inner_wall_speed = {spd.get('inner_wall_speed', 180)}\n"
        f"sparse_infill_speed = {spd.get('infill_speed', 250)}\n"
        f"travel_speed = {spd.get('travel_speed', 350)}\n"
        f"initial_layer_speed = {spd.get('first_layer_speed', 50)}\n"
        f"nozzle_temperature = {mat.get('nozzle_temperature', 220)}\n"
        f"bed_temperature = {mat.get('bed_temperature', 55)}\n"
    )
    dl3.download_button(
        label="⬇️ Config Snippet",
        data=config_snip,
        file_name=f"{profile.get('profile_name', 'profile')}_config.ini",
        mime="text/plain",
        use_container_width=True,
    )

    with st.expander("🧾 Raw JSON"):
        st.json(profile)


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    st.markdown(
        """<style>
        .stApp { max-width: 1100px; margin: 0 auto; }
        div[data-testid="stMetric"] { background:#f8f9fa; padding:12px; border-radius:8px; }
        .warning-box { background:#fff3cd; border-left:4px solid #ffc107; padding:12px; border-radius:4px; margin:8px 0; }
        .tip-box { background:#d1ecf1; border-left:4px solid #17a2b8; padding:12px; border-radius:4px; margin:8px 0; }
        .success-box { background:#d4edda; border-left:4px solid #28a745; padding:12px; border-radius:4px; margin:8px 0; }
        </style>""",
        unsafe_allow_html=True,
    )

    st.title("MakerSpace Print Advisor")
    st.caption("Upload your STL, describe what you need, get a safe Bambu Lab print profile.")

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        st.error("**GOOGLE_API_KEY not found.** Create a `.env` file with: `GOOGLE_API_KEY=your_key`")
        st.stop()

    # Sidebar
    with st.sidebar:
        st.header("Printer")
        printer = st.selectbox("Printer", ["X1C", "P1S", "A1", "A1 mini"])
        nozzle_opts = [0.4, 0.6, 0.8] if printer in ["X1C", "P1S"] else [0.4]
        nozzle = st.selectbox("Nozzle (mm)", nozzle_opts)
        st.divider()
        info = {
            "X1C": "256x256x256 | Enclosed | Lidar",
            "P1S": "256x256x256 | Enclosed",
            "A1": "256x256x256 | Open",
            "A1 mini": "180x180x180 | Open",
        }
        st.caption(info.get(printer, ""))

    # Upload & Prompt
    st.subheader("Upload & Describe")
    col1, col2 = st.columns([1, 1])

    with col1:
        uploaded = st.file_uploader("Upload STL", type=["stl"])
        if uploaded:
            st.caption(f"📎 {uploaded.name} ({uploaded.size / 1024:.1f} KB)")

    with col2:
        description = st.text_area(
            "What are you printing?",
            placeholder='e.g. "Strong motor bracket" or "Quick test fit prototype"',
            height=150,
        )

    can_go = bool(uploaded and description)
    go = st.button("Generate Profile", type="primary", use_container_width=True, disabled=not can_go)

    if not can_go:
        st.info("Upload an STL and describe your print to get started.")

    # Generate
    if go and uploaded and description:
        stl_bytes = uploaded.read()

        with st.spinner("Analyzing geometry..."):
            stl_info = analyze_stl(stl_bytes)

        if not stl_info:
            st.error("Could not parse STL. Check the file is valid.")
            return

        st.divider()
        st.subheader("Model Analysis")
        d = stl_info["dimensions_mm"]
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("X", f"{d['x']} mm")
        m2.metric("Y", f"{d['y']} mm")
        m3.metric("Z", f"{d['z']} mm")
        m4.metric("Volume", f"{stl_info['volume_cm3']} cm³")
        m5.metric("Triangles", f"{stl_info['triangle_count']:,}")

        fit_key = f"fits_{printer.lower().replace(' ', '_')}"
        if not stl_info.get(fit_key, True):
            st.error(f"Model may NOT fit on {printer}!")
        if stl_info["is_tall"]:
            st.warning("Tall print — extra adhesion will be added.")

        st.divider()
        st.subheader("Recommended Print Profile")

        with st.spinner("Generating optimized settings..."):
            try:
                info = dict(stl_info)
                info["nozzle_size_mm"] = nozzle
                profile = call_llm(info, description, printer)
                st.session_state["last_profile"] = profile
                st.session_state["last_stl"] = stl_bytes
                st.session_state["last_name"] = uploaded.name
            except json.JSONDecodeError as e:
                st.error(f"Profile generation failed. Try again.\n\nDetail: {e}")
                with st.expander("Debug"):
                    st.code(traceback.format_exc())
                return
            except ValueError as e:
                st.error(str(e))
                return
            except Exception as e:
                st.error(f"Error: {e}")
                with st.expander("Debug"):
                    st.code(traceback.format_exc())
                return

        display_profile(profile, stl_bytes, uploaded.name)

    elif "last_profile" in st.session_state and not go:
        st.divider()
        st.subheader("Recommended Print Profile")
        st.caption("Showing last generated profile.")
        display_profile(
            st.session_state["last_profile"],
            st.session_state.get("last_stl"),
            st.session_state.get("last_name", "model"),
        )


if __name__ == "__main__":
    main()
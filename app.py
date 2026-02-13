import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial import KDTree
import io, os
import zipfile
from streamlit_cropper import st_cropper

# --- CONSTANTS ---
SYMBOLS = "1234567890ABCDEFGHIJKLMNOPQRSTUVWXYZ@#$%&+*<>?ΩΣΦ"
CELL_SIZE_PX = 30  # Controls the resolution of the printable tiles

# --- PAGE CONFIG ---
if 'zip_ready' not in st.session_state:
    st.session_state.zip_ready = False
st.set_page_config(page_title="Diamond Art Canvas Generator", layout="wide")
st.markdown("""
    <style>
        [data-testid="column"] { height: auto !important; }
        [data-testid="column"]:nth-child(2) {
            position: -webkit-sticky; position: sticky;
            top: 20px; align-self: start;
        }
    </style>
    """, unsafe_allow_html=True)

def slice_pattern_for_printing(pattern_img, zf):
    # 300 DPI calculations
    PAGE_W, PAGE_H = 2550, 3300  # Full 8.5x11
    # Safe Printable Area (8" x 10.5") to avoid printer auto-scaling
    SAFE_W, SAFE_H = 2400, 3150  
    
    img_w, img_h = pattern_img.size
    rows = (img_h // SAFE_H) + 1
    cols = (img_w // SAFE_W) + 1

    for r in range(rows):
        for c in range(cols):
            # Crop the pattern based on the SAFE area
            left, top = c * SAFE_W, r * SAFE_H
            right, bottom = min(left + SAFE_W, img_w), min(top + SAFE_H, img_h)
            tile = pattern_img.crop((left, top, right, bottom))
            
            # Paste onto a full-sized white page
            page = Image.new('RGB', (PAGE_W, PAGE_H), 'white')
            # Center the tile on the page
            paste_x = (PAGE_W - (right - left)) // 2
            paste_y = (PAGE_H - (bottom - top)) // 2
            page.paste(tile, (paste_x, paste_y))
            
            # Save to ZIP
            t_buf = io.BytesIO()
            page.save(t_buf, format="PNG", dpi=(300, 300))
            zf.writestr(f"tiles/Sheet_{r+1}_{c+1}.png", t_buf.getvalue())

# --- CORE LOGIC FUNCTIONS ---
def load_dmc_data(df):
    # Map CSV headers to logic keys
    rgb_points = df[['Red', 'Green', 'Blue']].values
    dmc_info = [{"code": str(row['Floss#']), "name": row['Description']} for _, row in df.iterrows()]
    return rgb_points, dmc_info

def prune_palette(rgb_points, dmc_info, threshold):
    if threshold <= 0: return rgb_points, dmc_info
    pruned_rgbs, pruned_info = [], []
    for i, color in enumerate(rgb_points):
        if not pruned_rgbs or KDTree(pruned_rgbs).query(color)[0] > threshold:
            pruned_rgbs.append(color)
            pruned_info.append(dmc_info[i])
    return np.array(pruned_rgbs), pruned_info

def create_printable_legend(counts, symbol_map, pruned_rgbs, pruned_info):
    line_h = 60
    padding = 40
    img_h = (len(counts) * line_h) + 150
    img = Image.new('RGB', (1200, img_h), 'white')
    draw = ImageDraw.Draw(img)
    
    try:
        font = ImageFont.truetype("Roboto-Regular.ttf", 35)
        header_font = ImageFont.truetype("Roboto-Regular.ttf", 50)
    except:
        font = ImageFont.load_default()
        header_font = ImageFont.load_default()

    draw.text((padding, 20), "DIAMOND PAINTING KEY", fill='black', font=header_font)
    draw.line([(padding, 100), (1100, 100)], fill='black', width=3)

    for i, (code, count) in enumerate(sorted(counts.items(), key=lambda x: x[1], reverse=True)):
        y = 130 + (i * line_h)
        symbol = symbol_map[code]
        color_idx = next(idx for idx, info in enumerate(pruned_info) if info["code"] == code)
        rgb = tuple(pruned_rgbs[color_idx].astype(int))
        
        # Color Box
        draw.rectangle([padding, y, padding + 50, y + 50], fill=rgb, outline='black')
        # Symbol in Box
        brightness = sum(rgb) / 3
        symbol_color = 'white' if brightness < 128 else 'black'
        draw.text((padding + 12, y + 2), symbol, fill=symbol_color, font=font)
        # Label
        text_label = f"DMC {code:<5} | {count:>5} drills | {pruned_info[color_idx]['name']}"
        draw.text((padding + 80, y + 2), text_label, fill='black', font=font)
    return img

def process_image(cropped_img, threshold, min_drills, width_cm, height_cm, df_dmc):
    raw_rgbs, raw_info = load_dmc_data(df_dmc)
    pruned_rgbs, pruned_info = prune_palette(raw_rgbs, raw_info, threshold)
    tree = KDTree(pruned_rgbs)
    
    w_cells = int((width_cm * 10) / 2.5)
    h_cells = int((height_cm * 10) / 2.5)
    img_small = cropped_img.resize((w_cells, h_cells), Image.Resampling.BOX)
    pixels = np.array(img_small)
    
    # Matching
    flat_pixels = pixels.reshape(-1, 3)
    _, indices = tree.query(flat_pixels)
    counts = {}
    for idx in indices:
        code = pruned_info[idx]["code"]
        counts[code] = counts.get(code, 0) + 1

    # Common Logic
    common_idx = [i for i, info in enumerate(pruned_info) if counts.get(info["code"], 0) >= min_drills]
    common_tree = KDTree([pruned_rgbs[i] for i in common_idx])
    
    final_grid_codes = []
    final_counts = {}
    
    for r in range(h_cells):
        grid_row = []
        for c in range(w_cells):
            p = pixels[r, c]
            orig_idx = indices[r * w_cells + c]
            if counts[pruned_info[orig_idx]["code"]] < min_drills:
                _, c_idx = common_tree.query(p)
                final_idx = common_idx[c_idx]
            else:
                final_idx = orig_idx
            
            code = pruned_info[final_idx]["code"]
            grid_row.append(code)
            final_counts[code] = final_counts.get(code, 0) + 1
        final_grid_codes.append(grid_row)

    symbol_map = {code: SYMBOLS[i % len(SYMBOLS)] for i, code in enumerate(sorted(final_counts.keys(), key=lambda x: final_counts[x], reverse=True))}
    
    # Create Pattern
    pattern_img = Image.new('RGB', (w_cells * CELL_SIZE_PX, h_cells * CELL_SIZE_PX), 'white')
    draw = ImageDraw.Draw(pattern_img)
    try: font = ImageFont.truetype("Roboto-Regular.ttf", 18)
    except: font = ImageFont.load_default()

    for y in range(h_cells):
        for x in range(w_cells):
            code = final_grid_codes[y][x]
            color_idx = next(idx for idx, info in enumerate(pruned_info) if info["code"] == code)
            rgb = tuple(pruned_rgbs[color_idx].astype(int))
            x0, y0 = x * CELL_SIZE_PX, y * CELL_SIZE_PX
            draw.rectangle([x0, y0, x0+CELL_SIZE_PX, y0+CELL_SIZE_PX], fill=rgb, outline=(150,150,150))
            brightness = sum(rgb) / 3
            draw.text((x0+8, y0+4), symbol_map[code], fill=('white' if brightness < 128 else 'black'), font=font)

    return pattern_img, final_counts, symbol_map, pruned_rgbs, pruned_info

# --- UI SIDEBAR ---
st.sidebar.title("⚙️ Settings")
base_sizes = {
    "Small (30x40 cm)": (30, 40),
    "Medium (40x50 cm)": (40, 50),
    "Large (60x80 cm)": (60, 80),
    "XL (60x90 cm)": (60, 90)
}
selected_label = st.sidebar.selectbox("Canvas Size", list(base_sizes.keys()), key="main_size_picker")
orientation = st.sidebar.radio("Orientation", ["Portrait (Tall)", "Landscape (Wide)"])

# Extract dimensions
w_cm, h_cm = base_sizes[selected_label]

# Flip if Landscape
if orientation == "Landscape (Wide)":
    width_cm, height_cm = h_cm, w_cm  # Swap values
else:
    width_cm, height_cm = w_cm, h_cm

current_aspect_ratio = (width_cm, height_cm)

threshold = st.sidebar.slider("Color Merging", 0, 50, 25)
min_drills = st.sidebar.number_input("Min Drills", value=100)

# --- MAIN INTERFACE ---
st.title("💎 Diamond Art Pro")
DEFAULT_CSV = "dmc-floss.csv"
df_dmc = pd.read_csv(DEFAULT_CSV) if os.path.exists(DEFAULT_CSV) else None

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file and df_dmc is not None:
    img = Image.open(uploaded_file).convert("RGB")
    selected_label = st.sidebar.selectbox("Canvas Size", list(base_sizes.keys()))

    width_cm, height_cm = base_sizes[selected_label]
    current_aspect_ratio = (width_cm, height_cm)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("1. Adjust Your Crop")
        cropped_img = st_cropper(
            img, 
            aspect_ratio=current_aspect_ratio, 
            realtime_update=True, 
            box_color="#F8F8F8"
        )

    with col2:
        st.subheader("2. Diamond Art Preview")
        # Whenever this runs, it means a setting changed or the crop moved.
        # We set zip_ready to False to hide the old download button.
        st.session_state.zip_ready = False
        
        pattern_img, final_counts, symbol_map, p_rgbs, p_info = process_image(
            cropped_img, threshold, min_drills, width_cm, height_cm, df_dmc
        )
        st.image(pattern_img, use_container_width=True)

    st.divider()    

if st.button("🎁 Prepare Download Package"):
    # Export Logic
    st.divider()
    try: font = ImageFont.truetype("Roboto-Regular.ttf", 18)
    except: font = ImageFont.load_default()
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w") as zf:
        # Save Legend
        legend = create_printable_legend(final_counts, symbol_map, p_rgbs, p_info)
        leg_buf = io.BytesIO(); legend.save(leg_buf, format="PNG")
        zf.writestr("00_LEGEND.png", leg_buf.getvalue())
        
        # Save Tiles (8.5x11 at 300DPI)
        # 1. Calculate Buffer in Pixels
        # Since 2.5mm = 1 drill = CELL_SIZE_PX (30px)
        # 3cm (30mm) = 12 drills
        buffer_drills = 12 
        buffer_px = buffer_drills * CELL_SIZE_PX

        # 2. Create the Buffered Image
        iw, ih = pattern_img.size
        buffered_w = iw + (buffer_px * 2)
        buffered_h = ih + (buffer_px * 2)

        buffered_img = Image.new('RGB', (buffered_w, buffered_h), 'white')
        # Paste the original pattern into the center
        buffered_img.paste(pattern_img, (buffer_px, buffer_px))
        
        img_buf = io.BytesIO(); buffered_img.save(img_buf, format="PNG")
        zf.writestr("outline.png", img_buf.getvalue())
        # 3. Slice the BUFFERED image instead of the original
        pw, ph = 2400, 3100 # 8.5x11 at 300 DPI
        slice_pattern_for_printing(buffered_img, zf)

    # Store the result in session state so it persists
    st.session_state.zip_data = zip_buf.getvalue()
    st.session_state.zip_ready = True
    st.success("Package ready!")
            
if st.session_state.zip_ready:
    st.download_button(
        label="💾 Download ZIP", 
        data=st.session_state.zip_data, 
        file_name="DiamondArt_Kit.zip", 
        mime="application/zip"

    )






import streamlit as st
from core.group_parser import parse_group
from core.visuals import (
    generate_tiling_visual,
    generate_polyhedron_visual,
    generate_fractal_visual,
    generate_animation_visual,
    plot_cayley_graph,
    animate_tiling_visual,
    animate_polyhedron_visual,
    animate_fractal_visual,
    apply_group_symmetry_to_image,
    plot_subgroup_lattice,
    plot_element_orders,
    display_conjugacy_classes,
    mandelbrot_fractal,
    julia_fractal,
    penrose_tiling,
    cairo_tiling,
    dodecahedron_visual,
    truncated_icosahedron_visual,
    snub_cube_visual,
    export_svg,
    export_obj
)
from core.group_advanced import subgroup_lattice, element_orders, conjugacy_classes, character_table
from core.explanation import generate_explanation, suggest_next_group
from PIL import Image
import pandas as pd
from streamlit_drawable_canvas import st_canvas
import io
import matplotlib.pyplot as plt
import numpy as np

# --- LLM explanation (OpenAI, fallback to static) ---
def llm_explanation(prompt, api_key=None):
    try:
        import openai
        if api_key:
            openai.api_key = api_key
        else:
            return None
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content.strip()
    except Exception:
        return None

def img_to_bytes(img):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf.getvalue()

def svg_to_bytes(svg_str):
    return svg_str.encode('utf-8')

def obj_to_bytes(obj_str):
    return obj_str.encode('utf-8')

st.set_page_config(page_title="🧠⚛️ AI Symmetry Visualizer", layout="wide")
st.title("🧠⚛️ AI Symmetry Visualizer")
st.markdown("Mesmerizing math meets AI-powered art!\n---")

tabs = st.tabs(["Main", "Advanced Math", "About"])

tiling_type = 'square'
animation_type = 'tiling'
fractal_type = 'sierpinski'
fractal_cmap = 'magma'
fractal_zoom = 1.0
fractal_center = (0.0, 0.0)
julia_c = -0.7 + 0.27015j
solid_type = 'cube'
color_map = 'viridis'
openai_api_key = st.sidebar.text_input("OpenAI API Key (optional for LLM explanations)", type="password")

with st.sidebar:
    st.header("Input")
    group_input = st.text_input("Enter algebraic group or formula (e.g., D4, A5, SO(3))")
    style = st.selectbox("Choose visualization type", ["Tiling", "3D Polyhedron", "Fractal", "Animation", "User Drawing"])
    if style == "Tiling":
        tiling_type = st.selectbox("Tiling type", ["square", "triangle", "hex", "penrose", "cairo"])
        color_map = st.selectbox("Tiling color map", ["viridis", "plasma", "cividis", "twilight"])
    if style == "Fractal":
        fractal_type = st.selectbox("Fractal type", ["sierpinski", "mandelbrot", "julia"])
        fractal_cmap = st.selectbox("Fractal color map", ["magma", "twilight", "viridis", "plasma"])
        fractal_zoom = st.slider("Fractal zoom", 0.5, 5.0, 1.0, 0.1)
        fractal_center_x = st.slider("Fractal center X", -2.0, 2.0, 0.0, 0.01)
        fractal_center_y = st.slider("Fractal center Y", -2.0, 2.0, 0.0, 0.01)
        fractal_center = (fractal_center_x, fractal_center_y)
        if fractal_type == "julia":
            julia_c_real = st.slider("Julia c (real)", -1.0, 1.0, -0.7, 0.01)
            julia_c_imag = st.slider("Julia c (imag)", -1.0, 1.0, 0.27015, 0.01)
            julia_c = complex(julia_c_real, julia_c_imag)
    if style == "3D Polyhedron":
        solid_type = st.selectbox("Solid type", ["cube", "tetrahedron", "octahedron", "icosahedron", "dodecahedron", "truncated icosahedron", "snub cube"])
    if style == "Animation":
        animation_type = st.selectbox("Animation type", ["tiling", "polyhedron", "fractal"])
    generate = st.button("Generate")

if generate:
    group_info = parse_group(group_input)
    with tabs[0]:
        st.info(f"Generating visualization for group: {group_input} (Style: {style})")
        st.write("---")
        st.subheader("🔢 Group Theory Analysis")
        if group_info["error"]:
            st.error(group_info["error"])
        else:
            st.write(f"**Type:** {group_info['type']}")
            st.write(f"**Order:** {group_info['order']}")
            st.write(f"**Generators:** {group_info['generators']}")
            # Cayley table
            if group_info["cayley_table"] and group_info["cayley_elements"] and len(group_info["cayley_table"]) > 0 and len(group_info["cayley_elements"]) > 0:
                st.markdown("**Cayley Table:**")
                df = pd.DataFrame(group_info["cayley_table"], columns=group_info["cayley_elements"], index=group_info["cayley_elements"])
                st.dataframe(df)
            # Cayley graph
            if group_info["cayley_elements"] and group_info["cayley_edges"]:
                st.markdown("**Cayley Graph:**")
                cayley_img = plot_cayley_graph(group_info["cayley_elements"], group_info["cayley_edges"])
                st.image(cayley_img, use_container_width=False)
        st.subheader("🎨 Visual Output")
        if not group_info["error"]:
            img = None
            gif_buf = None
            svg_data = None
            obj_data = None
            if style == "Tiling":
                if tiling_type == "penrose":
                    img = penrose_tiling()
                elif tiling_type == "cairo":
                    img = cairo_tiling()
                else:
                    img = generate_tiling_visual(group_info, tiling_type=tiling_type)
                # SVG export for tiling
                fig = plt.figure()
                plt.imshow(np.zeros((10,10)), cmap=color_map)
                plt.axis('off')
                svg_data = export_svg(fig)
                plt.close(fig)
            elif style == "3D Polyhedron":
                if solid_type == "dodecahedron":
                    img = dodecahedron_visual()
                elif solid_type == "truncated icosahedron":
                    img = truncated_icosahedron_visual()
                elif solid_type == "snub cube":
                    img = snub_cube_visual()
                else:
                    img = generate_polyhedron_visual(group_info)
                # OBJ export for 3D (placeholder)
                obj_data = export_obj([[0,0,0],[1,0,0],[0,1,0]], [[0,1,2]])
            elif style == "Fractal":
                if fractal_type == "mandelbrot":
                    img = mandelbrot_fractal(zoom=fractal_zoom, center=fractal_center, cmap=fractal_cmap)
                elif fractal_type == "julia":
                    img = julia_fractal(zoom=fractal_zoom, center=fractal_center, c=julia_c, cmap=fractal_cmap)
                else:
                    img = generate_fractal_visual(group_info)
                # SVG export for fractal
                fig = plt.figure()
                plt.imshow(np.zeros((10,10)), cmap=fractal_cmap)
                plt.axis('off')
                svg_data = export_svg(fig)
                plt.close(fig)
            elif style == "Animation":
                if animation_type == "tiling":
                    gif_buf = animate_tiling_visual(group_info, tiling_type=tiling_type)
                elif animation_type == "polyhedron":
                    gif_buf = animate_polyhedron_visual(group_info)
                elif animation_type == "fractal":
                    gif_buf = animate_fractal_visual(group_info)
            elif style == "User Drawing":
                st.markdown("**Draw a shape below:**")
                canvas_result = st_canvas(
                    fill_color="rgba(255, 165, 0, 0.3)",
                    stroke_width=5,
                    stroke_color="#000000",
                    background_color="#FFFFFF",
                    width=400,
                    height=400,
                    drawing_mode="freedraw",
                    key="canvas",
                )
                if canvas_result.image_data is not None:
                    img = apply_group_symmetry_to_image(canvas_result.image_data, group_info)
            if img:
                st.image(img, use_container_width=True)
                st.download_button("Download image", data=img_to_bytes(img), file_name="visual.png", mime="image/png")
                if svg_data:
                    st.download_button("Download SVG", data=svg_to_bytes(svg_data), file_name="visual.svg", mime="image/svg+xml")
                if obj_data:
                    st.download_button("Download OBJ", data=obj_to_bytes(obj_data), file_name="visual.obj", mime="text/plain")
            if gif_buf:
                st.image(gif_buf, use_container_width=True, format="GIF")
                st.download_button("Download GIF", data=gif_buf, file_name="animation.gif", mime="image/gif")
        if not group_info["error"]:
            st.subheader("📝 Explanation")
            # LLM-powered explanation
            prompt = f"Explain the math and art behind this: {group_info['type']} of order {group_info['order']}, style {style}."
            llm_exp = llm_explanation(prompt, api_key=openai_api_key)
            if llm_exp:
                st.write(llm_exp)
            else:
                st.write(generate_explanation(group_info, style))
            st.subheader("💡 AI Group Suggestion")
            # LLM-powered suggestion
            prompt2 = f"Suggest a cool next group to explore after {group_info['type']} of order {group_info['order']}."
            llm_sugg = llm_explanation(prompt2, api_key=openai_api_key)
            if llm_sugg:
                st.write(llm_sugg)
            else:
                st.write(suggest_next_group(group_info))
            st.subheader("📤 Export & Share")
            if img:
                st.download_button("Download image (PNG)", data=img_to_bytes(img), file_name="visual.png", mime="image/png", key="download-png")
            if svg_data:
                st.download_button("Download SVG", data=svg_to_bytes(svg_data), file_name="visual.svg", mime="image/svg+xml", key="download-svg")
            if obj_data:
                st.download_button("Download OBJ", data=obj_to_bytes(obj_data), file_name="visual.obj", mime="text/plain", key="download-obj")
            if gif_buf:
                st.download_button("Download GIF", data=gif_buf, file_name="animation.gif", mime="image/gif", key="download-gif")
    # --- Advanced Math Tab ---
    with tabs[1]:
        st.header("Advanced Group Theory Features")
        if group_info["error"]:
            st.error(group_info["error"])
        else:
            # Reconstruct group object for advanced features
            G = None
            try:
                from sympy.combinatorics.named_groups import DihedralGroup, CyclicGroup, AlternatingGroup, SymmetricGroup
                if group_input.startswith('D') and group_input[1:].isdigit():
                    n = int(group_input[1:])
                    G = DihedralGroup(n)
                elif group_input.startswith('C') and group_input[1:].isdigit():
                    n = int(group_input[1:])
                    G = CyclicGroup(n)
                elif group_input.startswith('A') and group_input[1:].isdigit():
                    n = int(group_input[1:])
                    G = AlternatingGroup(n)
                elif group_input.startswith('S') and group_input[1:].isdigit():
                    n = int(group_input[1:])
                    G = SymmetricGroup(n)
            except Exception:
                pass
            if G:
                st.subheader("Subgroup Lattice")
                try:
                    lattice = subgroup_lattice(G)
                    st.image(plot_subgroup_lattice(lattice), use_container_width=True)
                except Exception as e:
                    st.info(f"Subgroup lattice unavailable: {e}")
                st.subheader("Element Order Distribution")
                try:
                    orders = element_orders(G)
                    st.image(plot_element_orders(orders), use_container_width=True)
                except Exception as e:
                    st.info(f"Element order histogram unavailable: {e}")
                st.subheader("Conjugacy Classes")
                try:
                    classes = conjugacy_classes(G)
                    st.text(display_conjugacy_classes(classes))
                except Exception as e:
                    st.info(f"Conjugacy classes unavailable: {e}")
                st.subheader("Character Table")
                try:
                    chars = character_table(G)
                    if chars:
                        st.dataframe(pd.DataFrame(chars))
                    else:
                        st.info("Character table unavailable for this group.")
                except Exception as e:
                    st.info(f"Character table unavailable: {e}")
            else:
                st.info("Advanced features only available for standard group types (D, C, A, S)")
    # --- About Tab ---
    with tabs[2]:
        st.header("About This Project")
        st.markdown("""
        **AI Symmetry Visualizer**
        
        - Created with ❤️ and mathematics
        - Features advanced group theory, mesmerizing visuals, and AI-powered suggestions
        - Explore, learn, and create!
        """)
else:
    st.write(":sparkles: Enter a group and click Generate to begin!") 
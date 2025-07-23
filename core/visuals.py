import matplotlib.pyplot as plt
import numpy as np
import io
from PIL import Image
import networkx as nx
from matplotlib.patches import RegularPolygon
import plotly.graph_objects as go
import torch
from torchvision import transforms

# Helper for drawing a single tile

def generate_tiling_visual(group_info, tiling_type='square'):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_aspect('equal')
    ax.axis('off')
    n_tiles = 6
    if tiling_type == 'square':
        for i in range(n_tiles):
            for j in range(n_tiles):
                angle = 0
                if group_info.get('type', '').startswith('Dihedral'):
                    # Alternate rotation for dihedral
                    angle = 90 * ((i + j) % 4)
                elif group_info.get('type', '').startswith('Cyclic'):
                    angle = 360 / n_tiles * (i % n_tiles)
                rect = plt.Rectangle((i, j), 1, 1, angle=angle, color=plt.cm.viridis((i+j)/(2*n_tiles)), ec='k')
                ax.add_patch(rect)
        ax.set_xlim(0, n_tiles)
        ax.set_ylim(0, n_tiles)
    elif tiling_type == 'triangle':
        for i in range(n_tiles):
            for j in range(n_tiles):
                x = i + 0.5 * (j % 2)
                y = j * np.sqrt(3)/2
                angle = 0
                if group_info.get('type', '').startswith('Dihedral'):
                    angle = 120 * ((i + j) % 3)
                tri = RegularPolygon((x, y), 3, radius=0.5, orientation=np.deg2rad(angle), color=plt.cm.plasma((i+j)/(2*n_tiles)), ec='k')
                ax.add_patch(tri)
        ax.set_xlim(0, n_tiles)
        ax.set_ylim(0, n_tiles * np.sqrt(3)/2)
    elif tiling_type == 'hex':
        for i in range(n_tiles):
            for j in range(n_tiles):
                x = i * 3/2
                y = np.sqrt(3) * (j + 0.5 * (i % 2))
                angle = 0
                if group_info.get('type', '').startswith('Dihedral'):
                    angle = 60 * ((i + j) % 6)
                hexagon = RegularPolygon((x, y), 6, radius=0.5, orientation=np.deg2rad(angle), color=plt.cm.cividis((i+j)/(2*n_tiles)), ec='k')
                ax.add_patch(hexagon)
        ax.set_xlim(0, n_tiles * 3/2)
        ax.set_ylim(0, n_tiles * np.sqrt(3))
    else:
        ax.text(0.5, 0.5, 'Unknown tiling type', ha='center', va='center')
    fig.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)

def generate_polyhedron_visual(group_info):
    # Map group to platonic solid
    solid = None
    if 'A4' in group_info.get('type', ''):
        solid = 'tetrahedron'
    elif 'S4' in group_info.get('type', ''):
        solid = 'octahedron'
    elif 'A5' in group_info.get('type', ''):
        solid = 'icosahedron'
    else:
        solid = 'cube'
    # Vertices and faces for each solid
    solids = {
        'tetrahedron': {
            'vertices': np.array([[1,1,1], [-1,-1,1], [-1,1,-1], [1,-1,-1]]),
            'faces': [[0,1,2], [0,1,3], [0,2,3], [1,2,3]]
        },
        'cube': {
            'vertices': np.array([[x,y,z] for x in [-1,1] for y in [-1,1] for z in [-1,1]]),
            'faces': [[0,1,3,2],[4,5,7,6],[0,1,5,4],[2,3,7,6],[0,2,6,4],[1,3,7,5]]
        },
        'octahedron': {
            'vertices': np.array([[1,0,0], [-1,0,0], [0,1,0], [0,-1,0], [0,0,1], [0,0,-1]]),
            'faces': [[0,2,4],[2,1,4],[1,3,4],[3,0,4],[0,2,5],[2,1,5],[1,3,5],[3,0,5]]
        },
        'icosahedron': {
            'vertices': np.array([
                [0, 1, 1.618], [0, -1, 1.618], [0, 1, -1.618], [0, -1, -1.618],
                [1, 1.618, 0], [-1, 1.618, 0], [1, -1.618, 0], [-1, -1.618, 0],
                [1.618, 0, 1], [-1.618, 0, 1], [1.618, 0, -1], [-1.618, 0, -1]
            ]),
            'faces': [
                [0,1,8],[0,4,5],[0,5,1],[0,8,4],[1,5,7],[1,7,6],[1,6,8],[2,3,10],[2,4,8],[2,8,10],[2,10,11],[2,11,5],[2,5,4],[3,7,5],[3,5,11],[3,6,7],[3,10,6],[3,11,10],[4,8,9],[4,9,5],[5,9,7],[6,7,9],[6,9,8],[8,9,10],[9,7,10],[10,9,11],[11,9,5]
            ]
        }
    }
    data = solids[solid]
    vertices = data['vertices']
    faces = data['faces']
    fig = go.Figure(data=[go.Mesh3d(
        x=vertices[:,0], y=vertices[:,1], z=vertices[:,2],
        i=[f[0] for f in faces],
        j=[f[1] for f in faces],
        k=[f[2] for f in faces],
        color='lightblue', opacity=0.7
    )])
    fig.update_layout(scene=dict(aspectmode='data'), margin=dict(l=0, r=0, t=0, b=0))
    # Save as static image
    buf = io.BytesIO()
    fig.write_image(buf, format='png')
    buf.seek(0)
    return Image.open(buf)

def sierpinski(ax, p1, p2, p3, depth):
    if depth == 0:
        triangle = plt.Polygon([p1, p2, p3], color='purple', alpha=0.7)
        ax.add_patch(triangle)
    else:
        mid12 = ((p1[0]+p2[0])/2, (p1[1]+p2[1])/2)
        mid23 = ((p2[0]+p3[0])/2, (p2[1]+p3[1])/2)
        mid31 = ((p3[0]+p1[0])/2, (p3[1]+p1[1])/2)
        sierpinski(ax, p1, mid12, mid31, depth-1)
        sierpinski(ax, p2, mid23, mid12, depth-1)
        sierpinski(ax, p3, mid31, mid23, depth-1)

def generate_fractal_visual(group_info):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_aspect('equal')
    ax.axis('off')
    # Sierpinski triangle
    p1 = (0, 0)
    p2 = (1, 0)
    p3 = (0.5, np.sqrt(3)/2)
    sierpinski(ax, p1, p2, p3, depth=5)
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.1, 1)
    fig.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)

def generate_animation_visual(group_info):
    # Placeholder: return a static image for now
    return generate_tiling_visual(group_info)

def animate_tiling_visual(group_info, tiling_type='square', frames=20):
    images = []
    n_tiles = 6
    for frame in range(frames):
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.set_aspect('equal')
        ax.axis('off')
        for i in range(n_tiles):
            for j in range(n_tiles):
                angle = 0
                if group_info.get('type', '').startswith('Dihedral'):
                    angle = 90 * ((i + j + frame) % 4)
                elif group_info.get('type', '').startswith('Cyclic'):
                    angle = (360 / n_tiles) * ((i + frame) % n_tiles)
                else:
                    angle = (360 / frames) * frame
                if tiling_type == 'square':
                    rect = plt.Rectangle((i, j), 1, 1, angle=angle, color=plt.cm.viridis((i+j)/(2*n_tiles)), ec='k')
                    ax.add_patch(rect)
                elif tiling_type == 'triangle':
                    x = i + 0.5 * (j % 2)
                    y = j * np.sqrt(3)/2
                    tri = RegularPolygon((x, y), 3, radius=0.5, orientation=np.deg2rad(angle), color=plt.cm.plasma((i+j)/(2*n_tiles)), ec='k')
                    ax.add_patch(tri)
                elif tiling_type == 'hex':
                    x = i * 3/2
                    y = np.sqrt(3) * (j + 0.5 * (i % 2))
                    hexagon = RegularPolygon((x, y), 6, radius=0.5, orientation=np.deg2rad(angle), color=plt.cm.cividis((i+j)/(2*n_tiles)), ec='k')
                    ax.add_patch(hexagon)
        if tiling_type == 'square':
            ax.set_xlim(0, n_tiles)
            ax.set_ylim(0, n_tiles)
        elif tiling_type == 'triangle':
            ax.set_xlim(0, n_tiles)
            ax.set_ylim(0, n_tiles * np.sqrt(3)/2)
        elif tiling_type == 'hex':
            ax.set_xlim(0, n_tiles * 3/2)
            ax.set_ylim(0, n_tiles * np.sqrt(3))
        fig.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        images.append(Image.open(buf).convert('RGBA'))
    gif_buf = io.BytesIO()
    images[0].save(gif_buf, format='GIF', save_all=True, append_images=images[1:], duration=80, loop=0)
    gif_buf.seek(0)
    return gif_buf

def animate_polyhedron_visual(group_info, frames=20):
    import plotly.graph_objects as go
    import imageio
    solid = None
    if 'A4' in group_info.get('type', ''):
        solid = 'tetrahedron'
    elif 'S4' in group_info.get('type', ''):
        solid = 'octahedron'
    elif 'A5' in group_info.get('type', ''):
        solid = 'icosahedron'
    else:
        solid = 'cube'
    solids = {
        'tetrahedron': {
            'vertices': np.array([[1,1,1], [-1,-1,1], [-1,1,-1], [1,-1,-1]]),
            'faces': [[0,1,2], [0,1,3], [0,2,3], [1,2,3]]
        },
        'cube': {
            'vertices': np.array([[x,y,z] for x in [-1,1] for y in [-1,1] for z in [-1,1]]),
            'faces': [[0,1,3,2],[4,5,7,6],[0,1,5,4],[2,3,7,6],[0,2,6,4],[1,3,7,5]]
        },
        'octahedron': {
            'vertices': np.array([[1,0,0], [-1,0,0], [0,1,0], [0,-1,0], [0,0,1], [0,0,-1]]),
            'faces': [[0,2,4],[2,1,4],[1,3,4],[3,0,4],[0,2,5],[2,1,5],[1,3,5],[3,0,5]]
        },
        'icosahedron': {
            'vertices': np.array([
                [0, 1, 1.618], [0, -1, 1.618], [0, 1, -1.618], [0, -1, -1.618],
                [1, 1.618, 0], [-1, 1.618, 0], [1, -1.618, 0], [-1, -1.618, 0],
                [1.618, 0, 1], [-1.618, 0, 1], [1.618, 0, -1], [-1.618, 0, -1]
            ]),
            'faces': [
                [0,1,8],[0,4,5],[0,5,1],[0,8,4],[1,5,7],[1,7,6],[1,6,8],[2,3,10],[2,4,8],[2,8,10],[2,10,11],[2,11,5],[2,5,4],[3,7,5],[3,5,11],[3,6,7],[3,10,6],[3,11,10],[4,8,9],[4,9,5],[5,9,7],[6,7,9],[6,9,8],[8,9,10],[9,7,10],[10,9,11],[11,9,5]
            ]
        }
    }
    data = solids[solid]
    vertices = data['vertices']
    faces = data['faces']
    images = []
    for frame in range(frames):
        angle = 360 * frame / frames
        fig = go.Figure(data=[go.Mesh3d(
            x=vertices[:,0], y=vertices[:,1], z=vertices[:,2],
            i=[f[0] for f in faces],
            j=[f[1] for f in faces],
            k=[f[2] for f in faces],
            color='lightblue', opacity=0.7
        )])
        fig.update_layout(scene=dict(camera=dict(eye=dict(x=2*np.cos(np.deg2rad(angle)), y=2*np.sin(np.deg2rad(angle)), z=1.2)), aspectmode='data'), margin=dict(l=0, r=0, t=0, b=0))
        buf = io.BytesIO()
        fig.write_image(buf, format='png')
        buf.seek(0)
        images.append(imageio.v2.imread(buf))
    gif_buf = io.BytesIO()
    imageio.mimsave(gif_buf, images, format='GIF', duration=0.08)
    gif_buf.seek(0)
    return gif_buf

def plot_cayley_graph(elements, edges):
    G = nx.MultiDiGraph()
    for i, e in enumerate(elements):
        G.add_node(i, label=str(e))
    for src, tgt, label in edges:
        G.add_edge(src, tgt, label=label)
    pos = nx.spring_layout(G, seed=42)
    fig, ax = plt.subplots(figsize=(5, 5))
    nx.draw(G, pos, ax=ax, with_labels=True, labels={i: str(e) for i, e in enumerate(elements)}, node_color='skyblue', node_size=700, font_size=8, arrows=True)
    edge_labels = {(src, tgt): lbl for src, tgt, lbl in edges}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=7)
    ax.set_title("Cayley Graph")
    ax.axis('off')
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)

def apply_style_transfer(image, style_name):
    # Supported styles: 'candy', 'mosaic', 'rain_princess', 'udnie'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.hub.load('pytorch/vision:v0.10.0', f'fast_neural_style_{style_name}', pretrained=True).to(device).eval()
    preprocess = transforms.Compose([
        transforms.Resize(512),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
    img_tensor = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(img_tensor).cpu().squeeze(0)
    postprocess = transforms.Compose([
        transforms.Lambda(lambda x: x.clamp(0, 255)),
        transforms.ToPILImage()
    ])
    return postprocess(output)

def animate_fractal_visual(group_info, frames=20):
    images = []
    for frame in range(frames):
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.set_aspect('equal')
        ax.axis('off')
        p1 = (0, 0)
        p2 = (1, 0)
        p3 = (0.5, np.sqrt(3)/2)
        depth = 1 + frame * 4 // frames  # Animate from depth 1 to 5
        sierpinski(ax, p1, p2, p3, depth=depth)
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-0.1, 1)
        fig.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        images.append(Image.open(buf).convert('RGBA'))
    gif_buf = io.BytesIO()
    images[0].save(gif_buf, format='GIF', save_all=True, append_images=images[1:], duration=80, loop=0)
    gif_buf.seek(0)
    return gif_buf 

def apply_group_symmetry_to_image(image_data, group_info):
    from PIL import Image, ImageOps
    import numpy as np
    img = Image.fromarray((image_data).astype(np.uint8))
    img = img.convert('RGBA')
    w, h = img.size
    result = Image.new('RGBA', (w*2, h*2), (255,255,255,0))
    group_type = group_info.get('type', '')
    n = 4
    if 'Dihedral' in group_type:
        n = int(group_type.split('D')[-1]) if 'D' in group_type else 4
        for i in range(n):
            rotated = img.rotate(360*i/n)
            result.paste(rotated, (w//2, h//2), rotated)
            flipped = ImageOps.mirror(rotated)
            result.paste(flipped, (w//2, h//2), flipped)
    elif 'Cyclic' in group_type:
        n = int(group_type.split('C')[-1]) if 'C' in group_type else 6
        for i in range(n):
            rotated = img.rotate(360*i/n)
            result.paste(rotated, (w//2, h//2), rotated)
    else:
        # Just tile the drawing
        for i in range(2):
            for j in range(2):
                result.paste(img, (i*w, j*h), img)
    return result 

def plot_subgroup_lattice(lattice):
    """Visualize a subgroup lattice (networkx DiGraph) as a graph image."""
    pos = nx.spring_layout(lattice, seed=42)
    fig, ax = plt.subplots(figsize=(6, 6))
    nx.draw(lattice, pos, ax=ax, with_labels=True, node_color='lightgreen', node_size=800, font_size=8, arrows=True)
    ax.set_title("Subgroup Lattice")
    ax.axis('off')
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)

def plot_element_orders(order_dict):
    """Plot a histogram of element orders."""
    orders = list(order_dict.keys())
    counts = [order_dict[o] for o in orders]
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.bar(orders, counts, color='orchid')
    ax.set_xlabel('Element Order')
    ax.set_ylabel('Count')
    ax.set_title('Element Order Distribution')
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)

def display_conjugacy_classes(classes):
    """Return a string for display of conjugacy classes."""
    return '\n'.join([f"Class {i+1}: {sorted(list(c))}" for i, c in enumerate(classes)]) 

def mandelbrot_fractal(width=400, height=400, zoom=1, center=(0,0), cmap='twilight'): 
    """Draw a Mandelbrot set fractal."""
    x0, y0 = center
    x = np.linspace(x0-2/zoom, x0+2/zoom, width)
    y = np.linspace(y0-2/zoom, y0+2/zoom, height)
    X, Y = np.meshgrid(x, y)
    C = X + 1j*Y
    Z = np.zeros_like(C)
    img = np.zeros(C.shape, dtype=int)
    for i in range(100):
        mask = np.abs(Z) < 2
        Z[mask] = Z[mask]**2 + C[mask]
        img[mask & (np.abs(Z) >= 2)] = i
    fig, ax = plt.subplots(figsize=(5,5))
    ax.imshow(img, cmap=cmap, extent=[x.min(), x.max(), y.min(), y.max()])
    ax.axis('off')
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)

def julia_fractal(width=400, height=400, zoom=1, center=(0,0), c=-0.7+0.27015j, cmap='twilight'): 
    """Draw a Julia set fractal."""
    x0, y0 = center
    x = np.linspace(x0-2/zoom, x0+2/zoom, width)
    y = np.linspace(y0-2/zoom, y0+2/zoom, height)
    X, Y = np.meshgrid(x, y)
    Z = X + 1j*Y
    img = np.zeros(Z.shape, dtype=int)
    for i in range(100):
        mask = np.abs(Z) < 2
        Z[mask] = Z[mask]**2 + c
        img[mask & (np.abs(Z) >= 2)] = i
    fig, ax = plt.subplots(figsize=(5,5))
    ax.imshow(img, cmap=cmap, extent=[x.min(), x.max(), y.min(), y.max()])
    ax.axis('off')
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)

def penrose_tiling(width=400, height=400):
    """Draw a Penrose tiling (simple demo)."""
    fig, ax = plt.subplots(figsize=(5,5))
    ax.set_aspect('equal')
    ax.axis('off')
    for i in range(10):
        for j in range(10):
            angle = np.pi * (i+j)/10
            x = i + 0.5 * np.cos(angle)
            y = j + 0.5 * np.sin(angle)
            ax.plot([x, x+np.cos(angle)], [y, y+np.sin(angle)], color=plt.cm.viridis((i+j)/20))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)

def cairo_tiling(width=400, height=400):
    """Draw a Cairo tiling (simple demo)."""
    fig, ax = plt.subplots(figsize=(5,5))
    ax.set_aspect('equal')
    ax.axis('off')
    for i in range(8):
        for j in range(8):
            x = i + (j%2)*0.5
            y = j * np.sqrt(3)/2
            ax.add_patch(plt.Polygon([(x, y), (x+1, y), (x+1.5, y+np.sqrt(3)/2), (x+0.5, y+np.sqrt(3)/2), (x, y)], closed=True, color=plt.cm.cividis((i+j)/16), ec='k'))
    ax.set_xlim(0, 9)
    ax.set_ylim(0, 8*np.sqrt(3)/2)
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)

def dodecahedron_visual():
    """Render a dodecahedron using plotly."""
    # Vertices and faces omitted for brevity; use a placeholder sphere
    phi = (1 + np.sqrt(5)) / 2
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    x = np.cos(u)*np.sin(v)
    y = np.sin(u)*np.sin(v)
    z = np.cos(v)
    fig = go.Figure(data=[go.Surface(x=x, y=y, z=z, colorscale='Viridis', opacity=0.8)])
    fig.update_layout(scene=dict(aspectmode='data'), margin=dict(l=0, r=0, t=0, b=0))
    buf = io.BytesIO()
    fig.write_image(buf, format='png')
    buf.seek(0)
    return Image.open(buf)

def truncated_icosahedron_visual():
    """Render a truncated icosahedron (soccer ball) using plotly."""
    # Placeholder: use a sphere with lines
    phi = (1 + np.sqrt(5)) / 2
    u, v = np.mgrid[0:2*np.pi:30j, 0:np.pi:15j]
    x = np.cos(u)*np.sin(v)
    y = np.sin(u)*np.sin(v)
    z = np.cos(v)
    fig = go.Figure(data=[go.Surface(x=x, y=y, z=z, colorscale='Cividis', opacity=0.8)])
    fig.update_layout(scene=dict(aspectmode='data'), margin=dict(l=0, r=0, t=0, b=0))
    buf = io.BytesIO()
    fig.write_image(buf, format='png')
    buf.seek(0)
    return Image.open(buf)

def snub_cube_visual():
    """Render a snub cube using plotly."""
    # Placeholder: use a cube
    fig = go.Figure(data=[go.Mesh3d(x=[0,1,1,0,0,1,1,0], y=[0,0,1,1,0,0,1,1], z=[0,0,0,0,1,1,1,1], color='orange', opacity=0.7)])
    fig.update_layout(scene=dict(aspectmode='data'), margin=dict(l=0, r=0, t=0, b=0))
    buf = io.BytesIO()
    fig.write_image(buf, format='png')
    buf.seek(0)
    return Image.open(buf)

# --- SVG/OBJ Export Helpers ---
def export_svg(fig):
    """Export a matplotlib figure as SVG string."""
    buf = io.BytesIO()
    fig.savefig(buf, format='svg')
    buf.seek(0)
    return buf.getvalue().decode('utf-8')

def export_obj(vertices, faces):
    """Export 3D mesh as OBJ string."""
    lines = []
    for v in vertices:
        lines.append(f"v {v[0]} {v[1]} {v[2]}")
    for f in faces:
        lines.append("f " + " ".join(str(i+1) for i in f))
    return "\n".join(lines) 
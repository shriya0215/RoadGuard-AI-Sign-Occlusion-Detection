"""
RoadGuard AI - Synthetic Occlusion Generator
Generates realistic physical occlusions (Leaves, Mud, Graffiti) on traffic sign images
using OpenCV and NumPy with realistic textures, opacity, and intensity scaling.
"""

import cv2
import numpy as np
import random


def apply_leaves_occlusion(image: np.ndarray, intensity: float = 0.5, seed: int = None) -> np.ndarray:
    """
    Overlays realistic foliage/leaves over a traffic sign image.
    
    Args:
        image: Input image (RGB or BGR, HxWxC, uint8)
        intensity: Float between 0.1 and 1.0 (controls leaf count, size, and clustering)
        seed: Optional random seed for reproducibility
        
    Returns:
        Occluded image (same shape and dtype as input)
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    h, w = image.shape[:2]
    result = image.copy().astype(np.float32)
    
    # Calculate leaf parameters based on intensity
    intensity = max(0.1, min(1.0, float(intensity)))
    min_leaves = max(2, int(4 * intensity))
    max_leaves = max(min_leaves + 1, int(15 * intensity))
    num_leaves = random.randint(min_leaves, max_leaves)
    
    # Cluster center on the sign to simulate branch or cluster hanging in
    cluster_x = random.choice([
        random.randint(0, int(w * 0.4)),
        random.randint(int(w * 0.6), w),
        random.randint(int(w * 0.2), int(w * 0.8))
    ])
    cluster_y = random.choice([
        random.randint(0, int(h * 0.4)),
        random.randint(int(h * 0.6), h),
        random.randint(int(h * 0.2), int(h * 0.8))
    ])

    # Leaf palette: varied greens, autumn yellows, olive browns
    leaf_palette = [
        (34, 139, 34),    # Forest green
        (46, 125, 50),    # Dark leaf green
        (107, 142, 35),   # Olive drab
        (85, 107, 47),    # Dark olive
        (124, 179, 66),   # Fresh green
        (160, 130, 40),   # Dry yellow-green
        (90, 70, 30),     # Autumn brown-green
    ]
    
    leaf_layer = np.zeros_like(result)
    mask_layer = np.zeros((h, w), dtype=np.float32)
    
    base_size = max(4, int(min(h, w) * (0.15 + 0.25 * intensity)))
    
    for _ in range(num_leaves):
        # Position leaf near cluster center with Gaussian jitter
        lx = int(np.clip(np.random.normal(cluster_x, w * 0.22), 0, w - 1))
        ly = int(np.clip(np.random.normal(cluster_y, h * 0.22), 0, h - 1))
        
        # Varied leaf size and rotation angle
        leaf_w = int(base_size * random.uniform(0.6, 1.4))
        leaf_h = int(leaf_w * random.uniform(0.4, 0.75))
        angle = random.uniform(0, 180)
        
        color = random.choice(leaf_palette)
        # Add slight color fluctuation
        color = tuple(int(np.clip(c + random.randint(-15, 15), 0, 255)) for c in color)
        
        # Draw pointed oval leaf using filled ellipse
        leaf_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.ellipse(leaf_mask, (lx, ly), (leaf_w, leaf_h), angle, 0, 360, 255, -1)
        
        # Draw center vein
        rad = np.radians(angle)
        dx = int(leaf_w * 0.9 * np.cos(rad))
        dy = int(leaf_w * 0.9 * np.sin(rad))
        cv2.line(leaf_mask, (lx - dx, ly - dy), (lx + dx, ly + dy), 220, max(1, int(leaf_h * 0.15)))
        
        # Slight blur for organic leaf edges
        ksize = max(3, int(leaf_h * 0.25) | 1)
        leaf_mask = cv2.GaussianBlur(leaf_mask, (ksize, ksize), 0)
        norm_mask = leaf_mask.astype(np.float32) / 255.0
        
        # Blend onto leaf layer
        for c in range(3):
            leaf_layer[:, :, c] = leaf_layer[:, :, c] * (1.0 - norm_mask) + color[c] * norm_mask
        mask_layer = np.maximum(mask_layer, norm_mask)

    # Add branch/twig
    if intensity > 0.35:
        twig_color = (60, 42, 20)
        twig_mask = np.zeros((h, w), dtype=np.uint8)
        pt1 = (cluster_x + random.randint(-10, 10), 0 if cluster_y < h // 2 else h)
        pt2 = (cluster_x, cluster_y)
        thickness = max(1, int(min(h, w) * 0.035 * intensity))
        cv2.line(twig_mask, pt1, pt2, 255, thickness)
        twig_mask_f = (cv2.GaussianBlur(twig_mask, (3, 3), 0) / 255.0)
        for c in range(3):
            leaf_layer[:, :, c] = leaf_layer[:, :, c] * (1.0 - twig_mask_f) + twig_color[c] * twig_mask_f
        mask_layer = np.maximum(mask_layer, twig_mask_f)

    # Composite leaf layer with original image
    mask_3d = np.repeat(mask_layer[:, :, np.newaxis], 3, axis=2)
    # Realistic opacity factor
    alpha = min(0.96, 0.70 + 0.28 * intensity)
    blended = result * (1.0 - mask_3d * alpha) + leaf_layer * (mask_3d * alpha)
    
    return np.clip(blended, 0, 255).astype(np.uint8)


def apply_mud_occlusion(image: np.ndarray, intensity: float = 0.5, seed: int = None) -> np.ndarray:
    """
    Overlays realistic mud splatters, dirt smudges, and dried grime over a traffic sign.
    
    Args:
        image: Input image (RGB or BGR, HxWxC, uint8)
        intensity: Float between 0.1 and 1.0 (controls mud volume and coverage)
        seed: Optional random seed for reproducibility
        
    Returns:
        Occluded image (same shape and dtype as input)
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    h, w = image.shape[:2]
    result = image.copy().astype(np.float32)
    
    intensity = max(0.1, min(1.0, float(intensity)))
    
    # Mud color tones (deep brown, clay red-brown, dried sandy brown)
    mud_tones = [
        (65, 42, 25),    # Wet dark mud
        (82, 54, 34),    # Rich loam brown
        (105, 75, 48),   # Clay brown
        (130, 95, 65),   # Dried dirt
        (48, 32, 18),    # Deep muck
    ]
    
    mud_canvas = np.zeros_like(result)
    density_mask = np.zeros((h, w), dtype=np.float32)
    
    # 1. Main splatter blobs (splat centers)
    num_large_splats = max(1, int(3 * intensity))
    for _ in range(num_large_splats):
        cx = random.randint(int(w * 0.15), int(w * 0.85))
        cy = random.randint(int(h * 0.15), int(h * 0.85))
        base_radius = int(min(h, w) * (0.12 + 0.22 * intensity))
        
        # Deform circle using random polygons to simulate organic splash
        num_pts = random.randint(8, 14)
        angles = np.linspace(0, 2 * np.pi, num_pts, endpoint=False)
        radii = [base_radius * random.uniform(0.65, 1.45) for _ in range(num_pts)]
        pts = np.array([
            [int(cx + r * np.cos(a)), int(cy + r * np.sin(a))]
            for a, r in zip(angles, radii)
        ], dtype=np.int32)
        
        color = random.choice(mud_tones)
        blob_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(blob_mask, [pts], 255)
        
        # Smooth and texture
        k = max(3, int(base_radius * 0.35) | 1)
        blob_mask = cv2.GaussianBlur(blob_mask, (k, k), 0)
        blob_f = blob_mask.astype(np.float32) / 255.0
        
        for c in range(3):
            mud_canvas[:, :, c] = mud_canvas[:, :, c] * (1.0 - blob_f) + color[c] * blob_f
        density_mask = np.maximum(density_mask, blob_f)
    
    # 2. Fine splatter droplets around splashes
    num_droplets = max(8, int(60 * intensity))
    for _ in range(num_droplets):
        dx = random.randint(0, w - 1)
        dy = random.randint(0, h - 1)
        drop_r = max(1, int(random.uniform(1.0, 3.5 + 2.5 * intensity)))
        color = random.choice(mud_tones)
        
        drop_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(drop_mask, (dx, dy), drop_r, 255, -1)
        drop_f = (cv2.GaussianBlur(drop_mask, (3, 3), 0) / 255.0) * random.uniform(0.6, 0.95)
        
        for c in range(3):
            mud_canvas[:, :, c] = mud_canvas[:, :, c] * (1.0 - drop_f) + color[c] * drop_f
        density_mask = np.maximum(density_mask, drop_f)

    # 3. Add gritty high-frequency texture noise to mud
    noise = np.random.normal(0, 18, (h, w, 3)).astype(np.float32)
    mud_canvas = np.clip(mud_canvas + noise, 0, 255)
    
    # 4. Composite mud onto sign
    mask_3d = np.repeat(density_mask[:, :, np.newaxis], 3, axis=2)
    alpha = min(0.98, 0.75 + 0.23 * intensity)
    blended = result * (1.0 - mask_3d * alpha) + mud_canvas * (mask_3d * alpha)
    
    return np.clip(blended, 0, 255).astype(np.uint8)


def apply_graffiti_occlusion(image: np.ndarray, intensity: float = 0.5, seed: int = None) -> np.ndarray:
    """
    Overlays realistic urban spray paint, markers, and graffiti strokes over a traffic sign.
    
    Args:
        image: Input image (RGB or BGR, HxWxC, uint8)
        intensity: Float between 0.1 and 1.0 (controls stroke width, number of tags/scribbles)
        seed: Optional random seed for reproducibility
        
    Returns:
        Occluded image (same shape and dtype as input)
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    h, w = image.shape[:2]
    result = image.copy().astype(np.float32)
    
    intensity = max(0.1, min(1.0, float(intensity)))
    
    # High-contrast spray paint & marker colors
    graffiti_palette = [
        (230, 25, 30),    # Spray red
        (20, 20, 20),     # Matte black marker
        (30, 144, 255),   # Dodger cyan/blue
        (255, 215, 0),    # Bright yellow
        (255, 20, 147),   # Hot pink / magenta
        (245, 245, 245),  # Chalk/primer white
        (50, 205, 50),    # Lime green
    ]
    
    spray_canvas = np.zeros_like(result)
    stroke_mask = np.zeros((h, w), dtype=np.float32)
    
    primary_color = random.choice(graffiti_palette)
    secondary_color = random.choice([c for c in graffiti_palette if c != primary_color])
    
    base_thickness = max(2, int(min(h, w) * (0.05 + 0.10 * intensity)))
    
    # 1. Scribble strokes / spray curves
    num_strokes = max(1, int(4 * intensity))
    for s_idx in range(num_strokes):
        color = primary_color if s_idx % 2 == 0 else secondary_color
        
        # Generate continuous jagged / curvy polyline
        num_points = random.randint(4, 7)
        points = []
        for _ in range(num_points):
            px = random.randint(int(w * 0.1), int(w * 0.9))
            py = random.randint(int(h * 0.1), int(h * 0.9))
            points.append([px, py])
        points = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
        
        line_mask = np.zeros((h, w), dtype=np.uint8)
        # Use random polyline or crossing 'X' / slash mark
        if random.random() < 0.4:
            # Giant tag mark / cross out
            p1 = (random.randint(0, int(w * 0.3)), random.randint(0, int(h * 0.3)))
            p2 = (random.randint(int(w * 0.7), w), random.randint(int(h * 0.7), h))
            cv2.line(line_mask, p1, p2, 255, base_thickness)
            if random.random() < 0.5:
                p3 = (random.randint(0, int(w * 0.3)), random.randint(int(h * 0.7), h))
                p4 = (random.randint(int(w * 0.7), w), random.randint(0, int(h * 0.3)))
                cv2.line(line_mask, p3, p4, 255, base_thickness)
        else:
            cv2.polylines(line_mask, [points], isClosed=False, color=255, thickness=base_thickness)
            
        # Spray paint aerosol dispersion effect (soft fuzzy edge)
        spray_blur = cv2.GaussianBlur(line_mask, (max(3, base_thickness | 1), max(3, base_thickness | 1)), 0)
        norm_spray = spray_blur.astype(np.float32) / 255.0
        
        for c in range(3):
            spray_canvas[:, :, c] = spray_canvas[:, :, c] * (1.0 - norm_spray) + color[c] * norm_spray
        stroke_mask = np.maximum(stroke_mask, norm_spray)

    # 2. Spray overspray / paint drip if high intensity
    if intensity > 0.45:
        num_drips = random.randint(1, max(2, int(4 * intensity)))
        for _ in range(num_drips):
            drip_x = random.randint(int(w * 0.2), int(w * 0.8))
            start_y = random.randint(int(h * 0.3), int(h * 0.6))
            end_y = min(h - 1, start_y + int(h * random.uniform(0.2, 0.45 * intensity)))
            drip_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.line(drip_mask, (drip_x, start_y), (drip_x, end_y), 255, max(1, base_thickness // 3))
            cv2.circle(drip_mask, (drip_x, end_y), max(1, base_thickness // 2), 255, -1)
            drip_f = cv2.GaussianBlur(drip_mask, (3, 3), 0) / 255.0
            for c in range(3):
                spray_canvas[:, :, c] = spray_canvas[:, :, c] * (1.0 - drip_f) + primary_color[c] * drip_f
            stroke_mask = np.maximum(stroke_mask, drip_f)

    # Composite onto sign
    mask_3d = np.repeat(stroke_mask[:, :, np.newaxis], 3, axis=2)
    alpha = min(0.96, 0.80 + 0.18 * intensity)
    blended = result * (1.0 - mask_3d * alpha) + spray_canvas * (mask_3d * alpha)
    
    return np.clip(blended, 0, 255).astype(np.uint8)


def apply_synthetic_occlusion(
    image: np.ndarray,
    occlusion_type: str = "leaves",
    intensity: float = 0.5,
    seed: int = None
) -> np.ndarray:
    """
    Unified function to apply any supported occlusion type.
    
    Args:
        image: Input image (RGB or BGR, HxWxC, uint8)
        occlusion_type: One of 'leaves', 'mud', 'graffiti', or 'random'
        intensity: Float between 0.1 and 1.0
        seed: Optional random seed
        
    Returns:
        Occluded image
    """
    occ = occlusion_type.lower().strip()
    if occ == "random":
        occ = random.choice(["leaves", "mud", "graffiti"])
        
    if occ == "leaves":
        return apply_leaves_occlusion(image, intensity=intensity, seed=seed)
    elif occ == "mud":
        return apply_mud_occlusion(image, intensity=intensity, seed=seed)
    elif occ == "graffiti":
        return apply_graffiti_occlusion(image, intensity=intensity, seed=seed)
    else:
        # Default fallback to leaves
        return apply_leaves_occlusion(image, intensity=intensity, seed=seed)

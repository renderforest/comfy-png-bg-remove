# My bg remove for ComfyUI

One node for logos and images, background of any color:
- Detects the background color itself, removes it (also inside letter holes), removes floor shadows.
- Clean edges even after RTX / ESRGAN 4x upscale: the dark/light halos upscalers draw around shapes are removed.
- One-tone logos (white logo on dark, black on white, glass / glow effects) get a smooth "glass" matte.
- Multi-color logos are cut out solid; white highlights inside the logo stay, letter holes and gaps go.
- Photos / busy backgrounds: connect an AI mask (Remove Background Pixaroma -> mask).

Install: copy this folder into ComfyUI/custom_nodes/ and restart ComfyUI.
Node: double-click canvas -> "My bg remove".

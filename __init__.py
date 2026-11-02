import numpy as np
import torch
from .core import process


class MyBgRemove:
    """Universal background removal.
    Solid background (any color) -> automatic color key with shadow removal and edge cleanup.
    Busy background (photos)   -> uses the connected AI mask (e.g. Remove Background Pixaroma)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mode": (["auto", "color key", "ai mask"], {"default": "auto",
                         "tooltip": "auto: color key when the background is one solid color, otherwise the AI mask."}),
                "tolerance": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 255.0, "step": 1.0,
                              "tooltip": "0 = auto. How close to the background color a pixel must be to be removed. Background left over -> raise (40-80). Holes in the subject -> lower."}),
                "softness": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 255.0, "step": 1.0,
                             "tooltip": "0 = auto (same as tolerance). Width of the soft transition between removed and kept colors."}),
                "remove_shadows": ("BOOLEAN", {"default": True,
                                   "tooltip": "Removes shadows that are a darker version of the background color."}),
                "shadow_depth": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 0.9, "step": 0.05,
                                 "tooltip": "How dark a shadow may be and still be removed. Raise for darker shadows; lower if dark parts of the subject disappear."}),
                "choke": ("INT", {"default": 0, "min": -20, "max": 20, "step": 1,
                          "tooltip": "Positive shrinks the edge (Simple Choker), negative grows it. Keep 0 for logos with thin lines or small text, the edge color is already cleaned by decontaminate_edges."}),
                "feather": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 20.0, "step": 0.1,
                            "tooltip": "Edge smoothing (blur radius in pixels)."}),
                "despeckle": ("FLOAT", {"default": 0.0005, "min": 0.0, "max": 5.0, "step": 0.0005,
                              "tooltip": "Removes leftover islands smaller than this % of the image."}),
                "decontaminate_edges": ("BOOLEAN", {"default": True,
                                        "tooltip": "Removes the background color bleeding into the edge pixels."}),
                "inside": (["auto", "outside only", "inside too"], {"default": "auto",
                           "tooltip": "Background-colored areas INSIDE the logo. auto: removes real holes (inside of R, O, A), keeps thin/small white parts (highlights, gaps). outside only: removes only the background around the logo. inside too: removes every background-colored area."}),
                "transparency": (["auto", "solid", "glass"], {"default": "auto",
                                 "tooltip": "auto: a logo in ONE color tone (white logo on dark, black logo on white, cream glass ribbon) gets a perfect smooth matte where glows and glass stay see-through; a multi-color logo is cut out solid. solid / glass: force one of them."}),
                "hole_size": ("FLOAT", {"default": 2.5, "min": 0.1, "max": 20.0, "step": 0.1,
                              "tooltip": "auto only: how thick (% of image) an inside area must be to count as a hole and be removed. White parts inside get removed -> raise. Letter holes not removed -> lower."}),
            },
            "optional": {
                "ai_mask": ("MASK", {"tooltip": "Optional AI mask (white = keep), used for photos / busy backgrounds."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "MASK", "STRING")
    RETURN_NAMES = ("image", "mask", "inverted_mask", "info")
    FUNCTION = "run"
    CATEGORY = "My bg remove"

    def run(self, image, mode, tolerance, softness, remove_shadows, shadow_depth,
            choke, feather, despeckle, decontaminate_edges, inside="auto", hole_size=2.5, transparency="auto", ai_mask=None):
        outs, masks, infos = [], [], []
        for i in range(image.shape[0]):
            img = image[i, :, :, :3].cpu().numpy().astype(np.float32) * 255.0
            m = None
            if ai_mask is not None:
                mm = ai_mask[min(i, ai_mask.shape[0] - 1)]
                if tuple(mm.shape) != img.shape[:2]:
                    mm = torch.nn.functional.interpolate(mm[None, None].float(), size=img.shape[:2],
                                                         mode="bilinear", align_corners=False)[0, 0]
                m = mm.cpu().numpy()
            rgba, alpha, info = process(img, m, mode, tolerance, softness, remove_shadows,
                                        shadow_depth, choke, feather, despeckle, decontaminate_edges,
                                        inside, hole_size, transparency)
            outs.append(torch.from_numpy(rgba))
            masks.append(torch.from_numpy(alpha))
            infos.append(info)
        out = torch.stack(outs)
        mask = torch.stack(masks)
        print("[My bg remove]", " / ".join(infos))
        return (out, mask, 1.0 - mask, "\n".join(infos))


NODE_CLASS_MAPPINGS = {"MyBgRemove": MyBgRemove}
NODE_DISPLAY_NAME_MAPPINGS = {"MyBgRemove": "My bg remove"}

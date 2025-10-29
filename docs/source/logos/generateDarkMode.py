# from PIL import Image

# def recolor_gato_hep(
#     infile="gato-hep.png",
#     outfile="gato-hep-darkmode.png",
#     black_to=(255, 255, 255),        # pure white for outlines / text
#     darkgray_to=(100, 100, 100),     # light gray for tail fill / ear patch
#     dark_cutoff=10,                  # anything with all channels < 50 -> treat as black
#     gray_low=5,                     # anything between gray_low and gray_high -> treat as dark gray
#     gray_high=110
# ):
#     """
#     infile  : original transparent PNG (light-mode version)
#     outfile : new PNG for dark backgrounds

#     black_to     : replacement RGB for very dark/black strokes & text
#     darkgray_to  : replacement RGB for dark gray fills (tail, ear patch)
#     dark_cutoff  : max channel value to consider "black"
#     gray_low/high: channel range to consider "dark gray"
#     """

#     img = Image.open(infile).convert("RGBA")
#     pixels = img.getdata()

#     new_pixels = []
#     for (r, g, b, a) in pixels:
#         if a == 0:
#             # transparent pixel, keep as-is
#             new_pixels.append((r, g, b, a))
#             continue

#         # Get brightness as simple average
#         avg = (r + g + b) / 3.0

#         # Case 1: near-black strokes (outlines, text, whiskers, histogram frame)
#         if r < dark_cutoff and g < dark_cutoff and b < dark_cutoff:
#             new_pixels.append((*black_to, a))
#             continue

#         # Case 2: dark gray fills (cat ear patch, tail fill)
#         # we identify them by being dark-ish but not pitch black
#         if gray_low <= avg <= gray_high:
#             new_pixels.append((*darkgray_to, a))
#             continue

#         # Otherwise (orange bars, blue bars, etc.) keep original
#         new_pixels.append((r, g, b, a))

#     img.putdata(new_pixels)
#     img.save(outfile)
#     print(f"Saved dark-mode logo to {outfile}")


# if __name__ == "__main__":
#     recolor_gato_hep()


from PIL import Image
import math

def dist2(c1, c2):
    """squared Euclidean distance between two RGB tuples"""
    return ((c1[0]-c2[0])**2 +
            (c1[1]-c2[1])**2 +
            (c1[2]-c2[2])**2)

def recolor_gato_hep_smart(
    infile="gato-hep.png",
    outfile="gato-hep-darkmode.png",
    outline_to=(255, 255, 255),      # new color for black outline/text
    fillgray_to=(180, 180, 180),     # new color for dark gray fills
    black_ref=(0, 0, 0),
    gray_ref=(60, 60, 60),           # sample from the tail/ear patch in the original
    max_alpha_for_processing=10,     # treat almost-transparent as transparent
    despeckle=True
):
    """
    Strategy:
    - Compare each opaque pixel against two reference dark colors:
        black_ref ~ pure stroke/text
        gray_ref  ~ tail/ear patch dark gray fill
      Whichever it's closer to decides mapping.
    - Everything else (blue/orange bars etc.) is kept.
    - Optional despeckle pass to kill isolated dark dots.

    Tweak gray_ref to match your actual gray patch in the source.
    """

    # Load original
    img = Image.open(infile).convert("RGBA")
    w, h = img.size
    px = img.load()

    # First pass: recolor
    out = Image.new("RGBA", (w, h))
    out_px = out.load()

    for y in range(h):
        for x in range(w):
            r, g, b, a = px[x, y]

            # keep transparent (or nearly transparent) as-is
            if a <= max_alpha_for_processing:
                out_px[x, y] = (r, g, b, 0)
                continue

            rgb = (r, g, b)

            # distances to reference darks
            d_black = dist2(rgb, black_ref)
            d_gray  = dist2(rgb, gray_ref)

            # how "dark" overall
            avg = (r + g + b) / 3.0

            # Heuristic:
            # If pixel is really dark overall (avg < ~80), it's part of stroke or tail patch.
            if avg < 80:
                # Decide if it's outline-ish or gray-fill-ish:
                if d_black * 1.0 < d_gray * 0.8:
                    # closer to pure black than to tail gray
                    out_px[x, y] = (*outline_to, a)
                else:
                    # closer to tail gray
                    out_px[x, y] = (*fillgray_to, a)
            else:
                # not dark (blue/orange bars etc.): keep
                out_px[x, y] = (r, g, b, a)

    # Optional 2nd pass: despeckle tiny leftover dark dots
    if despeckle:
        # We'll treat "speck" = pixel that's still darker than midgray
        # but is surrounded by much brighter neighbors. We'll lift it.
        def is_bright(rgb):
            return (rgb[0] + rgb[1] + rgb[2]) / 3.0 > 200

        def is_dark(rgb):
            return (rgb[0] + rgb[1] + rgb[2]) / 3.0 < 140

        out2 = out.copy()
        out2_px = out2.load()

        for y in range(1, h-1):
            for x in range(1, w-1):
                r, g, b, a = out_px[x, y]
                if a == 0:
                    continue
                if not is_dark((r, g, b)):
                    continue

                # check 4-neighborhood
                nbrs = [
                    out_px[x-1, y],
                    out_px[x+1, y],
                    out_px[x, y-1],
                    out_px[x, y+1],
                ]
                if all(n[3] > 0 and is_bright(n[:3]) for n in nbrs):
                    # lift this pixel toward outline_to (white)
                    out2_px[x, y] = (*outline_to, a)

        out = out2

    out.save(outfile, "PNG")
    print(f"Saved dark-mode logo to {outfile}")


if __name__ == "__main__":
    recolor_gato_hep_smart()
"""The Closed Tabs visual preset for generated frames.

The channel is about things found on the internet that should not quite exist,
so the look is the room the viewer is sitting in at 2 a.m.: dark interiors lit
by a single screen, old hardware, empty corridors, long exposures of cities at
night. Photographic rather than illustrated, so it sits beside archive material
without a seam.

Screens are the hard part. FLUX draws text on a monitor as a smear of fake
letters, and the visual check rejects any frame with lettering in it. So screens
are always shown as glow - light on a wall, a blank lit panel, a CRT seen from
the side - and never as something to read.

Everything here is appended by the pipeline; a scene prompt describes only what
is visible in that shot.
"""
import re

# What every generated frame gets. Lit night, not a black frame: the first hook
# measured a mean brightness of 16 out of 255 and read as a dark rectangle.
BASE = (
    "photorealistic photograph, 35mm lens, full frame edge to edge, well exposed night "
    "scene, colourful night palette, teal and magenta light, warm amber lamp light, the "
    "background melted into a soft smooth bokeh of coloured light, shallow depth of field, "
    "lifted shadows with full detail, rich colour, "
    "gentle haze, fine film grain, deep focus, high detail"
)

# What it must never contain.
NEGATIVE = (
    "no text, no letters, no numbers, no readable screen content, no user interface, "
    "no captions, no watermark, no logo, no signature, no borders, no collage, "
    "no split screen, no cartoon, no anime, no illustration, no painting, "
    "no 3d render look, no pitch black frame, no gore, no blood, no weapons, no corpses, "
    "no lens flare, no fisheye, no tilted horizon, "
    "no extra limbs, no deformed faces, "
    "no letterbox, no black bars, no cinematic bars, image fills the whole frame"
)

# A person may be in shot for scale and mood, never as someone identifiable:
# nobody real is depicted, and a face invites the uncanny-valley failure that
# marks a frame as generated.
PEOPLE = (
    "one person seen from behind or in silhouette, face not visible, "
    "small in frame"
)

# Screens appear constantly on this channel; this keeps them unreadable.
SCREEN = "the screen shows only soft featureless light, nothing legible"
_SCREEN_WORDS = re.compile(
    r"\b(screen|monitor|display|computer|laptop|phone|crt|tv|television)s?\b", re.I)

# Paper is allowed - old books, letters, maps, flyers on a pole - but whatever is
# written on it must not be readable, because FLUX writes nonsense.
PAPER = ("any writing on the paper is tiny, faded and out of focus, impossible to read, "
         "shallow depth of field")
_PAPER_WORDS = re.compile(
    r"\b(book|books|letter|letters|envelope|map|maps|paper|papers|page|pages|poster|posters|"
    r"flyer|flyers|notes?|document|documents|photograph|photographs|typewriter)\b", re.I)

ASPECT = {"width": 1344, "height": 768}      # 16:9-ish and a multiple of 64; filled to 1920x1080 by a slight crop
STEPS = 4                                     # schnell is distilled to 4 steps
GUIDANCE = 0.0                                # and ignores CFG entirely


def scene_prompt(description, with_people=False):
    """Full prompt for one scene from a short visual description."""
    parts = [description.strip().rstrip(".")]
    if with_people:
        parts.append(PEOPLE)
    if _SCREEN_WORDS.search(description):
        parts.append(SCREEN)
    if _PAPER_WORDS.search(description):
        parts.append(PAPER)
    parts.append(BASE)
    parts.append(NEGATIVE)
    return ", ".join(parts)


# Ten probes chosen where this preset is most likely to break: screens (the
# lettering failure), hardware close-ups, human scale, night exteriors, and the
# purely atmospheric shots no archive can supply.
STYLE_TEST_SCENES = [
    ("a dark bedroom at night, an old CRT monitor on a desk casting cyan light "
     "across the wall", True),
    ("a row of beige 1990s computers in an empty office after hours", False),
    ("a server room corridor, rows of blinking indicator lights receding into "
     "darkness", False),
    ("a payphone on an empty street corner at night under a sodium streetlight", False),
    ("an old television set in an empty living room showing static glow", False),
    ("a shortwave radio receiver on a wooden table, dial lit amber, dark room", False),
    ("a city skyline at night seen through a rain-streaked window", True),
    ("a dusty attic with a cardboard box of old game cartridges in a shaft of light", False),
    ("an empty fluorescent-lit office corridor with yellowed walls and damp carpet", False),
    ("a laptop open on a kitchen table at 3 a.m., the only light in the house", True),
]

# Thumbnail backgrounds: the type block sits on the LEFT, so the subject owns
# the RIGHT of the frame.
THUMBNAIL_BACKGROUNDS = [
    ("an old CRT monitor glowing in a pitch-dark room on the right of frame, "
     "harsh cyan light, dust in the air", False),
    ("an endless empty yellow office corridor vanishing to the right, flickering "
     "fluorescent light", False),
    ("a lone payphone ringing under a streetlight on the right of frame, deep "
     "night, wet asphalt", False),
]

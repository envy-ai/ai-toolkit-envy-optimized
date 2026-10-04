"""Versioned RGB presentation for specialized Krea edit conditioning only."""
from PIL import Image, ImageOps


KREA_EDIT_CONTROL_PRESENTATION = 'krea-edit-rgb-v2'


def load_control_rgb(path, background=(0, 0, 0)):
    """Read first frame, honor EXIF, and composite every supported alpha mode.

    Conversion deliberately matches PIL RGB presentation, including 16-bit
    grayscale. Preview uploads use this same representation, avoiding Comfy's
    alternative grayscale scaling or animated-image batch decoding.
    """
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source)
        if image.mode in ('RGBA', 'LA') or 'transparency' in image.info:
            image = image.convert('RGBA')
            result = Image.new('RGB', image.size, tuple(background))
            result.paste(image, mask=image.getchannel('A'))
            return result
        return image.convert('RGB')

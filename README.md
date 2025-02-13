# DoodlePix
Diffusion based Drawing Assistant

TODOs
Folder Structure:
colors/
tags/
toBeEdited(someway)/

_bis for Hue and Contrast and Vibrant changes
_tris for rotatation and scale crop changes


- [x] change hue of images (white bg and call them bis to correctly learn color changes of same subject)
- [ ] resize images an save them as tris
- [ ] Halo? Tag?
- [ ] f = fidelity, (0-10) HED f = 9, Canny f = 10, Scribble f = 3, Other Scribble f = 7
- [ ] p = shading, (flat, painted, 3D)
- [ ] remove <> from prompts

f=5, p=flat, bench, #f9c473, #cb6240, #fdfcf8, #ffffff background, <tags: gold, shield, diamond, currency, emblem>

- [ ] Dataset to train general use IpAdapter: controlnet canny + style with Flux Redux at low intensity



example:
- [ ] f=5, k=3, s=Earth, p=flat, bench, blue metal structure, silver metal bolts, beige wood bench, white background


LEGACY:
- [ ] k = complexity, (0-10) (in DoodlePixV4 there are some images with K indexes correct, newly ones dont)
- [ ] s = style, (Earth, Future, Fantasy, Whimsy) (TRY to train)
- [ ] prompt = must list subject of the image, the parts that compose the image and their relative colors, materials 





tags= halo, black outline, normal style, 3d shading, Earth style, Future style, Fantasy style, Whimsy style

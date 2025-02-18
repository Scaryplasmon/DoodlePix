# DoodlePix
Diffusion based Drawing Assistant

This repo is a custom implementation of the InstructPix2Pix model, a text-to-image generation model that uses a text encoder to generate images from text prompts and input images.
in this case the input image is a doodle and the text prompt has this shape example:

f=5, [flat], <tags: gold, shield, diamond, currency, emblem>,   #f9c473, #cb6240, #fdfcf8, #ffffff background.

TODOs

_bis for Hue and Contrast and Vibrant changes
_tris for rotatation and scale crop changes

Data

- [x] change hue of images (white bg and call them bis to correctly learn color changes of same subject)
- [x] HEX Colors NEW (dalleNew - A_outline - A_flat - edited_image - A_3D - A_painted )
- [x] Canny with various fidelity values
- [x] sort Canny after renaming (add <f=x> in front of the name. when copying the images we remove the f=x and we save it as individual txt file in another folder)
- [X] Add Doodles and change fidelity
- [X] Join Prompts
- [X] add [Shading] to Prompts
- [ ] Check all _bis

- [ ] check all normal folders for painted items, delete them all using the DataManager app
- [ ] separate normal and painted [Shading]
- [ ] resize images an save them as tris (use the ImageProcessor App)

- [ ] Halo? High low Contrast, Vibrant, Tags?
- [ ] Token Analysis
- [ ] Plot Colors in data, visualize most common tags, 
- [ ] f = fidelity, (0-9) HED f = 9, Canny f = 10, Scribble f = 3, Other Scribble f = 7
- [ ] p = shading, (flat, painted, 3D, outline)
- [ ] remove <tags:> from prompts ?

f=5, p=flat, bench, #f9c473, #cb6240, #fdfcf8, #ffffff background, <tags: gold, shield, diamond, currency, emblem>

- [ ] Dataset to train general use IpAdapter: controlnet canny + style with Flux Redux at low intensity

--REPORTS

Colors dont follow the prompt, they are random, and change a ton from seed to seed.
--Training with txtEncoder requiring grad kept giving errors
--tried training in BF16 but noticed quite some worsening in the results
--PreTrained txtEncoder in FP32 then load it into unet FP16 training.

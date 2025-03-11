# DoodlePix
Diffusion based Drawing Assistant


https://github.com/user-attachments/assets/b6a44dc0-6d01-4285-a5ad-9f6fedf91656


This is a custom implementation of the InstructPix2Pix pipeline.

<details>
  <summary><strong>Click to view reasoning behind Development</strong></summary>
  Objective is to analyze the subtle difference between canny edges and hand-made drawings.

  
  While the first just extracts precisely the most prominent lines in an image,
  drawings are made with intention, a few squiggly lines placed in the right place can deliver a much better idea of what's being represented in the image:
  
  <table>
    <tr>
      <td style="text-align: center;">
        <strong>Drawing</strong><br>
        <img src="assets/alien/alienDrawing.png" alt="Drawing" width="240" height="240">
      </td>
      <td style="text-align: center;">
        <strong>Canny</strong><br>
        <img src="assets/alien/alienCanny.png" alt="Canny" width="240" height="240">
      </td>
    </tr>
  </table>
  
  To address this I train a Fidelity embedding to inject an explicit fidelity signal to the Unet that learns how to modulate its denoising behaviour accordingly.
  
  The FidelityMLP ranges from 0 to 9, (f0-f9), allowing the user to choose how much the model should "correct" their drawing.
  
  The InstructPix2Pix pipeline already supports an ImageGuidance factor, that can be passed during inference to control how much the model should follow the Image Input; 
  
  but this results only in higher values following TOO Much the drawing input, 
  while lower values completely lose composition and its nuisances.
    
</details>

## Fidelity embedding in action

-Fidelity values ranging from 0 to 9 while keeping prompt, seed and steps constant.
<table>
  <tr>
    <td colspan="5" style="text-align: center; font-weight: bold; padding-bottom: 8px;">
      Prompt: f*, axe, metal, wooden handle. grey, brown wood
    </td>
  </tr>
  <tr>
    <td style="text-align: center;">
      <strong>Image</strong><br>
      <img src="assets/axe.png" alt="Image" width="124" height="124">
    </td>
    <td style="text-align: center;">
      <strong>Normal</strong><br>
      <img src="assets/AxeNormal.gif" alt="Normal" width="124" height="124">
    </td>
    <td style="text-align: center;">
      <strong>3D</strong><br>
      <img src="assets/Axe3D.gif" alt="3D" width="124" height="124">
    </td>
    <td style="text-align: center;">
      <strong>Outline</strong><br>
      <img src="assets/AxeOutline.gif" alt="Outline" width="124" height="124">
    </td>
    <td style="text-align: center;">
      <strong>Flat</strong><br>
      <img src="assets/AxeFlat.gif" alt="Flat" width="124" height="124">
    </td>
  </tr>
</table>
<table>
  <tr>
    <td colspan="5" style="text-align: center; font-weight: bold; padding-bottom: 8px;">
      Prompt: f*, red heart, white background.
    </td>
  </tr>
  <tr>
    <td style="text-align: center;">
      <strong>Image</strong><br>
      <img src="assets/heart.png" alt="Image" width="124" height="124">
    </td>
    <td style="text-align: center;">
      <strong>Normal</strong><br>
      <img src="assets/Heart.gif" alt="Normal" width="124" height="124">
    </td>
    <td style="text-align: center;">
      <strong>3D</strong><br>
      <img src="assets/Heart3D.gif" alt="3D" width="124" height="124">
    </td>
    <td style="text-align: center;">
      <strong>Outline</strong><br>
      <img src="assets/HeartOutline.gif" alt="Outline" width="124" height="124">
    </td>
    <td style="text-align: center;">
      <strong>Flat</strong><br>
      <img src="assets/HeartFlat.gif" alt="Flat" width="124" height="124">
    </td>
  </tr>
</table>

-DooodlePix also fully supports canny or any type of line art inputs, while still keeping creativity freedom thanks to the fidelity embedding

<table>
  <tr>
    <td colspan="5" style="text-align: center; font-weight: bold; padding-bottom: 8px;">
      Prompt: f*, woman, portrait, frame. black hair, pink, black background.
    </td>
  </tr>
  <tr>
    <td style="text-align: center;">
      <strong>Image</strong><br>
      <img src="assets/woman.png" alt="Image" width="124" height="124">
    </td>
    <td style="text-align: center;">
      <strong>Normal</strong><br>
      <img src="assets/WomanNormal.gif" alt="Normal" width="124" height="124">
    </td>
    <td style="text-align: center;">
      <strong>3D</strong><br>
      <img src="assets/Woman3D.gif" alt="3D" width="124" height="124">
    </td>
    <td style="text-align: center;">
      <strong>Outline</strong><br>
      <img src="assets/WomanOutline.gif" alt="Outline" width="124" height="124">
    </td>
    <td style="text-align: center;">
      <strong>Flat</strong><br>
      <img src="assets/WomanFlat.gif" alt="Flat" width="124" height="124">
    </td>
  </tr>
</table>

More examples:


<table style="width: 100%; table-layout: fixed;">
  <tr>
    <td colspan="2" style="text-align: center; font-weight: bold; font-size: 0.9rem; padding-bottom: 8px;">
      Prompt: f*, potion, bottle, cork. blue, brown, black background.
    </td>
    <td colspan="2" style="text-align: center; font-weight: bold; font-size: 0.9rem; padding-bottom: 8px;">
      Prompt: f*, maul, hammer. gray, brown, white background.
    </td>
    <td colspan="2" style="text-align: center; font-weight: bold; font-size: 0.9rem; padding-bottom: 8px;">
      Prompt: f*, torch, flame. red, brown, black background.
    </td>
  </tr>
  <tr>
    <td style="text-align: center;">
      <img src="assets/potion.png" alt="Potion Image" style="width: 100%; max-width: 150px; height: auto; object-fit: contain;">
    </td>
    <td style="text-align: center;">
      <img src="assets/PotionSingle.gif" alt="Potion Normal" style="width: 100%; max-width: 150px; height: auto; object-fit: contain;">
    </td>
    <td style="text-align: center;">
      <img src="assets/maul.png" alt="Maul Image" style="width: 100%; max-width: 150px; height: auto; object-fit: contain;">
    </td>
    <td style="text-align: center;">
      <img src="assets/maulNormal.gif" alt="Maul Normal" style="width: 100%; max-width: 150px; height: auto; object-fit: contain;">
    </td>
    <td style="text-align: center;">
      <img src="assets/torch.png" alt="Torch Image" style="width: 100%; max-width: 150px; height: auto; object-fit: contain;">
    </td>
    <td style="text-align: center;">
      <img src="assets/TorchSingle.gif" alt="Torch Normal" style="width: 100%; max-width: 150px; height: auto; object-fit: contain;">
    </td>
  </tr>
</table>


DoodlePix shows great color understanding, byproduct of the InstructPix2Pix architecture.

<table>
  <tr>
    <td colspan="8" style="text-align: center; font-weight: bold; padding-bottom: 8px;">
      Prompt: f9, flower, stylized. *color, green, white
    </td>
  </tr>
  <tr>
    <td style="text-align: center;">
      <strong></strong><br>
      <img src="assets/flowerInput.png" alt="Image" width="100" height="100">
    </td>
    <td style="text-align: center;">
      <strong>red</strong><br>
      <img src="assets/flower2.png" alt="Normal" width="100" height="100">
    </td>
    <td style="text-align: center;">
      <strong>light blue</strong><br>
      <img src="assets/flower3.png" alt="3D" width="100" height="100">
    </td>
    <td style="text-align: center;">
      <strong>purple</strong><br>
      <img src="assets/flower4.png" alt="Outline" width="100" height="100">
    </td>
  </tr>
  <tr>
    <td style="text-align: center;">
      <strong>green</strong><br>
      <img src="assets/flower1.png" alt="Flat" width="100" height="100">
    </td>
    <td style="text-align: center;">
      <strong>cyan</strong><br>
      <img src="assets/flower6.png" alt="Flat" width="100" height="100">
    </td>
    <td style="text-align: center;">
      <strong>light green</strong><br>
      <img src="assets/flower7.png" alt="Flat" width="100" height="100">
    </td>
    <td style="text-align: center;">
      <strong>orange</strong><br>
      <img src="assets/flower8.png" alt="Flat" width="100" height="100">
    </td>
  </tr>
</table>


DoodlePix can achieve acceptable results within 4 steps

<table>
  <tr>
    <td colspan="8" style="text-align: center; font-weight: bold; padding-bottom: 8px;">
      Prompt: f4, alien, red skin, white shirt, white background.
    </td>
  </tr>
  <tr>
    <td style="text-align: center;">
      <strong>Drawing</strong><br>
      <img src="assets/alien/alienDrawing.png" alt="Image" width="100" height="100">
    </td>
    <td style="text-align: center;">
      <strong>4 steps</strong><br>
      <img src="assets/alien/AlienD_4steps.png" alt="Image" width="100" height="100">
    </td>
    <td style="text-align: center;">
      <strong>8 steps</strong><br>
      <img src="assets/alien/AlienD_8steps.png" alt="Image" width="100" height="100">
    </td>
    <td style="text-align: center;">
      <strong>14 steps</strong><br>
      <img src="assets/alien/AlienD_14steps.png" alt="Image" width="100" height="100">
    </td>
    <td style="text-align: center;">
      <strong>20 steps</strong><br>
      <img src="assets/alien/AlienD_20steps.png" alt="Image" width="100" height="100">
    </td>
    <td style="text-align: center;">
      <strong>32 steps</strong><br>
      <img src="assets/alien/AlienD_32steps.png" alt="Image" width="100" height="100">
    </td>
    <td style="text-align: center;">
      <strong>44 steps</strong><br>
      <img src="assets/alien/AlienD_44steps.png" alt="Image" width="100" height="100">
    </td>
    <td style="text-align: center;">
      <strong>60 steps</strong><br>
      <img src="assets/alien/AlienD_60steps.png" alt="Image" width="100" height="100">
    </td>
    
  </tr>
</table>

<table>
  <tr>
    <td colspan="8" style="text-align: center; font-weight: bold; padding-bottom: 8px;">
      Prompt: f4, alien, red skin, white shirt, white background.
    </td>
  </tr>
  <tr>
    <td style="text-align: center;">
      <strong>Canny</strong><br>
      <img src="assets/alien/alienCanny.png" alt="Image" width="100" height="100">
    </td>
    <td style="text-align: center;">
      <strong>4 steps</strong><br>
      <img src="assets/alien/AlienC_4steps.png" alt="Image" width="100" height="100">
    </td>
    <td style="text-align: center;">
      <strong>8 steps</strong><br>
      <img src="assets/alien/AlienC_8steps.png" alt="Image" width="100" height="100">
    </td>
    <td style="text-align: center;">
      <strong>14 steps</strong><br>
      <img src="assets/alien/AlienC_14steps.png" alt="Image" width="100" height="100">
    </td>
    <td style="text-align: center;">
      <strong>20 steps</strong><br>
      <img src="assets/alien/AlienC_20steps.png" alt="Image" width="100" height="100">
    </td>
    <td style="text-align: center;">
      <strong>32 steps</strong><br>
      <img src="assets/alien/AlienC_32steps.png" alt="Image" width="100" height="100">
    </td>
    <td style="text-align: center;">
      <strong>44 steps</strong><br>
      <img src="assets/alien/AlienC_44steps.png" alt="Image" width="100" height="100">
    </td>
    <td style="text-align: center;">
      <strong>60 steps</strong><br>
      <img src="assets/alien/AlienC_60steps.png" alt="Image" width="100" height="100">
    </td>
    
  </tr>
</table>






ALMMOST THERE WITH THE DATA

copy paste input edges (1) inside the Edited image folder, 
compare them briefly, use Data manager to remove or improve input edges

NEED more low values fidelity, more quick bad drawings

add black f0 values images( might be a cool idea)


Training results:
- FidelityMLP still to be implemented in the actual pipeline (was giving issues like size 78 exceeds token limits of ClipTextEncoder which is 77)
- Training with txtEncoder requiring grad kept giving errors so we split the training into two parts:
    - PreTrained txtEncoder in FP32 then load it into unet FP16 training.
- Tried training in BF16 but noticed quite some worsening in the results

Feels data needs more fidelity variation. (seemed like the image guidance was sort of working ranging from 0.5 to 5.0)- maybe FidelityMLP is not needed?

training seems to give good results in alrady 2k steps, but its a fluke, add real VALIDATION drawings, not just canny images.

learning rate should be 5e-5 dont change it.

DELETE SHITTY IMAGES, or too repetive ones.
Check "image_XXXXX, there´s a lot of repetitive ones that could be deleted or at least prompts should be matching.



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

### DoodlePix  
*Diffusion based Drawing Assistant*

[![DoodlePix](https://github.com/user-attachments/assets/b6a44dc0-6d01-4285-a5ad-9f6fedf91656)](https://github.com/user-attachments/assets/b6a44dc0-6d01-4285-a5ad-9f6fedf91656)

-------

<details>
  <summary><strong>The PIPE</strong></summary>
  
  - **Base Model:** StableDiffusion 2.1  
  - **Inference:** fits in < 4GB  
  - **Speed:** ~15 steps/second  
  - **Training Requirements:** < 14GB
  - **Pipeline:** InstructPix2Pix (+ custom fidelity input)
</details>

-------

<details>
  <summary><strong>The DATA</strong></summary>

  - **Data Size:** ~4.5k images
  - **Image Generation:** Dalle-3, FLUX-PRO 1.1 and Flux-Redux-DEV 
  - **Edge Extraction:** Canny, Fake Scribble, Scribble Xdog, HED soft edge 
  - **Doodles** were hand-drawn and compose about 10% of the edges

    To have maximum control over the Dataset, a few  apps were built 
</details>


## Fidelity Embedding in Action

*Fidelity values range from 0 to 9 while keeping prompt, seed, and steps constant.*

<table style="width:100%; table-layout: fixed;">
  <tr>
    <td colspan="5" style="text-align:center; font-weight:bold; font-size:0.9rem; padding-bottom:8px;">
      Prompt: f*, red heart, white background.
    </td>
  </tr>
  <tr>
    <td style="text-align:center;">
      <strong>Image</strong><br>
      <img src="assets/heart.png" alt="Heart Image" style="width:150px; height:150px; object-fit:contain;">
    </td>
    <td style="text-align:center;">
      <strong>Normal</strong><br>
      <img src="assets/Heart.gif" alt="Heart Normal" style="width:150px; height:150px; object-fit:contain;">
    </td>
    <td style="text-align:center;">
      <strong>3D</strong><br>
      <img src="assets/Heart3D.gif" alt="Heart 3D" style="width:150px; height:150px; object-fit:contain;">
    </td>
    <td style="text-align:center;">
      <strong>Outline</strong><br>
      <img src="assets/HeartOutline.gif" alt="Heart Outline" style="width:150px; height:150px; object-fit:contain;">
    </td>
    <td style="text-align:center;">
      <strong>Flat</strong><br>
      <img src="assets/HeartFlat.gif" alt="Heart Flat" style="width:150px; height:150px; object-fit:contain;">
    </td>
  </tr>
</table>

-The model also accepts canny edges as input, while keeping fidelity injection relevant
<table style="width:100%; table-layout: fixed;">
  <tr>
    <td colspan="5" style="text-align:center; font-weight:bold; font-size:0.9rem; padding-bottom:8px;">
      Prompt: f*, woman, portrait, frame. black hair, pink, black background.
    </td>
  </tr>
  <tr>
    <td style="text-align:center;">
      <strong>Image</strong><br>
      <img src="assets/woman.png" alt="Woman Image" style="width:150px; height:150px; object-fit:contain;">
    </td>
    <td style="text-align:center;">
      <strong>Normal</strong><br>
      <img src="assets/WomanNormal.gif" alt="Woman Normal" style="width:150px; height:150px; object-fit:contain;">
    </td>
    <td style="text-align:center;">
      <strong>3D</strong><br>
      <img src="assets/Woman3D.gif" alt="Woman 3D" style="width:150px; height:150px; object-fit:contain;">
    </td>
    <td style="text-align:center;">
      <strong>Outline</strong><br>
      <img src="assets/WomanOutline.gif" alt="Woman Outline" style="width:150px; height:150px; object-fit:contain;">
    </td>
    <td style="text-align:center;">
      <strong>Flat</strong><br>
      <img src="assets/WomanFlat.gif" alt="Woman Flat" style="width:150px; height:150px; object-fit:contain;">
    </td>
  </tr>
</table>

More Examples

<table style="width:100%; table-layout: fixed;">
  <tr>
    <td colspan="2" style="text-align:center; font-weight:bold; font-size:0.9rem; padding-bottom:8px;">
      Prompt: f*, potion, bottle, cork. blue, brown, black background.
    </td>
    <td colspan="2" style="text-align:center; font-weight:bold; font-size:0.9rem; padding-bottom:8px;">
      Prompt: f*, maul, hammer. gray, brown, white background.
    </td>
    <td colspan="2" style="text-align:center; font-weight:bold; font-size:0.9rem; padding-bottom:8px;">
      Prompt: f*, torch, flame. red, brown, black background.
    </td>
  </tr>
  <tr>
    <td style="text-align:center;">
      <img src="assets/potion.png" alt="Potion Image" style="width:100%; max-width:150px; height:auto; object-fit:contain;">
    </td>
    <td style="text-align:center;">
      <img src="assets/PotionSingle.gif" alt="Potion Normal" style="width:100%; max-width:150px; height:auto; object-fit:contain;">
    </td>
    <td style="text-align:center;">
      <img src="assets/maul.png" alt="Maul Image" style="width:100%; max-width:150px; height:auto; object-fit:contain;">
    </td>
    <td style="text-align:center;">
      <img src="assets/maulNormal.gif" alt="Maul Normal" style="width:100%; max-width:150px; height:auto; object-fit:contain;">
    </td>
    <td style="text-align:center;">
      <img src="assets/torch.png" alt="Torch Image" style="width:100%; max-width:150px; height:auto; object-fit:contain;">
    </td>
    <td style="text-align:center;">
      <img src="assets/TorchSingle.gif" alt="Torch Normal" style="width:100%; max-width:150px; height:auto; object-fit:contain;">
    </td>
  </tr>
</table>
<table style="width:100%; height: 164px; table-layout: fixed;">
  <tr>
    <td colspan="5" style="text-align:center; font-weight:bold; font-size:0.9rem; padding-bottom:8px;">
      Prompt: f*, axe, metal, wooden handle. grey, brown wood
    </td>
  </tr>
  <tr>
    <td style="text-align:center;">
      <strong>Image</strong><br>
      <img src="assets/axe.png" alt="Axe Image" style="width:150px; height:150px; object-fit:contain;">
    </td>
    <td style="text-align:center;">
      <strong>Normal</strong><br>
      <img src="assets/AxeNormal.gif" alt="Axe Normal" style="width:150px; height:150px; object-fit:contain;">
    </td>
    <td style="text-align:center;">
      <strong>3D</strong><br>
      <img src="assets/Axe3D.gif" alt="Axe 3D" style="width:150px; height:150px; object-fit:contain;">
    </td>
    <td style="text-align:center;">
      <strong>Outline</strong><br>
      <img src="assets/AxeOutline.gif" alt="Axe Outline" style="width:150px; height:150px; object-fit:contain;">
    </td>
    <td style="text-align:center;">
      <strong>Flat</strong><br>
      <img src="assets/AxeFlat.gif" alt="Axe Flat" style="width:150px; height:150px; object-fit:contain;">
    </td>
  </tr>
</table>

# LORAs

Lora training is an efficient way to fine-tune DoodlePix for specific styles and ways of drawing.

<table style="width:100%; height: 124px; table-layout: fixed;">
  <tr>
    <td colspan="3" style="text-align:center; font-weight:italic; font-size:0.9rem; padding-bottom:0px;">
    </td>
  </tr>
  <tr>
    <td style="text-align:center;">
      input<br>
      <img src="assets/Googh/sunflower_DR.png" alt="Input" style="width:200px; height:200px; object-fit:contain;">
    </td>
    <td style="text-align:center;">
      Googh<br>
      <img src="assets/Googh/sunflower_0.png" alt="Googh" style="width:200px; height:200px; object-fit:contain;">
    </td>
    <td style="text-align:center;">
      DontStarve<br>
      <img src="assets/DontStarve/SunFlowers_4.png" alt="DontStarve" style="width:200px; height:200px; object-fit:contain;">
    </td>
  </tr>
  <tr>
    <td style="text-align:center;">
      input<br>
      <img src="assets/Googh/gift_DR.png" alt="Input" style="width:200px; height:200px; object-fit:contain;">
    </td>
    <td style="text-align:center;">
      Googh<br>
      <img src="assets/Googh/gift_3.png" alt="Googh" style="width:200px; height:200px; object-fit:contain;">
    </td>
    <td style="text-align:center;">
      DontStarve<br>
      <img src="assets/DontStarve/gift_20.png" alt="DontStarve" style="width:200px; height:200px; object-fit:contain;">
    </td>
  </tr>
</table>

-----

## Googh

Loras retains Styles and Fidelity injection from DoodlePix 

<table style="width:100%; height: 124px; table-layout: fixed;">
  <tr>
    <td colspan="5" style="text-align:center; font-weight:italic; font-size:0.9rem; padding-bottom:0px;">
    </td>
  </tr>
  <tr>
    <td style="text-align:center;">
      input<br>
      <img src="assets/Googh/man_DR2.png" alt="Input" style="width:150px; height:150px; object-fit:contain;">
    </td>
    <td style="text-align:center;">
      Normal<br>
      <img src="assets/Googh/manNormal.png" alt="Normal" style="width:150px; height:150px; object-fit:contain;">
    </td>
    <td style="text-align:center;">
      3D<br>
      <img src="assets/Googh/man3D.png" alt="3D" style="width:150px; height:150px; object-fit:contain;">
    </td>
    <td style="text-align:center;">
      Outline<br>
      <img src="assets/Googh/manOutline.png" alt="Outline" style="width:150px; height:150px; object-fit:contain;">
    </td>
    <td style="text-align:center;">
      Flat<br>
      <img src="assets/Googh/manFlat.png" alt="Flat" style="width:150px; height:150px; object-fit:contain;">
    </td>
    </tr>
    <tr>
    <td style="text-align:center;">
      Low Fidelity<br>
      <img src="assets/Googh/man_3.png" alt="Flat" style="width:150px; height:150px; object-fit:contain;">
    </td>
    <td style="text-align:center;">
      High Fidelity<br>
      <img src="assets/Googh/manFidelity7.png" alt="Flat" style="width:150px; height:150px; object-fit:contain;">
    </td>
  </tr>
</table>
<table style="width:100%; height: 124px; table-layout: fixed;">
  <tr>
    <td colspan="5" style="text-align:center; font-weight:italic; font-size:0.9rem; padding-bottom:0px;">
    </td>
  </tr>
  <tr>
    <td style="text-align:center;">
      input<br>
      <img src="assets/Googh/gift_DR.png" alt="Input" style="width:150px; height:150px; object-fit:contain;">
    </td>
    <td style="text-align:center;">
      Normal<br>
      <img src="assets/Googh/giftNormal.png" alt="Normal" style="width:150px; height:150px; object-fit:contain;">
    </td>
    <td style="text-align:center;">
      3D<br>
      <img src="assets/Googh/gift3D.png" alt="3D" style="width:150px; height:150px; object-fit:contain;">
    </td>
    <td style="text-align:center;">
      Outline<br>
      <img src="assets/Googh/giftOutline.png" alt="Outline" style="width:150px; height:150px; object-fit:contain;">
    </td>
    <td style="text-align:center;">
      Flat<br>
      <img src="assets/Googh/giftFlat.png" alt="Flat" style="width:150px; height:150px; object-fit:contain;">
    </td>
  </tr>
</table>

More Examples:

<table style="width:100%; height: 140px; table-layout: fixed;">
  <tr>
    <td colspan="3" style="text-align:center; font-weight:italic; font-size:0.9rem; padding-bottom:0px;">
    </td>
  </tr>
  <tr>
    <td style="text-align:center;">
      <img src="assets/Googh/road_DR.png" alt="Input" style="width:200px; height:200px; object-fit:contain;">
    </td>
    <td style="text-align:center;">
      <img src="assets/Googh/road4.png" alt="Normal" style="width:200px; height:200px; object-fit:contain;">
    </td>
    <td style="text-align:center;">
      <img src="assets/Googh/road5.png" alt="3D" style="width:200px; height:200px; object-fit:contain;">
    </td>
  </tr>
  <tr>
    <td style="text-align:center;">
      <img src="assets/Googh/road7.png" alt="Outline" style="width:200px; height:200px; object-fit:contain;">
    </td>
    <td style="text-align:center;">
      <img src="assets/Googh/road6.png" alt="Flat" style="width:200px; height:200px; object-fit:contain;">
    </td>
    <td style="text-align:center;">
      <img src="assets/Googh/road3.png" alt="Flat" style="width:200px; height:200px; object-fit:contain;">
    </td>
  </tr>
</table>

<table style="width:100%; height: 140px; table-layout: fixed;">
  <tr>
    <td colspan="6" style="text-align:center; font-weight:italic; font-size:0.9rem; padding-bottom:0px;">
    </td>
  </tr>
  <tr>
    <td style="text-align:center;">
      <img src="assets/Googh/flower_DR.png" alt="Input" style="width:200px; height:200px; object-fit:contain;">
    </td>
    <td style="text-align:center;">
      <img src="assets/Googh/flower1.png" alt="Normal" style="width:200px; height:200px; object-fit:contain;">
    </td>
    <td style="text-align:center;">
      <img src="assets/Googh/flower2.png" alt="3D" style="width:200px; height:200px; object-fit:contain;">
    </td>
  </tr>
  <tr>
    <td style="text-align:center;">
      <img src="assets/Googh/flower3.png" alt="Outline" style="width:200px; height:200px; object-fit:contain;">
    </td>
    <td style="text-align:center;">
      <img src="assets/Googh/flower4.png" alt="Flat" style="width:200px; height:200px; object-fit:contain;">
    </td>
    <td style="text-align:center;">
      <img src="assets/Googh/flower5.png" alt="Flat" style="width:200px; height:200px; object-fit:contain;">
    </td>
  </tr>
</table>

-----

## DontStarve

<table style="width:100%; height: 124px; table-layout: fixed;">
  <tr>
    <td colspan="5" style="text-align:center; font-weight:italic; font-size:0.9rem; padding-bottom:0px;">
    </td>
  </tr>
  <tr>
    <td style="text-align:center;">
      Flower<br>
      <img src="assets/DontStarve/flower_DR.png" alt="Input" style="width:150px; height:150px; object-fit:contain;">
    </td>
    <td style="text-align:center;">
      <br>
      <img src="assets/DontStarve/flower (1).png" alt="Normal" style="width:150px; height:150px; object-fit:contain;">
    </td>
    <td style="text-align:center;">
      <br>
      <img src="assets/DontStarve/flower (2).png" alt="Normal" style="width:150px; height:150px; object-fit:contain;">
    </td>
    <td style="text-align:center;">
      <br>
      <img src="assets/DontStarve/flower (3).png" alt="Normal" style="width:150px; height:150px; object-fit:contain;">
    </td>
    <td style="text-align:center;">
      <br>
      <img src="assets/DontStarve/flower (4).png" alt="Normal" style="width:150px; height:150px; object-fit:contain;">
    </td>
  </tr>
  <tr>
    <td style="text-align:center;">
      Gift<br>
      <img src="assets/DontStarve/gift_DR.png" alt="Input" style="width:150px; height:150px; object-fit:contain;">
    </td>
    <td style="text-align:center;">
      <br>
      <img src="assets/DontStarve/gift_14.png" alt="Normal" style="width:150px; height:150px; object-fit:contain;">
    </td>
    <td style="text-align:center;">
      <br>
      <img src="assets/DontStarve/gift_15.png" alt="Normal" style="width:150px; height:150px; object-fit:contain;">
    </td>
    <td style="text-align:center;">
      <br>
      <img src="assets/DontStarve/gift_16.png" alt="Normal" style="width:150px; height:150px; object-fit:contain;">
    </td>
    <td style="text-align:center;">
      <br>
      <img src="assets/DontStarve/gift_17.png" alt="Normal" style="width:150px; height:150px; object-fit:contain;">
    </td>
  <tr>
    <td style="text-align:center;">
      Carrot<br>
      <img src="assets/DontStarve/carrot_DR.png" alt="Input" style="width:150px; height:150px; object-fit:contain;">
    </td>
    <td style="text-align:center;">
      <br>
      <img src="assets/DontStarve/carrot_0.png" alt="Normal" style="width:150px; height:150px; object-fit:contain;">
    </td>
    <td style="text-align:center;">
      <br>
      <img src="assets/DontStarve/carrot_1.png" alt="Normal" style="width:150px; height:150px; object-fit:contain;">
    </td>
    <td style="text-align:center;">
      <br>
      <img src="assets/DontStarve/carrot_4.png" alt="Normal" style="width:150px; height:150px; object-fit:contain;">
    </td>
    <td style="text-align:center;">
      <br>
      <img src="assets/DontStarve/carrot_6.png" alt="Normal" style="width:150px; height:150px; object-fit:contain;">
    </td>
  </tr>
  <tr>
    <td style="text-align:center;">
      Rope<br>
      <img src="assets/DontStarve/rope_DR.png" alt="Input" style="width:150px; height:150px; object-fit:contain;">
    </td>
    <td style="text-align:center;">
      <br>
      <img src="assets/DontStarve/rope_0.png" alt="Normal" style="width:150px; height:150px; object-fit:contain;">
    </td>
    <td style="text-align:center;">
      <br>
      <img src="assets/DontStarve/rope_3.png" alt="Normal" style="width:150px; height:150px; object-fit:contain;">
    </td>
    <td style="text-align:center;">
      <br>
      <img src="assets/DontStarve/rope_4.png" alt="Normal" style="width:150px; height:150px; object-fit:contain;">
    </td>
    <td style="text-align:center;">
      <br>
      <img src="assets/DontStarve/rope_5.png" alt="Normal" style="width:150px; height:150px; object-fit:contain;">
    </td>
  </tr>
  <tr>
    <td style="text-align:center;">
      Potato<br>
      <img src="assets/DontStarve/potato_DR.png" alt="Input" style="width:150px; height:150px; object-fit:contain;">
    </td>
    <td style="text-align:center;">
      <br>
      <img src="assets/DontStarve/potato_0.png" alt="Normal" style="width:150px; height:150px; object-fit:contain;">
    </td>
    <td style="text-align:center;">
      <br>
      <img src="assets/DontStarve/potato_1.png" alt="Normal" style="width:150px; height:150px; object-fit:contain;">
    </td>
    <td style="text-align:center;">
      <br>
      <img src="assets/DontStarve/potato_5.png" alt="Normal" style="width:150px; height:150px; object-fit:contain;">
    </td>
    <td style="text-align:center;">
      <br>
      <img src="assets/DontStarve/potato_6.png" alt="Normal" style="width:150px; height:150px; object-fit:contain;">
    </td>
  </tr>
  <tr>
    <td style="text-align:center;">
      Heart<br>
      <img src="assets/heart.png" alt="Input" style="width:150px; height:150px; object-fit:contain;">
    </td>
    <td style="text-align:center;">
      <br>
      <img src="assets/DontStarve/heart_1.png" alt="Normal" style="width:150px; height:150px; object-fit:contain;">
    </td>
    <td style="text-align:center;">
      <br>
      <img src="assets/DontStarve/heart_0.png" alt="Normal" style="width:150px; height:150px; object-fit:contain;">
    </td>
    <td style="text-align:center;">
      <br>
      <img src="assets/DontStarve/heart_2.png" alt="Normal" style="width:150px; height:150px; object-fit:contain;">
    </td>
    <td style="text-align:center;">
      <br>
      <img src="assets/DontStarve/heart_4.png" alt="Normal" style="width:150px; height:150px; object-fit:contain;">
    </td>
  </tr>
  <tr>
    <td style="text-align:center;">
      Axe<br>
      <img src="assets/axe.png" alt="Input" style="width:150px; height:150px; object-fit:contain;">
    </td>
    <td style="text-align:center;">
      <br>
      <img src="assets/DontStarve/axe_0.png" alt="Normal" style="width:150px; height:150px; object-fit:contain;">
    </td>
    <td style="text-align:center;">
      <br>
      <img src="assets/DontStarve/axe_2.png" alt="Normal" style="width:150px; height:150px; object-fit:contain;">
    </td>
    <td style="text-align:center;">
      <br>
      <img src="assets/DontStarve/axe_3.png" alt="Normal" style="width:150px; height:150px; object-fit:contain;">
    </td>
    <td style="text-align:center;">
      <br>
      <img src="assets/DontStarve/axe_5.png" alt="Normal" style="width:150px; height:150px; object-fit:contain;">
    </td>
  </tr>
  <tr>
    <td style="text-align:center;">
      Potion<br>
      <img src="assets/potion.png" alt="Input" style="width:150px; height:150px; object-fit:contain;">
    </td>
    <td style="text-align:center;">
      <br>
      <img src="assets/DontStarve/potion_0.png" alt="Normal" style="width:150px; height:150px; object-fit:contain;">
    </td>
    <td style="text-align:center;">
      <br>
      <img src="assets/DontStarve/potion_5.png" alt="Normal" style="width:150px; height:150px; object-fit:contain;">
    </td>
    <td style="text-align:center;">
      <br>
      <img src="assets/DontStarve/potion_8.png" alt="Normal" style="width:150px; height:150px; object-fit:contain;">
    </td>
    <td style="text-align:center;">
      <br>
      <img src="assets/DontStarve/potion_10.png" alt="Normal" style="width:150px; height:150px; object-fit:contain;">
    </td>
  </tr>
  <tr>
    <td style="text-align:center;">
      Torch<br>
      <img src="assets/torch.png" alt="Input" style="width:150px; height:150px; object-fit:contain;">
    </td>
    <td style="text-align:center;">
      <br>
      <img src="assets/DontStarve/torch_0.png" alt="Normal" style="width:150px; height:150px; object-fit:contain;">
    </td>
    <td style="text-align:center;">
      <br>
      <img src="assets/DontStarve/torch_1.png" alt="Normal" style="width:150px; height:150px; object-fit:contain;">
    </td>
    <td style="text-align:center;">
      <br>
      <img src="assets/DontStarve/torch_2.png" alt="Normal" style="width:150px; height:150px; object-fit:contain;">
    </td>
    <td style="text-align:center;">
      <br>
      <img src="assets/DontStarve/torch_3.png" alt="Normal" style="width:150px; height:150px; object-fit:contain;">
    </td>
  </tr>
</table>



The model shows great color understanding as a byproduct of the InstructPix2Pix architecture.

<table style="width:100%; height: 164px; table-layout: fixed;">
  <tr>
    <td colspan="8" style="text-align:center; font-weight:bold; font-size:0.9rem; padding-bottom:8px;">
      Prompt: f9, flower, stylized. *color, green, white
    </td>
  </tr>
  <tr>
    <td style="text-align:center;">
      <strong>input</strong><br>
      <img src="assets/flowerInput.png" alt="Flower Input" style="width:150px; height:150px object-fit:contain;">
    </td>
    <td style="text-align:center;">
      <strong>red</strong><br>
      <img src="assets/flower2.png" alt="Flower red" style="width:150px; height:150px object-fit:contain;">
    </td>
    <td style="text-align:center;">
      <strong>blue</strong><br>
      <img src="assets/flower3.png" alt="Flower light blue" style="width:150px; height:150px object-fit:contain;">
    </td>
    <td style="text-align:center;">
      <strong>purple</strong><br>
      <img src="assets/flower4.png" alt="Flower purple" style="width:150px; height:150px object-fit:contain;">
    </td>
  </tr>
  <tr>
    <td style="text-align:center;">
      <strong>green</strong><br>
      <img src="assets/flower1.png" alt="Flower green" style="width:150px; height:150px object-fit:contain;">
    </td>
    <td style="text-align:center;">
      <strong>cyan</strong><br>
      <img src="assets/flower6.png" alt="Flower cyan" style="width:150px; height:150px object-fit:contain;">
    </td>
    <td style="text-align:center;">
      <strong>yellow</strong><br>
      <img src="assets/flower7.png" alt="Flower light green" style="width:150px; height:150px object-fit:contain;">
    </td>
    <td style="text-align:center;">
      <strong>orange</strong><br>
      <img src="assets/flower8.png" alt="Flower orange" style="width:150px; height:150px object-fit:contain;">
    </td>
  </tr>
</table>

The model generates acceptable results with as few as 4 steps.

<table style="width:100%; height: 164px; table-layout: fixed;">
  <tr>
    <td colspan="8" style="text-align:center; font-weight:bold; font-size:0.9rem; padding-bottom:8px;">
      Prompt: f4, alien, red skin, white shirt, white background.
    </td>
  </tr>
  <tr>
    <td style="text-align:center;">
      <strong>Drawing</strong><br>
      <img src="assets/alien/alienDrawing.png" alt="Alien Drawing" style="width:150px; height:150px; height:auto; object-fit:contain;">
    </td>
    <td style="text-align:center;">
      <strong>4 </strong><br>
      <img src="assets/alien/AlienD_4steps.png" alt="Alien 4" style="width:150px; height:150px; height:auto; object-fit:contain;">
    </td>
    <td style="text-align:center;">
      <strong>8 </strong><br>
      <img src="assets/alien/AlienD_8steps.png" alt="Alien 8" style="width:150px; height:150px; height:auto; object-fit:contain;">
    </td>
    <td style="text-align:center;">
      <strong>14</strong><br>
      <img src="assets/alien/AlienD_14steps.png" alt="Alien 14" style="width:150px; height:150px; height:auto; object-fit:contain;">
    </td>
  </tr>
  <tr>
    <td style="text-align:center;">
      <strong>20</strong><br>
      <img src="assets/alien/AlienD_20steps.png" alt="Alien 20" style="width:150px; height:150px; height:auto; object-fit:contain;">
    </td>
    <td style="text-align:center;">
      <strong>32</strong><br>
      <img src="assets/alien/AlienD_32steps.png" alt="Alien 32" style="width:150px; height:150px; height:auto; object-fit:contain;">
    </td>
    <td style="text-align:center;">
      <strong>44</strong><br>
      <img src="assets/alien/AlienD_44steps.png" alt="Alien 44" style="width:150px; height:150px; height:auto; object-fit:contain;">
    </td>
    <td style="text-align:center;">
      <strong>60</strong><br>
      <img src="assets/alien/AlienD_60steps.png" alt="Alien 60" style="width:150px; height:150px; height:auto; object-fit:contain;">
    </td>
  </tr>
</table>

<table style="width:100%; height: 164px; table-layout: fixed;">
  <tr>
    <td colspan="8" style="text-align:center; font-weight:bold; font-size:0.9rem; padding-bottom:8px;">
      Prompt: f4, alien, red skin, white shirt, white background.
    </td>
  </tr>
  <tr>
    <td style="text-align:center;">
      <strong>Canny</strong><br>
      <img src="assets/alien/alienCanny.png" alt="Alien Canny" style="width:150px; height:150px; height:auto; object-fit:contain;">
    </td>
    <td style="text-align:center;">
      <strong>4 </strong><br>
      <img src="assets/alien/AlienC_4steps.png" alt="Alien Canny 4" style="width:150px; height:150px; height:auto; object-fit:contain;">
    </td>
    <td style="text-align:center;">
      <strong>8 </strong><br>
      <img src="assets/alien/AlienC_8steps.png" alt="Alien Canny 8" style="width:150px; height:150px; height:auto; object-fit:contain;">
    </td>
    <td style="text-align:center;">
      <strong>14</strong><br>
      <img src="assets/alien/AlienC_14steps.png" alt="Alien Canny 14" style="width:150px; height:150px; height:auto; object-fit:contain;">
    </td>
  </tr>
  <tr>
    <td style="text-align:center;">
      <strong>20</strong><br>
      <img src="assets/alien/AlienC_20steps.png" alt="Alien Canny 20" style="width:150px; height:150px; height:auto; object-fit:contain;">
    </td>
    <td style="text-align:center;">
      <strong>32</strong><br>
      <img src="assets/alien/AlienC_32steps.png" alt="Alien Canny 32" style="width:150px; height:150px; height:auto; object-fit:contain;">
    </td>
    <td style="text-align:center;">
      <strong>44</strong><br>
      <img src="assets/alien/AlienC_44steps.png" alt="Alien Canny 44" style="width:150px; height:150px; height:auto; object-fit:contain;">
    </td>
    <td style="text-align:center;">
      <strong>60</strong><br>
      <img src="assets/alien/AlienC_60steps.png" alt="Alien Canny 60" style="width:150px; height:150px; height:auto; object-fit:contain;">
    </td>
  </tr>
</table>

<details>
  <summary><strong>Limitations</strong></summary>
  
  - The **Model** was trained mainly on objects, items. Things rather than Characters.
  - Swords and Blades are a work in progress (lack of Doodle inputs).
  - Flat style wasn't properly learned due to lack of data.
  - Fidelity 0 (f0) is actually high fidelity due to lack of data.
  - It inherits most of the limitations of the StableDiffusion 2.1 model.
  - Training was done at minimum batch size and resolution cause of GPU limitations.
    
</details>

<details>
  <summary><strong>Reasoning</strong></summary>
  
  <p>
    The objective is to train a model able to take drawings as inputs.
  </p>
  
  <p>
    While most models and controlnets were trained using canny or similar line extractors as inputs (which focus on the most prominent lines in an image),
  drawings are made with intention. A few squiggly lines placed in the right place can sometimes deliver a much better idea of what's being represented in the image:
  </p>
  
  <table style="width: 60%; table-layout: fixed;">
    <tr>
      <td style="text-align: center;">
        <strong>Drawing</strong><br>
        <img src="assets/alien/alienDrawing.png" alt="Drawing" style="width: 60%; max-width: 240px; height: auto; object-fit: contain;">
      </td>
      <td style="text-align: center;">
        <strong>Canny</strong><br>
        <img src="assets/alien/alienCanny.png" alt="Canny" style="width: 60%; max-width: 240px; height: auto; object-fit: contain;">
      </td>
    </tr>
  </table>
  
  <p>
    To address this, I train a *Fidelity embedding* that injects an explicit fidelity signal into the Unet, allowing it to modulate its denoising behavior accordingly.
  </p>
  
  <p>
    The FidelityMLP (ranging from 0 to 9; f0–f9) lets users decide how much the model should "correct" their drawing. 
 </p> 
 <p> 
  Although the InstructPix2Pix pipeline supports an ImageGuidance factor to control adherence to the input image, it tends to follow the drawing too strictly at higher values while losing compositional nuances at lower values.
 </p> 
  
  
</details>


# TODOs

<details>
  <summary><strong>DATA</strong></summary>
  
- [ ] Increase amount of hand-drawn line inputs
- [ ] Smaller-Bigger subject variations
- [ ] Background Variations
- [ ] Increase Flat style references
- [ ] Improve color matches in prompts
- [ ] Clean up

</details>

<details>
  <summary><strong>Training</strong></summary>
  
- [ ] Train full-precision with bigger batch size.
- [ ] Implement "Details" injection.
- [ ] Release V1.
- [ ] Release DoodleCharacters (DoodlePix but for characters)
- [ ] Release Lora Training code
- [ ] Test Bigger Models
      
</details>

## Credits

 - This is a custom implementation of the [Training](https://github.com/huggingface/diffusers/blob/main/examples/instruct_pix2pix/train_instruct_pix2pix.py) and [Pipeline](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_instruct_pix2pix.py) scripts from the [Diffusers repo](https://github.com/huggingface/diffusers)
  
 - Dataset was generated using Chat based DALLE-3, FLUX-1.1 PRO and [FLUX-REDUX-DEV](https://huggingface.co/black-forest-labs/FLUX.1-Redux-dev)
   
 - Edge extraction was made easy thanks to [Fannovel16's ComfyUI Controlnet Aux](https://github.com/Fannovel16/comfyui_controlnet_aux)

 - [ComfyUI](https://www.comfy.org/) was a big part of the Data Development process
 - Around 40% of the images were captioned using [Moondream2](https://huggingface.co/vikhyatk/moondream2)
 - Dataset Handlers were built using [PyQT](https://doc.qt.io/qtforpython-6/index.html)
 - Huge Thanks to the OpenSource community for hosting and sharing so much cool stuff online

bl_info = {
    "name": "DoodlePix",
    "blender": (4, 0, 2),
    "category": "Render",
    "description": "Generate images from doodles or 3D renders using InstructPix2Pix",
    "author": "Your Name",
    "version": (1, 0, 0),
    "location": "ObjectProperties"
}
from datetime import datetime

import os
import bpy
import time
import threading
from PIL import Image
from . import doodle_pipeline
import torch
import numpy as np

# Global variables
pipeline_instance = None
last_rendered_image_path = None

class DoodlePixProperties(bpy.types.PropertyGroup):
    # Model Setup
    model_path: bpy.props.StringProperty(
        name="Model Path",
        description="Path to local model folder",
        subtype='DIR_PATH'
    ) # type: ignore
    
    use_loaded_model: bpy.props.BoolProperty(
        name="Use Loaded Model",
        description="Use previously loaded model",
        default=False
    ) # type: ignore

    # Mode Selection
    is_drawing_mode: bpy.props.BoolProperty(
        name="Drawing Mode",
        description="Toggle between drawing and 3D render mode",
        default=True
    ) # type: ignore

    control_type: bpy.props.EnumProperty(
        name="Control Type",
        description="Type of control net processing for 3D mode",
        items=[
            ('canny', "Canny Edge", ""),
            ('scribble', "Scribble", ""),
            ('hed', "HED", "")
        ],
        default='canny'
    ) # type: ignore

    # Prompt Building
    is_doodle: bpy.props.BoolProperty(
        name="Doodle Mode",
        description="Add doodle prefix to prompt",
        default=True
    ) # type: ignore

    style_mode: bpy.props.EnumProperty(
        name="Style",
        description="Select generation style",
        items=[
            ('fantasy', "Fantasy", "Fantasy style", 'GHOST_ENABLED', 0),
            ('cute', "Cute", "Cute style", 'HEART', 1),
            ('3d', "3D", "3D style", 'VIEW3D', 2),
        ],
        default='fantasy'
    ) # type: ignore

    # New Prompt Components
    subject: bpy.props.StringProperty(
        name="Subject (s)",
        description="Main subject of the image"
    ) # type: ignore

    colors: bpy.props.StringProperty(
        name="Colors & Materials (c)",
        description="Colors and materials description"
    ) # type: ignore

    world_style: bpy.props.EnumProperty(
        name="World Style (w)",
        description="World style for the generation",
        items=[
            ('earth', "Earth", "Earth style", 'WORLD', 0),
            ('galactic', "Galactic", "Galactic style", 'LIGHT_SUN', 1),
            ('fantasy', "Fantasy", "Fantasy style", 'GHOST_ENABLED', 2),
            ('whimsy', "Whimsy", "Whimsical style", 'HEART', 3),
        ],
        default='fantasy'
    ) # type: ignore

    complexity: bpy.props.IntProperty(
        name="Complexity (k)",
        description="Complexity level of the generation",
        min=0,
        max=10,
        default=3
    ) # type: ignore

    perspective: bpy.props.EnumProperty(
        name="Perspective (p)",
        description="Perspective style",
        items=[
            ('flat', "Flat", "Flat style", 'MESH_PLANE', 0),
            ('3d', "3D", "3D style", 'VIEW3D', 1),
            ('painted', "Painted", "Painted style", 'BRUSH_DATA', 2),
        ],
        default='3d'
    ) # type: ignore

    description: bpy.props.StringProperty(
        name="Description (d)",
        description="Detailed description of the desired output"
    ) # type: ignore

    # Generation Parameters
    num_inference_steps: bpy.props.IntProperty(
        name="Steps",
        description="Number of inference steps",
        default=20,
        min=1,
        max=100
    ) # type: ignore

    guidance_scale: bpy.props.FloatProperty(
        name="Guidance Scale",
        description="Guidance scale for generation",
        default=7.5,
        min=1.0,
        max=20.0
    ) # type: ignore

    image_guidance_scale: bpy.props.FloatProperty(
        name="Image Guidance Scale",
        description="Image guidance scale",
        default=1.5,
        min=0.0,
        max=5.0
    ) # type: ignore

    # Output Settings
    output_path: bpy.props.StringProperty(
        name="Output Path",
        description="Path to save generated images",
        subtype='DIR_PATH',
        default="//generated_images/"
    ) # type: ignore

    negative_prompt: bpy.props.StringProperty(
        name="Negative Prompt",
        description="What to avoid in generation"
    ) # type: ignore

    seed: bpy.props.IntProperty(
        name="Seed",
        description="Random seed for generation",
        default=42
    ) # type: ignore

    # Scheduler Selection
    scheduler_type: bpy.props.EnumProperty(
        name="Scheduler",
        description="Select the scheduler for generation",
        items=[
            ('DDIM', "DDIM", "DDIM Scheduler"),
            ('DDPM', "DDPM", "DDPM Scheduler"),
            ('PNDM', "PNDM", "PNDM Scheduler"),
            ('Euler A', "Euler A", "Euler Ancestral Scheduler"),
            ('DPM++', "DPM++", "DPM++ Solver"),
        ],
        default='DDIM'
    ) # type: ignore
    
    # LoRA Text Encoder Path
    text_encoder_lora_path: bpy.props.StringProperty(
        name="Text Encoder LoRA",
        description="Path to LoRA weights for text encoder",
        subtype='FILE_PATH'
    ) # type: ignore
    
    use_text_encoder_lora: bpy.props.BoolProperty(
        name="Use Text Encoder LoRA",
        description="Enable custom text encoder LoRA",
        default=False
    ) # type: ignore

class DoodlePixSetWorldStyleOperator(bpy.types.Operator):
    bl_idname = "doodlepix.set_world_style"
    bl_label = "Set World Style"
    bl_description = "Set the world style for generation"
    
    style: bpy.props.StringProperty()
    
    def execute(self, context):
        context.scene.doodle_pix.world_style = self.style
        return {'FINISHED'}

class DoodlePixSetPerspectiveOperator(bpy.types.Operator):
    bl_idname = "doodlepix.set_perspective"
    bl_label = "Set Perspective"
    bl_description = "Set the perspective style"
    
    perspective: bpy.props.StringProperty()
    
    def execute(self, context):
        context.scene.doodle_pix.perspective = self.perspective
        return {'FINISHED'}
    
class DoodlePixPanel(bpy.types.Panel):
    bl_label = "DoodlePix Generator"
    bl_idname = "VIEW3D_PT_doodle_pix"
    bl_space_type = 'PROPERTIES'
    bl_region_type = 'WINDOW'
    bl_context = "object"
    bl_category = "DoodlePix"
    bl_options = {'DEFAULT_CLOSED'}
    
    def draw(self, context):
        layout = self.layout
        props = context.scene.doodle_pix

        # Model Setup
        box = layout.box()
        box.label(text="Model Setup", icon='FILE_BLEND')
        box.prop(props, "model_path")
        box.prop(props, "use_loaded_model")
        box.prop(props, "scheduler_type")
        
        # LoRA Setup
        box = layout.box()
        box.label(text="LoRA Setup", icon='FILE_FONT')
        box.prop(props, "use_text_encoder_lora")
        if props.use_text_encoder_lora:
            box.prop(props, "text_encoder_lora_path")

        # Mode Selection
        box = layout.box()
        box.label(text="Mode Selection", icon='MODIFIER')
        row = box.row()
        row.prop(props, "is_drawing_mode", text="Drawing Mode" if props.is_drawing_mode else "3D Mode")
        
        if not props.is_drawing_mode:
            box.prop(props, "control_type")

        # Updated Prompt Building
        box = layout.box()
        box.label(text="Prompt Building", icon='TEXT')
        box.prop(props, "is_doodle")
        
        # Style Selection
        box.label(text="Style:")
        row = box.row(align=True)
        for style in ['fantasy', 'cute', '3d']:
            icon = 'RADIOBUT_ON' if props.style_mode == style else 'RADIOBUT_OFF'
            row.operator(
                "doodlepix.set_style",
                text=style.capitalize(),
                icon=icon
            ).style = style

        # Prompt Components
        box.prop(props, "subject")
        box.prop(props, "colors")
        box.prop(props, "world_style")
        box.prop(props, "complexity")
        box.prop(props, "perspective")
        box.prop(props, "description")

        # Generation Parameters
        box = layout.box()
        box.label(text="Generation Parameters", icon='SETTINGS')
        box.prop(props, "num_inference_steps")
        box.prop(props, "guidance_scale")
        box.prop(props, "image_guidance_scale")

        # Output Settings
        box = layout.box()
        box.label(text="Output Settings", icon='OUTPUT')
        box.prop(props, "output_path")
        box.prop(props, "negative_prompt")
        box.prop(props, "seed")
        
        box = layout.box()
        box.label(text="Run Settings", icon='FILE_BACKUP')
        row = box.row(align=True)
        row.operator("doodlepix.save_settings", icon='FILE_TICK')
        row.operator("doodlepix.load_settings", icon='FILE_REFRESH')

        # Operators
        box = layout.box()
        row = box.row(align=True)
        row.operator("doodlepix.generate", icon='RENDER_STILL')
        row.operator("doodlepix.offload", icon='GHOST_DISABLED')

class DoodlePixSetStyleOperator(bpy.types.Operator):
    bl_idname = "doodlepix.set_style"
    bl_label = "Set Style"
    bl_description = "Set the generation style"
    
    style: bpy.props.StringProperty()
    
    def execute(self, context):
        context.scene.doodle_pix.style_mode = self.style
        return {'FINISHED'}

class DoodlePixGenerateOperator(bpy.types.Operator):
    bl_idname = "doodlepix.generate"
    bl_label = "Generate Image"
    
    def execute(self, context):
        props = context.scene.doodle_pix
        global pipeline_instance
        
        try:
            # Initialize or get pipeline with scheduler and LoRA
            if pipeline_instance is None or not props.use_loaded_model:
                lora_path = props.text_encoder_lora_path if props.use_text_encoder_lora else None
                pipeline_instance = doodle_pipeline.setup_pipeline(
                    props.model_path,
                    scheduler_name=props.scheduler_type,
                    text_encoder_lora_path=lora_path
                )
            
            # Get the current render result
            bpy.ops.render.render(write_still=True)
            rendered_image = bpy.data.images['Render Result']
            
            # Convert render result to PIL Image
            image_path = os.path.join(bpy.path.abspath(props.output_path), "temp_render.png")
            rendered_image.save_render(image_path)
            input_image = Image.open(image_path)
            
            # Process image based on mode
            processed_image = doodle_pipeline.process_image(
                input_image,
                props.is_drawing_mode,
                props.control_type
            )
            process_path = os.path.join(bpy.path.abspath(props.output_path), "processed_image.png")
            processed_image.save(process_path)
            
            # Updated prompt building
            prompt = doodle_pipeline.build_prompt(
                subject=props.subject,
                colors=props.colors,
                world_style=props.world_style,
                complexity=props.complexity,
                perspective=props.perspective,
                description=props.description
            )
            print(f"Generated prompt: {prompt}")  # Debug print
            
            #set seed
            generator= torch.Generator("cuda").manual_seed(props.seed)
            
            # Generate image
            output_image = doodle_pipeline.generate_image(
                pipe=pipeline_instance,
                image=processed_image,
                prompt=prompt,
                negative_prompt=props.negative_prompt,
                num_inference_steps=props.num_inference_steps,
                guidance_scale=props.guidance_scale,
                image_guidance_scale=props.image_guidance_scale,
                generator=generator
            )
            
            # Update Blender's image viewer
            schedule_image_update(output_image)
            
            return {'FINISHED'}
            
        except Exception as e:
            self.report({'ERROR'}, f"Generation failed: {str(e)}")
            return {'CANCELLED'}
        
def schedule_image_update(image_pil):
    bpy.app.timers.register(lambda: update_image_in_blender(image_pil))

def update_image_in_blender(image_pil, image_name="generated_image"):
    # Flip the image vertically before converting to numpy array
    image_data = np.array(image_pil.convert("RGBA").transpose(Image.FLIP_TOP_BOTTOM))
    image_data = image_data / 255.0 
    flat_image_data = image_data.ravel()

    if image_name in bpy.data.images:
        bpy.data.images[image_name].name = "old_" + image_name

    image = bpy.data.images.new(name=image_name, width=image_pil.width, height=image_pil.height)
    image.pixels = list(flat_image_data)
    image.update()

    for area in bpy.context.screen.areas:
        if area.type == 'IMAGE_EDITOR':
            for space in area.spaces:
                if space.type == 'IMAGE_EDITOR':
                    space.image = image
    print("Image updated")
class DoodlePixSaveSettingsOperator(bpy.types.Operator):
    bl_idname = "doodlepix.save_settings"
    bl_label = "Save Run Settings"
    
    filepath: bpy.props.StringProperty(
        subtype='FILE_PATH',
        default="//doodlepix_settings.json"
    ) # type: ignore
    
    def execute(self, context):
        props = context.scene.doodle_pix
        
        settings = {
            "timestamp": datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
            "model_path": props.model_path,
            "is_drawing_mode": props.is_drawing_mode,
            "control_type": props.control_type,
            "is_doodle": props.is_doodle,
            "style_mode": props.style_mode,
            "subject": props.subject,
            "colors": props.colors,
            "world_style": props.world_style,
            "complexity": props.complexity,
            "perspective": props.perspective,
            "description": props.description,
            "num_inference_steps": props.num_inference_steps,
            "guidance_scale": props.guidance_scale,
            "image_guidance_scale": props.image_guidance_scale,
            "negative_prompt": props.negative_prompt,
            "seed": props.seed,
            "scheduler_type": props.scheduler_type,
            "use_text_encoder_lora": props.use_text_encoder_lora,
            "text_encoder_lora_path": props.text_encoder_lora_path,
        }
        
        filepath = bpy.path.abspath(self.filepath)
        if doodle_pipeline.save_run_settings(settings, filepath):
            self.report({'INFO'}, f"Settings saved to {filepath}")
        else:
            self.report({'ERROR'}, "Failed to save settings")
            
        return {'FINISHED'}
    
    def invoke(self, context, event):
        self.filepath = f"//doodlepix_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}

class DoodlePixLoadSettingsOperator(bpy.types.Operator):
    bl_idname = "doodlepix.load_settings"
    bl_label = "Load Run Settings"
    
    filepath: bpy.props.StringProperty(
        subtype='FILE_PATH'
    ) # type: ignore
    
    def execute(self, context):
        filepath = bpy.path.abspath(self.filepath)
        settings = doodle_pipeline.load_run_settings(filepath)
        
        if settings:
            props = context.scene.doodle_pix
            
            # Apply loaded settings
            for key, value in settings.items():
                if key != "timestamp" and hasattr(props, key):
                    setattr(props, key, value)
                    
            self.report({'INFO'}, f"Settings loaded from {filepath}")
        else:
            self.report({'ERROR'}, "Failed to load settings")
            
        return {'FINISHED'}
    
    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}    
class DoodlePixOffloadOperator(bpy.types.Operator):
    bl_idname = "doodlepix.offload"
    bl_label = "Offload Model"
    
    def execute(self, context):
        global pipeline_instance
        if pipeline_instance is not None:
            del pipeline_instance
            pipeline_instance = None
            torch.cuda.empty_cache()
            self.report({'INFO'}, "Model offloaded successfully")
        return {'FINISHED'}

def register():
    bpy.utils.register_class(DoodlePixProperties)
    bpy.utils.register_class(DoodlePixPanel)
    bpy.utils.register_class(DoodlePixSetStyleOperator)
    bpy.utils.register_class(DoodlePixGenerateOperator)
    bpy.utils.register_class(DoodlePixOffloadOperator)
    bpy.utils.register_class(DoodlePixSaveSettingsOperator)
    bpy.utils.register_class(DoodlePixLoadSettingsOperator)
    bpy.utils.register_class(DoodlePixSetWorldStyleOperator)
    bpy.utils.register_class(DoodlePixSetPerspectiveOperator)
    bpy.types.Scene.doodle_pix = bpy.props.PointerProperty(type=DoodlePixProperties)


def unregister():
    del bpy.types.Scene.doodle_pix
    bpy.utils.unregister_class(DoodlePixProperties)
    bpy.utils.unregister_class(DoodlePixPanel)
    bpy.utils.unregister_class(DoodlePixSetStyleOperator)
    bpy.utils.unregister_class(DoodlePixGenerateOperator)
    bpy.utils.unregister_class(DoodlePixOffloadOperator)
    bpy.utils.unregister_class(DoodlePixSaveSettingsOperator)
    bpy.utils.unregister_class(DoodlePixLoadSettingsOperator)
    bpy.utils.unregister_class(DoodlePixSetWorldStyleOperator)
    bpy.utils.unregister_class(DoodlePixSetPerspectiveOperator)



if __name__ == "__main__":
    register()
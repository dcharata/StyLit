'''
This script renders the Light Path Expressions   \

Read README.md for Usage and more information.
'''

import bpy
from bpy import context, data, ops

import os
import math

PATH_TO_OBJS = ''

class Blender():
    def __init__(self):
    	self.model = None
    def import_model(self, path):
    	old_obj = set(context.scene.objects)
    	bpy.ops.import_scene.obj(filepath=path, filter_glob="*.obj")
    	self.model = (set(context.scene.objects) - old_obj).pop()
    def set_material(self):
        # Uncomment these to apply subdivision and smoothing
        #bpy.ops.object.modifier_add(type='SUBSURF')
        #context.object.modifiers["Subdivision"].levels = 2
        #bpy.ops.object.shade_smooth()
        mat = bpy.data.materials.new(name="Mat_Tile")
        mat.use_nodes = True
        bsdf = mat.node_tree.nodes["Principled BSDF"]
        bsdf.inputs[0].default_value = (0.0504358, 0.0843873, 0.8, 1)        
        ob = context.view_layer.objects.active
        if ob.data.materials:
            ob.data.materials[0] = mat
        else:
            ob.data.materials.append(mat)
    def render_modality(self, path):
    	context.scene.render.filepath = path.split('.')[0] + '.png' 
    	bpy.ops.render.render(write_still=True)

def get_objs_in_directory(path):
    objs = []
    for f in os.listdir(path):
        f_split  = os.path.splitext(f)
        f_name = f_split[0]
        ext = f_split[1]
        if ext.lower() not in ['.obj']:
            continue
        objs.append(f)
    return objs

def execute_blender(path):
    DATA = get_objs_in_directory(path)
    blender_instance = Blender()
    for model_fname in DATA:
        blender_instance.import_model(os.path.join(path, model_fname))
        filename = model_fname.split('.')[0].split('/')[-1]
        context.view_layer.objects.active = bpy.data.objects[filename]
        blender_instance.set_material()
        blender_instance.render_modality(os.path.join(path, model_fname))
        bpy.ops.object.select_all(action='DESELECT')
        bpy.data.objects[filename].select_set(True)
        bpy.ops.object.delete()


if __name__ == "__main__":
    execute_blender(PATH_TO_OBJS)
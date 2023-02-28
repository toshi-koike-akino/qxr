# Blender --python blender_surface.py
# require 'rgb.png' and 'dep.png' for reconstructed RGB-D
# Credit: <https://medium.com/swlh/3d-surface-plots-in-blender-54b349e2398d>

import bpy
import os


# clean scene
def clean():
    for name in ["Camera", "Cube", "Surface"]:
        print("remove", name)
        try:
            bpy.data.objects.remove(bpy.data.objects[name])
        except Exception as err:
            print("ignore", err)


clean()


width = 640
height = 480
z_height = -1.1  # Displace modifier strength
tex_res = 1  # Texture resolution (1:1)
mesh_res = 2  # Mesh resolution (8:1)


# Variables

mesh_width = int(width / mesh_res)
mesh_height = int(height / mesh_res)
tex_width = int(width / tex_res)
tex_height = int(height / tex_res)
size = 2
aspect_ratio = width / height

# Create and name a grid

bpy.ops.mesh.primitive_grid_add(
    x_subdivisions=mesh_width, y_subdivisions=mesh_height, size=size, location=(0, 0, 0)
)
plotObject = bpy.context.active_object
plotObject.name = "Surface Plot"

# Size grid properly

plotObject.scale[0] = aspect_ratio
plotObject.scale[1] = 1

# Generate a displace and diffuse map
rgb_image = bpy.data.images.load("rgb.png")
dep_image = bpy.data.images.load("dep.png")


displace_image = bpy.data.images.new("Displace Map", width=tex_width, height=tex_height)
diffuse_image = bpy.data.images.new("Diffuse Map", width=tex_width, height=tex_height)


# Create a displace texture

displace_map = bpy.data.textures.new("Displace Texture", type="IMAGE")
# displace_map.image = displace_image
displace_map.image = dep_image

# Create a displace modifier

displace_mode = plotObject.modifiers.new("Displace", type="DISPLACE")
displace_mode.texture = displace_map
displace_mode.strength = z_height

# Create a material

material = bpy.data.materials.new(name="Plot Material")

# Use nodes

material.use_nodes = True

# Add Principled BSDF

bsdf = material.node_tree.nodes["Principled BSDF"]

# Add an ImageTexture

diffuse_map = material.node_tree.nodes.new("ShaderNodeTexImage")

# Set diffuse image

# diffuse_map.image = diffuse_image
diffuse_map.image = rgb_image

# Link ImageTexture to Principled BSDF

material.node_tree.links.new(bsdf.inputs["Base Color"], diffuse_map.outputs["Color"])

# Assign it to object

if plotObject.data.materials:
    plotObject.data.materials[0] = material
else:
    plotObject.data.materials.append(material)

# Shade smooth

mesh = bpy.context.active_object.data
for f in mesh.polygons:
    f.use_smooth = True


# save blend file
filepath = os.path.join(os.getcwd(), "surface.fbx")
bpy.ops.export_scene.fbx(
    filepath=filepath,
    check_existing=False,
    use_selection=True,
    use_visible=True,
    axis_forward="Y",
    axis_up="Z",
    use_mesh_modifiers=True,
    path_mode="COPY",
    embed_textures=True,
)

# filepath = os.path.join(os.getcwd(), "surface.blend")
# print("saving", filepath)
# bpy.ops.wm.save_as_mainfile(filepath=filepath)

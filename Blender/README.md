This script renders the Light Path Expressions

### Requirements

- Blender `>= v.2.8.0` and blender python (bpy, will be installed along with blender).


### USAGE

Tested on: Blender `v2.8.0` and Ubuntu `14.04`


```
blender --background <filename>.blend --factory-startup --python lpe_render.py -- --type={source, target} --guide={a,b,c,d,e,style} --model={Sphere, <path to .obj file>} --output_dir=./ --resolution=1200x912 --samples=50 --device={GPU,CPU}
```
example:

```
blender --background blender_scenes/lpe_render_a.blend --factory-startup --python lpe_render.py -- --type=source --guide=a --model=Sphere --output_dir=./ --resolution=1200x912 --samples=50  --device=GPU
```

### RESULT

resolution : 600x456 

![](output/600x456/result.png?raw=true)

resolution : 240x182 

![](output/240x182/result.png?raw=true)

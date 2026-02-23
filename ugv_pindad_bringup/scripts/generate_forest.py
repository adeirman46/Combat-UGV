#!/usr/bin/env python3
import os
import random
import numpy as np
import noise

# ============================================================================
# CONFIGURATION
# ============================================================================
PKG_DIR = os.path.join(os.environ['HOME'], 'ugv_pindad_real', 'ugv_pindad_bringup')
MODELS_DIR = os.path.join(PKG_DIR, 'models')
MAPS_DIR = os.path.join(PKG_DIR, 'maps')
WORLDS_DIR = os.path.join(PKG_DIR, 'worlds')
SDF_PATH = os.path.join(WORLDS_DIR, 'forest.sdf')

MAP_SIZE = 100        # Resolution of the terrain grid (100x100)
SCALE = 100.0         # Spatial size of the terrain in meters (100m x 100m)
MAX_HEIGHT = 8.0      # Maximum height of hills
TREE_COUNT = 150      # Number of trees to scatter

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(MAPS_DIR, exist_ok=True)
os.makedirs(WORLDS_DIR, exist_ok=True)

# ============================================================================
# TERRAIN GENERATION (OBJ MESH)
# ============================================================================
def generate_height_data():
    z_data = np.zeros((MAP_SIZE, MAP_SIZE))
    scale_noise = 25.0
    
    for y in range(MAP_SIZE):
        for x in range(MAP_SIZE):
            nx = x / scale_noise
            ny = y / scale_noise
            
            # Base hills (Large slopes)
            elevation = noise.pnoise2(nx * 0.5, ny * 0.5, octaves=4) * MAX_HEIGHT
            
            # Rough terrain (High frequency)
            roughness = noise.pnoise2(nx * 4.0, ny * 4.0, octaves=2) * 0.5
            
            z = elevation + roughness
            
            # Physical coordinates
            px = (x / (MAP_SIZE - 1)) * SCALE - (SCALE / 2)
            py = (y / (MAP_SIZE - 1)) * SCALE - (SCALE / 2)
            
            # Flatten the center spawn area for the UGV (Radius 5m)
            dist_to_center = np.sqrt(px**2 + py**2)
            if dist_to_center < 5.0:
                # Smooth transition to flat center
                z = z * (dist_to_center / 5.0)**2
            
            z_data[y, x] = z
            
    return z_data

def export_terrain_model(z_data):
    terrain_dir = os.path.join(MODELS_DIR, 'terrain')
    os.makedirs(terrain_dir, exist_ok=True)
    
    obj_path = os.path.join(terrain_dir, 'terrain.obj')
    print(f"Exporting terrain mesh to {obj_path}...")
    
    with open(obj_path, 'w') as f:
        f.write("# Procedural Terrain OBJ\n")
        
        # Write vertices
        for y in range(MAP_SIZE):
            for x in range(MAP_SIZE):
                px = (x / (MAP_SIZE - 1)) * SCALE - (SCALE / 2)
                py = (y / (MAP_SIZE - 1)) * SCALE - (SCALE / 2)
                pz = z_data[y, x]
                f.write(f"v {px:.4f} {py:.4f} {pz:.4f}\n")
                
        # Write dummy UV for texture mapping
        f.write("vt 0.0 0.0\n")
        f.write("vt 1.0 0.0\n")
        f.write("vt 0.0 1.0\n")
        f.write("vt 1.0 1.0\n")
        
        # Write faces (Triangles)
        for y in range(MAP_SIZE - 1):
            for x in range(MAP_SIZE - 1):
                i00 = y * MAP_SIZE + x + 1
                i10 = i00 + 1
                i01 = (y + 1) * MAP_SIZE + x + 1
                i11 = i01 + 1
                
                f.write(f"f {i00}/1 {i10}/2 {i01}/3\n")
                f.write(f"f {i10}/2 {i11}/4 {i01}/3\n")
                
    # Create model.config
    with open(os.path.join(terrain_dir, 'model.config'), 'w') as f:
        f.write("""<?xml version="1.0"?>
<model>
  <name>terrain</name>
  <version>1.0</version>
  <sdf version="1.8">model.sdf</sdf>
  <author><name>Generated</name></author>
  <description>Procedural terrain mesh</description>
</model>
""")

    # Create model.sdf
    with open(os.path.join(terrain_dir, 'model.sdf'), 'w') as f:
        f.write("""<?xml version="1.0" ?>
<sdf version="1.8">
  <model name="terrain">
    <static>true</static>
    <link name="link">
      <collision name="collision">
        <geometry>
          <mesh>
            <uri>model://terrain/terrain.obj</uri>
            <scale>1 1 1</scale>
          </mesh>
        </geometry>
        <surface>
          <friction>
            <ode>
              <mu>2.0</mu>
              <mu2>2.0</mu2>
            </ode>
          </friction>
        </surface>
      </collision>
      <visual name="visual">
        <geometry>
          <mesh>
            <uri>model://terrain/terrain.obj</uri>
            <scale>1 1 1</scale>
          </mesh>
        </geometry>
        <material>
          <ambient>0.2 0.35 0.15 1</ambient>
          <diffuse>0.2 0.35 0.15 1</diffuse>
          <specular>0.05 0.05 0.05 1</specular>
        </material>
      </visual>
    </link>
  </model>
</sdf>
""")

# ============================================================================
# WORLD SDF GENERATION
# ============================================================================
def generate_world_sdf(z_data):
    print(f"Generating SDF world at {SDF_PATH}...")
    
    trees_sdf = ""
    tree_models = ["pine_tree", "oak_tree"]
    
    for i in range(TREE_COUNT):
        r_x = random.randint(0, MAP_SIZE-1)
        r_y = random.randint(0, MAP_SIZE-1)
        
        px = (r_x / (MAP_SIZE - 1)) * SCALE - (SCALE / 2)
        py = (r_y / (MAP_SIZE - 1)) * SCALE - (SCALE / 2)
        pz = z_data[r_y, r_x]
        
        # Avoid spawn location
        if abs(px) < 6.0 and abs(py) < 6.0:
            continue
            
        model_name = random.choice(tree_models)
        yaw = random.uniform(0, 6.28)
        
        # Scale some trees randomly for variety
        # Since Fuel models might not support scaling in SDF 1.8 easily without grouping, we just use random rotations
        
        trees_sdf += f"""
    <include>
      <uri>model://{model_name}</uri>
      <name>{model_name}_{i}</name>
      <pose>{px:.2f} {py:.2f} {pz:.2f} 0 0 {yaw:.2f}</pose>
    </include>
"""

    sdf_content = f"""<?xml version="1.0" ?>
<sdf version="1.8">
  <world name="forest_world">
    
    <!-- Physics -->
    <physics name="default_physics" type="ode">
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1.0</real_time_factor>
      <real_time_update_rate>1000</real_time_update_rate>
    </physics>

    <!-- Gravity -->
    <gravity>0 0 -9.81</gravity>

    <!-- Required Gz Systems -->
    <plugin filename="ignition-gazebo-physics-system" name="ignition::gazebo::systems::Physics"/>
    <plugin filename="ignition-gazebo-user-commands-system" name="ignition::gazebo::systems::UserCommands"/>
    <plugin filename="ignition-gazebo-scene-broadcaster-system" name="ignition::gazebo::systems::SceneBroadcaster"/>
    <plugin filename="ignition-gazebo-sensors-system" name="ignition::gazebo::systems::Sensors">
      <render_engine>ogre2</render_engine>
    </plugin>
    <plugin filename="ignition-gazebo-contact-system" name="ignition::gazebo::systems::Contact"/>

    <!-- Environment Settings -->
    <scene>
      <ambient>0.5 0.6 0.7 1.0</ambient>
      <background>0.6 0.8 0.9 1.0</background>
      <shadows>true</shadows>
      <!-- Fog for atmospheric realism -->
      <fog>
         <type>linear</type>
         <color>0.6 0.8 0.9 1.0</color>
         <density>0.02</density>
         <start>15.0</start>
         <end>80.0</end>
      </fog>
    </scene>

    <light type="directional" name="sun">
      <cast_shadows>true</cast_shadows>
      <pose>0 0 20 0 0 0</pose>
      <diffuse>0.95 0.9 0.8 1</diffuse>
      <specular>0.2 0.2 0.2 1</specular>
      <attenuation>
        <range>1000</range>
        <constant>0.9</constant>
        <linear>0.01</linear>
        <quadratic>0.001</quadratic>
      </attenuation>
      <!-- Afternoon sun angle -->
      <direction>-0.7 0.3 -0.5</direction>
    </light>

    <!-- Terrain Mesh -->
    <include>
      <uri>model://terrain</uri>
      <name>procedural_terrain</name>
      <pose>0 0 0 0 0 0</pose>
    </include>

    <!-- Forest Trees -->
{trees_sdf}

  </world>
</sdf>
"""
    with open(SDF_PATH, 'w') as f:
        f.write(sdf_content)

if __name__ == '__main__':
    z_data = generate_height_data()
    export_terrain_model(z_data)
    generate_world_sdf(z_data)
    print("Forest generation complete.")

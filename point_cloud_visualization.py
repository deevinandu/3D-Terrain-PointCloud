import open3d as o3d
import numpy as np
import os

# File paths
base_name = "terrain_pointcloud"
files = [
    f"{base_name}.pcd",
    f"{base_name}.ply", 
    f"{base_name}.xyz"
]

# Try to read each file format
for file_path in files:
    if os.path.exists(file_path):
        print(f"\n--- Reading {file_path} ---")
        try:
            pcd = o3d.io.read_point_cloud(file_path)
            
            # Check if file was read successfully
            if len(pcd.points) == 0:
                print(f"Warning: {file_path} appears to be empty or couldn't be read properly")
                continue
            
            # Get points as numpy array
            points = np.asarray(pcd.points)
            
            # Print basic information
            print(f"Number of points: {len(points)}")
            print(f"Point cloud dimensions: {points.shape}")
            print(f"First 5 points:")
            print(points[:5])
            
            # Check if colors exist
            if len(pcd.colors) > 0:
                colors = np.asarray(pcd.colors)
                print(f"Has colors: {colors.shape}")
            
            # Check if normals exist
            if len(pcd.normals) > 0:
                normals = np.asarray(pcd.normals)
                print(f"Has normals: {normals.shape}")
            
            # Visualize the point cloud
            print(f"Visualizing {file_path}...")
            o3d.visualization.draw_geometries([pcd], 
                                            window_name=f"Point Cloud - {file_path}",
                                            width=800, height=600)
            
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
    else:
        print(f"File not found: {file_path}")

# Alternative: Try reading .txt file if it exists
txt_file = f"{base_name}.txt"
if os.path.exists(txt_file):
    print(f"\n--- Reading {txt_file} as text ---")
    try:
        # Try different delimiters
        for delimiter in [' ', ',', '\t']:
            try:
                points = np.loadtxt(txt_file, delimiter=delimiter)
                print(f"Successfully read with delimiter '{delimiter}'")
                print(f"Shape: {points.shape}")
                print(f"First 5 rows:")
                print(points[:5])
                
                # Create Open3D point cloud from numpy array
                pcd_from_txt = o3d.geometry.PointCloud()
                if points.shape[1] >= 3:
                    pcd_from_txt.points = o3d.utility.Vector3dVector(points[:, :3])
                    print("Visualizing TXT file as point cloud...")
                    o3d.visualization.draw_geometries([pcd_from_txt], 
                                                    window_name=f"Point Cloud - {txt_file}",
                                                    width=800, height=600)
                break
            except:
                continue
    except Exception as e:
        print(f"Error reading {txt_file}: {e}")
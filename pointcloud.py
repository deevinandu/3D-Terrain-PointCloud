import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import csv
from datetime import datetime

# --- Configuration ---
LOG_FILENAME = 'D:\mavlink_log_packets_20250703_010304.json'
LIDAR_SEPARATION_METERS = 0.10  # 10cm separation
QUALITY_THRESHOLD = 50  

def save_pointcloud_formats(terrain_data, base_filename='terrain_pointcloud'):
    """Save point cloud data in multiple formats."""
    if not terrain_data:
        print("No terrain data to save.")
        return
    
    formats = {
        'csv': save_pointcloud_csv,
        'ply': save_pointcloud_ply,
        'pcd': save_pointcloud_pcd,
        'xyz': save_pointcloud_xyz,
        'json': save_pointcloud_json,
        'npy': save_pointcloud_numpy
    }
    
    for format_name, save_func in formats.items():
        filename = f"{base_filename}.{format_name}"
        save_func(terrain_data, filename)

def save_pointcloud_csv(terrain_data, filename):
    """Save point cloud data to CSV format."""
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['X', 'Y', 'Z'])
        for point in terrain_data:
            writer.writerow([point[0], point[1], point[2]])
    print(f"Saved CSV: {filename} ({len(terrain_data)} points)")

def save_pointcloud_ply(terrain_data, filename):
    """Save point cloud data to PLY format."""
    with open(filename, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(terrain_data)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("end_header\n")
        for point in terrain_data:
            f.write(f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f}\n")
    print(f"Saved PLY: {filename} ({len(terrain_data)} points)")

def save_pointcloud_pcd(terrain_data, filename):
    """Save point cloud data to PCD format."""
    with open(filename, 'w') as f:
        f.write("# .PCD v0.7 - Point Cloud Data file format\n")
        f.write("VERSION 0.7\n")
        f.write("FIELDS x y z\n")
        f.write("SIZE 4 4 4\n")
        f.write("TYPE F F F\n")
        f.write("COUNT 1 1 1\n")
        f.write(f"WIDTH {len(terrain_data)}\n")
        f.write("HEIGHT 1\n")
        f.write("VIEWPOINT 0 0 0 1 0 0 0\n")
        f.write(f"POINTS {len(terrain_data)}\n")
        f.write("DATA ascii\n")
        for point in terrain_data:
            f.write(f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f}\n")
    print(f"Saved PCD: {filename} ({len(terrain_data)} points)")

def save_pointcloud_xyz(terrain_data, filename):
    """Save point cloud data to XYZ format."""
    with open(filename, 'w') as f:
        for point in terrain_data:
            f.write(f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f}\n")
    print(f"Saved XYZ: {filename} ({len(terrain_data)} points)")

def save_pointcloud_json(terrain_data, filename):
    """Save point cloud data to JSON format."""
    pointcloud_dict = {
        'points': [{'x': p[0], 'y': p[1], 'z': p[2]} for p in terrain_data],
        'count': len(terrain_data),
        'metadata': {
            'units': 'meters',
            'coordinate_system': 'NED_relative',
            'lidar_separation': LIDAR_SEPARATION_METERS,
            'generated_from': 'MAVLink optical flow and distance sensors'
        }
    }
    with open(filename, 'w') as f:
        json.dump(pointcloud_dict, f, indent=2)
    print(f"Saved JSON: {filename} ({len(terrain_data)} points)")

def save_pointcloud_numpy(terrain_data, filename):
    """Save point cloud data as NumPy array."""
    points_array = np.array(terrain_data)
    np.save(filename, points_array)
    print(f"Saved NumPy: {filename} ({len(terrain_data)} points)")

def extract_timestamp(packet):
    """Extract and parse timestamp from packet."""
    timestamp_str = packet.get('packet_utc_timestamp', '')
    try:
        return datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
    except:
        return None

def load_and_process_data(filename: str):
    """Load MAVLink data and reconstruct flight path and terrain point cloud using optical flow."""
    print(f"Loading data from '{filename}'...")
    try:
        with open(filename, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        return None, None, None, None
    
    # Data storage
    path = {'x': [], 'y': [], 'z': [], 'timestamps': []}
    attitude_data = {'roll': [], 'pitch': [], 'yaw': [], 'timestamps': []}
    sensor_quality = {'optical_flow_quality': [], 'distance_sensors': [], 'timestamps': []}
    terrain_points = []
    
    # State variables for optical flow integration
    current_x, current_y = 0.0, 0.0
    home_altitude = None
    prev_timestamp = None
    
    sorted_packet_ids = sorted(data.keys(), key=int)
    
    print("Processing packets...")
    for i, packet_id in enumerate(sorted_packet_ids):
        packet = data[packet_id]
        messages = packet.get('messages', {})
        timestamp = extract_timestamp(packet)
        
        # Calculate time step dynamically
        time_step = 0.1  # Default 10Hz
        if timestamp and prev_timestamp:
            time_step = (timestamp - prev_timestamp).total_seconds()
        
        # --- Flight Path via Optical Flow ---
        if 'OPTICAL_FLOW' in messages and 'AHRS2' in messages:
            flow_msg = messages['OPTICAL_FLOW'][0]
            ahrs_msg = messages['AHRS2'][0]
            
            flow_rate_x = flow_msg.get('flow_rate_x')
            flow_rate_y = flow_msg.get('flow_rate_y')
            quality = flow_msg.get('quality', 0)
            altitude = ahrs_msg.get('altitude')
            
            sensor_quality['optical_flow_quality'].append(quality)
            sensor_quality['timestamps'].append(timestamp)
            
            if all(v is not None for v in [flow_rate_x, flow_rate_y, altitude]) and quality > QUALITY_THRESHOLD:
                delta_x = -flow_rate_x * time_step
                delta_y = -flow_rate_y * time_step
                current_x += delta_x
                current_y += delta_y
                
                if home_altitude is None:
                    home_altitude = altitude
                
                relative_z = altitude - home_altitude
                
                path['x'].append(current_x)
                path['y'].append(current_y)
                path['z'].append(relative_z)
                path['timestamps'].append(timestamp)
                
                # --- Terrain Point Cloud ---
                if 'DISTANCE_SENSOR' in messages:
                    distances = {msg['id']: msg.get('current_distance') for msg in messages['DISTANCE_SENSOR']}
                    sensor_quality['distance_sensors'].append(distances)
                    
                    if 0 in distances and 1 in distances and distances[0] > 0 and distances[1] > 0:
                        dist_0_cm = distances[0]
                        dist_1_cm = distances[1]
                        
                        lidar_0_y_offset = -LIDAR_SEPARATION_METERS / 2.0
                        lidar_1_y_offset = LIDAR_SEPARATION_METERS / 2.0
                        
                        terrain_z_0 = relative_z - (dist_0_cm / 100.0)
                        terrain_z_1 = relative_z - (dist_1_cm / 100.0)
                        
                        terrain_x_0 = current_x
                        terrain_y_0 = current_y + lidar_0_y_offset
                        terrain_x_1 = current_x
                        terrain_y_1 = current_y + lidar_1_y_offset
                        
                        terrain_points.append((terrain_x_0, terrain_y_0, terrain_z_0))
                        terrain_points.append((terrain_x_1, terrain_y_1, terrain_z_1))
        
        # --- Attitude Data ---
        if 'ATTITUDE' in messages:
            att_msg = messages['ATTITUDE'][0]
            attitude_data['roll'].append(att_msg.get('roll', 0))
            attitude_data['pitch'].append(att_msg.get('pitch', 0))
            attitude_data['yaw'].append(att_msg.get('yaw', 0))
            attitude_data['timestamps'].append(timestamp)
        elif 'AHRS2' in messages:
            ahrs_msg = messages['AHRS2'][0]
            attitude_data['roll'].append(ahrs_msg.get('roll', 0))
            attitude_data['pitch'].append(ahrs_msg.get('pitch', 0))
            attitude_data['yaw'].append(ahrs_msg.get('yaw', 0))
            attitude_data['timestamps'].append(timestamp)
        
        prev_timestamp = timestamp
        if i % 100 == 0:
            print(f"Processed {i}/{len(sorted_packet_ids)} packets...")
    
    print(f"Data processing complete!")
    print(f"Flight path points: {len(path['x'])}")
    print(f"Terrain cloud points: {len(terrain_points)}")
    print(f"Attitude data points: {len(attitude_data['roll'])}")
    
    return path, attitude_data, sensor_quality, terrain_points

def plot_comprehensive_analysis(path, attitude_data, sensor_quality, terrain_points):
    """Create comprehensive visualization of flight path, terrain, attitude, and sensor quality."""
    fig = plt.figure(figsize=(18, 12))
    
    # --- Flight Path ---
    if path['x']:
        ax1 = fig.add_subplot(2, 2, 1, projection='3d')
        x, y, z = path['x'], path['y'], path['z']
        scatter = ax1.scatter(x, y, z, c=z, cmap='viridis', marker='o', s=20)
        ax1.plot(x, y, z, color='blue', alpha=0.6, linewidth=1)
        ax1.scatter(x[0], y[0], z[0], color='green', s=100, label='Start')
        ax1.scatter(x[-1], y[-1], z[-1], color='red', s=100, label='End')
        
        ax1.set_title('Flight Path (Optical Flow)', fontweight='bold')
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.set_zlabel('Z (m)')
        ax1.legend()
        fig.colorbar(scatter, ax=ax1, shrink=0.6)
    
    # --- Terrain Point Cloud ---
    if terrain_points:
        ax2 = fig.add_subplot(2, 2, 2, projection='3d')
        tx, ty, tz = zip(*terrain_points)
        scatter2 = ax2.scatter(tx, ty, tz, c=tz, cmap='terrain', marker='.', s=10)
        
        ax2.set_title('Terrain Point Cloud', fontweight='bold')
        ax2.set_xlabel('X (m)')
        ax2.set_ylabel('Y (m)')
        ax2.set_zlabel('Ground Altitude (m)')
        fig.colorbar(scatter2, ax=ax2, shrink=0.6)
    
    # --- Attitude Data ---
    if attitude_data['roll']:
        ax3 = fig.add_subplot(2, 2, 3)
        timestamps = range(len(attitude_data['roll']))
        ax3.plot(timestamps, np.degrees(attitude_data['roll']), label='Roll', color='red')
        ax3.plot(timestamps, np.degrees(attitude_data['pitch']), label='Pitch', color='green')
        ax3.plot(timestamps, np.degrees(attitude_data['yaw']), label='Yaw', color='blue')
        
        ax3.set_title('Attitude Data', fontweight='bold')
        ax3.set_xlabel('Time Steps')
        ax3.set_ylabel('Angle (degrees)')
        ax3.legend()
        ax3.grid(True)
    
    # --- Sensor Quality ---
    if sensor_quality['optical_flow_quality']:
        ax4 = fig.add_subplot(2, 2, 4)
        timestamps = range(len(sensor_quality['optical_flow_quality']))
        ax4.plot(timestamps, sensor_quality['optical_flow_quality'], 'b-', label='Optical Flow Quality')
        ax4.axhline(y=QUALITY_THRESHOLD, color='r', linestyle='--', label='Quality Threshold')
        
        ax4.set_title('Sensor Quality', fontweight='bold')
        ax4.set_xlabel('Time Steps')
        ax4.set_ylabel('Quality Score')
        ax4.legend()
        ax4.grid(True)
        ax4.set_ylim(0, 255)
    
    plt.tight_layout()
    return fig

def create_data_quality_report(sensor_quality, terrain_points):
    """Generate a data quality assessment report."""
    print("\n" + "="*60)
    print("DATA QUALITY ASSESSMENT REPORT")
    print("="*60)
    
    print(f"\n OPTICAL FLOW POSITIONING:")
    print(f"   • Terrain points generated: {len(terrain_points)}")
    if sensor_quality['optical_flow_quality']:
        avg_quality = np.mean(sensor_quality['optical_flow_quality'])
        good_quality_pct = sum(1 for q in sensor_quality['optical_flow_quality'] if q > QUALITY_THRESHOLD) / len(sensor_quality['optical_flow_quality']) * 100
        print(f"\n OPTICAL FLOW QUALITY:")
        print(f"   • Average quality: {avg_quality:.1f}/255")
        print(f"   • Good quality readings (>50): {good_quality_pct:.1f}%")
        if avg_quality > 70:
            print(f"   • Status:  EXCELLENT optical flow quality")
        elif avg_quality > 50:
            print(f"   • Status:  MODERATE optical flow quality")
        else:
            print(f"   • Status:  POOR optical flow quality")
    
    print("\n RECOMMENDATION:")
    print("   Using optical flow for positioning (no GPS)")
    print("   Consider recalibrating sensors if quality is consistently low")
    print("="*60)

if __name__ == "__main__":
    # Load and process data
    path, attitude_data, sensor_quality, terrain_points = load_and_process_data(LOG_FILENAME)
    
    # Generate data quality report
    if sensor_quality['optical_flow_quality']:
        create_data_quality_report(sensor_quality, terrain_points)
    
    # Save point cloud in multiple formats
    if terrain_points:
        print("\n--- Saving Point Cloud Data ---")
        save_pointcloud_formats(terrain_points)
    
    # Create visualization
    if path['x'] or terrain_points:
        print("\nGenerating visualization...")
        fig = plot_comprehensive_analysis(path, attitude_data, sensor_quality, terrain_points)
        print("\nDisplaying analysis. Close the plot window to exit.")
        plt.show()
    else:
        print("No data available for visualization.")
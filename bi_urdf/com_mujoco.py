# import mujoco
# import numpy as np

# # Load the model and data
# model = mujoco.MjModel.from_xml_path("biped_wheel_2dof_new.xml")  # Replace with your XML file path
# data = mujoco.MjData(model)

# # Initialize simulation
# mujoco.mj_step(model, data)  # Step simulation to initialize

# # Function to calculate the COM
# def calculate_com(model, data):
#     total_mass = 0
#     com_sum = np.zeros(3)

#     for i in range(model.nbody):
#         # Get the mass and position of each body
#         mass = model.body_mass[i]
#         pos = data.xipos[i]  # Position of the body COM in world coordinates
#         print(f'link index: {i} COM: {pos} mass: {mass}')

#         # Accumulate mass-weighted positions
#         com_sum += mass * pos
#         total_mass += mass

#     # Calculate overall COM
#     center_of_mass = com_sum / total_mass
#     return center_of_mass

# # Calculate COM at the current simulation state
# com = calculate_com(model, data)
# print("Center of Mass:", com)
import mujoco
import numpy as np

# Load the model and data
model = mujoco.MjModel.from_xml_path("biped_wheel_2dof.xml")  # Replace with your XML file path
data = mujoco.MjData(model)

# Initialize simulation
mujoco.mj_step(model, data)  # Step simulation to initialize

def calculate_com_in_base_frame(model, data, base_body_id):
    total_mass = 0.0
    com_sum = np.zeros(3)

    # Get base position and rotation
    base_pos = data.xipos[base_body_id]  # Position of the base in world coordinates
    base_rot = data.ximat[base_body_id].reshape(3, 3)  # Rotation matrix of the base

    for i in range(model.nbody):
        # Get body mass and world COM position
        mass = model.body_mass[i]
        world_com = data.xipos[i]

        # Transform COM to base coordinates
        local_com = world_com - base_pos  # Translate to base origin
        local_com = base_rot.T @ local_com  # Rotate into base frame
        print(f'link id: {i} link COM coordinates: {local_com} mass: {mass}')

        # Accumulate mass-weighted positions
        com_sum += mass * local_com
        total_mass += mass

    # Compute overall COM in base coordinates
    print(f'total mass: {total_mass}')
    center_of_mass_base = com_sum / total_mass
    return center_of_mass_base

# Replace `base_body_id` with the appropriate body index for your robot's base
base_body_id = 0  # Usually 0, but confirm with your robot's XML
com_base = calculate_com_in_base_frame(model, data, base_body_id)
print("Center of Mass in Base Coordinates:", com_base)

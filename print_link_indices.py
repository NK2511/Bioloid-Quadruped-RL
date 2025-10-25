import pybullet as p
import os

# Connect to the physics server
client = p.connect(p.DIRECT)

# Load the URDF file
urdf_path = os.path.join(os.path.dirname(__file__), r"C:\Users\nandh\Downloads\Bioloid_Quadruped_Model\Bioloid_Quadruped_Model.urdf")
robot_id = p.loadURDF(urdf_path)

num_joints = p.getNumJoints(robot_id)

print(f"--- Joint and Link Information for bioloid_ant_like.urdf ---")
print(f"Number of joints: {num_joints}")
print("\n{:<5} {:<30} {:<30}".format("Index", "Joint Name", "Link Name"))
print("-" * 65)

for i in range(num_joints):
    joint_info = p.getJointInfo(robot_id, i)
    joint_index = joint_info[0]
    joint_name = joint_info[1].decode('utf-8')
    link_name = joint_info[12].decode('utf-8')
    print("{:<5} {:<30} {:<30}".format(joint_index, joint_name, link_name))

print("\n")
p.disconnect()
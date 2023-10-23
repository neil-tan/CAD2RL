import numpy as np
from PIL import Image

def getCameraImage(p, width, height):
    img_arr = p.getCameraImage(
        width,
        height,
        viewMatrix=p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=[0, 0, 0.5],
            distance=2.5,
            yaw=0,
            pitch=-15,
            roll=0,
            upAxisIndex=2,
        ),
        projectionMatrix=p.computeProjectionMatrixFOV(
            fov=60,
            aspect=width/height,
            nearVal=0.01,
            farVal=100,
        ),
        shadow=True,
        lightDirection=[1, 1, 1],
    )
    w, h, rgba, depth, mask = img_arr
    rgba_image = Image.fromarray(rgba.reshape(h, w, 4).astype(np.uint8))

    depth = Image.fromarray((depth.reshape(h, w)*255).astype('uint8'))
    mask = Image.fromarray(np.interp(mask.reshape(h, w), (-1, mask.max()), (0, 255)).astype('uint8'))

    return rgba_image, depth, mask

def print_joint_info(p, body_id):

    joint_type_dict = {
        p.JOINT_REVOLUTE: 'revolute',
        p.JOINT_PRISMATIC: 'prismatic',
        p.JOINT_SPHERICAL: 'spherical',
        p.JOINT_PLANAR: 'planar',
        p.JOINT_FIXED: 'fixed'
    }

    for i in range(p.getNumJoints(body_id)):
        joint_info = p.getJointInfo(body_id, i)
        joint_type_int = joint_info[2]
        joint_type_str = joint_type_dict[joint_type_int]
        print(f"Joint Index: {i}, Name: {joint_info[1].decode('utf-8')}, Type: {joint_type_str}, Max Force: {joint_info[10]}, Max Velocity: {joint_info[11]}")

def print_joint_states(p, body_id, joint_id):
    joint_states = p.getJointState(body_id, joint_id)
    print("Joint States:")
    print(f"Joint Position: {joint_states[0]}, Joint Velocity: {joint_states[1]}, Joint Reaction Forces: {joint_states[2]}, Applied Joint Motor Torque: {joint_states[3]}")

def print_body_states(p, body_id, link_id):
    link_state = p.getLinkState(body_id, link_id, computeLinkVelocity=1)
    print("Link States:")
    print(f"Link Position: {link_state[0]}, Link World Position: {link_state[4]}, Link World Link Velocity: {link_state[6]}")
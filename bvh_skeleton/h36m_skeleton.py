from . import math3d
from . import bvh_helper

import numpy as np


class H36mSkeleton(object):

    def __init__(self):
        self.root = 'Hip'
        self.keypoint2index = {
        # Face Landmarks (0-10)
        "nose": 0,
        "left_eye_inner": 1,
        "left_eye": 2,
        "left_eye_outer": 3,
        "right_eye_inner": 4,
        "right_eye": 5,
        "right_eye_outer": 6,
        "left_ear": 7,
        "right_ear": 8,
        "mouth_left": 9,
        "mouth_right": 10,

        # Main Body Joints (11-24)
        "left_shoulder": 11,
        "right_shoulder": 12,
        "left_elbow": 13,
        "right_elbow": 14,
        "left_wrist": 15,
        "left_wrist_pose": 16,
        "right_wrist": 17,
        "right_wrist_pose": 18,
        "left_pinky": 19,
        "right_pinky": 20,
        "left_index": 21,
        "right_index": 22,
        "left_thumb": 23,
        "right_thumb": 24,

        # Leg Joints (25-36)
        "left_hip": 25,
        "left_hip_original": 26,
        "right_hip": 27,
        "right_hip_original": 28,
        "left_knee": 29,
        "right_knee": 30,
        "left_ankle": 31,
        "right_ankle": 32,
        "left_heel": 33,
        "right_heel": 34,
        "left_foot_index": 35,
        "right_foot_index": 36,

        # Torso and Arm Segments (37-46)
        "hips": 37,
        "neck": 38,
        "spine": 39,
        "spine1": 40,
        "spine2": 41,
        "left_arm": 42,
        "left_forearm": 43,
        "right_arm": 44,
        "right_forearm": 45,

        # Detailed Left Hand (46-67)
        "left_hand_wrist": 46,
        "left_hand_thumb_cmc": 47,
        "left_hand_thumb_mcp": 48,
        "left_hand_thumb_ip": 49,
        "left_hand_thumb_tip": 50,
        "left_hand_index_mcp": 51,
        "left_hand_index_pip": 52,
        "left_hand_index_dip": 53,
        "left_hand_index_tip": 54,
        "left_hand_middle_mcp": 55,
        "left_hand_middle_pip": 56,
        "left_hand_middle_dip": 57,
        "left_hand_middle_tip": 58,
        "left_hand_ring_mcp": 59,
        "left_hand_ring_pip": 60,
        "left_hand_ring_dip": 61,
        "left_hand_ring_tip": 62,
        "left_hand_pinky_mcp": 63,
        "left_hand_pinky_pip": 64,
        "left_hand_pinky_dip": 65,
        "left_hand_pinky_tip": 66,

        # Detailed Right Hand (67-88)
        "right_hand_wrist": 67,
        "right_hand_thumb_cmc": 68,
        "right_hand_thumb_mcp": 69,
        "right_hand_thumb_ip": 70,
        "right_hand_thumb_tip": 71,
        "right_hand_index_mcp": 72,
        "right_hand_index_pip": 73,
        "right_hand_index_dip": 74,
        "right_hand_index_tip": 75,
        "right_hand_middle_mcp": 76,
        "right_hand_middle_pip": 77,
        "right_hand_middle_dip": 78,
        "right_hand_middle_tip": 79,
        "right_hand_ring_mcp": 80,
        "right_hand_ring_pip": 81,
        "right_hand_ring_dip": 82,
        "right_hand_ring_tip": 83,
        "right_hand_pinky_mcp": 84,
        "right_hand_pinky_pip": 85,
        "right_hand_pinky_dip": 86,
        "right_hand_pinky_tip": 87,

        # Virtual/Derived Joints (88-89)
        "virtual_left_hand_base": 88,
        "virtual_right_hand_base": 89,
        }
        self.map_to_mediapipe: Dict[str, str] = self._get_mediapipe_map()
        self.index2keypoint = {v: k for k, v in self.keypoint2index.items()}
        self.keypoint_num = len(self.keypoint2index)

        self.children = {
    'mixamorigHips': ['mixamorigSpine', 'mixamorigRightUpLeg', 'mixamorigLeftUpLeg'],
    'mixamorigSpine': ['mixamorigSpine1'],
    'mixamorigSpine1': ['mixamorigSpine2'],
    'mixamorigSpine2': ['mixamorigNeck', 'mixamorigRightShoulder', 'mixamorigLeftShoulder'],
    'mixamorigNeck': ['mixamorigHead'],
    'mixamorigHead': ['mixamorigHeadTop_End'],
    'mixamorigHeadTop_End': [],
    'mixamorigRightShoulder': ['mixamorigRightArm'],
    'mixamorigRightArm': ['mixamorigRightForeArm'],
    'mixamorigRightForeArm': ['mixamorigRightHand'],
    'mixamorigRightHand': [
        'mixamorigRightHandThumb1', 
        'mixamorigRightHandIndex1', 
        'mixamorigRightHandMiddle1', 
        'mixamorigRightHandRing1', 
        'mixamorigRightHandPinky1'
    ],
    'mixamorigRightHandThumb1': ['mixamorigRightHandThumb2'],
    'mixamorigRightHandThumb2': ['mixamorigRightHandThumb3'],
    'mixamorigRightHandThumb3': ['mixamorigRightHandThumb4'],
    'mixamorigRightHandThumb4': ['mixamorigRightHandThumb4_End'],
    'mixamorigRightHandThumb4_End': [],
    'mixamorigRightHandIndex1': ['mixamorigRightHandIndex2'],
    'mixamorigRightHandIndex2': ['mixamorigRightHandIndex3'],
    'mixamorigRightHandIndex3': ['mixamorigRightHandIndex4'],
    'mixamorigRightHandIndex4': ['mixamorigRightHandIndex4_End'],
    'mixamorigRightHandIndex4_End': [],
    'mixamorigRightHandMiddle1': ['mixamorigRightHandMiddle2'],
    'mixamorigRightHandMiddle2': ['mixamorigRightHandMiddle3'],
    'mixamorigRightHandMiddle3': ['mixamorigRightHandMiddle4'],
    'mixamorigRightHandMiddle4': ['mixamorigRightHandMiddle4_End'],
    'mixamorigRightHandMiddle4_End': [],
    'mixamorigRightHandRing1': ['mixamorigRightHandRing2'],
    'mixamorigRightHandRing2': ['mixamorigRightHandRing3'],
    'mixamorigRightHandRing3': ['mixamorigRightHandRing4'],
    'mixamorigRightHandRing4': ['mixamorigRightHandRing4_End'],
    'mixamorigRightHandRing4_End': [],
    'mixamorigRightHandPinky1': ['mixamorigRightHandPinky2'],
    'mixamorigRightHandPinky2': ['mixamorigRightHandPinky3'],
    'mixamorigRightHandPinky3': ['mixamorigRightHandPinky4'],
    'mixamorigRightHandPinky4': ['mixamorigRightHandPinky4_End'],
    'mixamorigRightHandPinky4_End': [],
    'mixamorigLeftShoulder': ['mixamorigLeftArm'],
    'mixamorigLeftArm': ['mixamorigLeftForeArm'],
    'mixamorigLeftForeArm': ['mixamorigLeftHand'],
    'mixamorigLeftHand': [
        'mixamorigLeftHandThumb1', 
        'mixamorigLeftHandIndex1', 
        'mixamorigLeftHandMiddle1', 
        'mixamorigLeftHandRing1', 
        'mixamorigLeftHandPinky1'
    ],
    'mixamorigLeftHandThumb1': ['mixamorigLeftHandThumb2'],
    'mixamorigLeftHandThumb2': ['mixamorigLeftHandThumb3'],
    'mixamorigLeftHandThumb3': ['mixamorigLeftHandThumb4'],
    'mixamorigLeftHandThumb4': ['mixamorigLeftHandThumb4_End'],
    'mixamorigLeftHandThumb4_End': [],
    'mixamorigLeftHandIndex1': ['mixamorigLeftHandIndex2'],
    'mixamorigLeftHandIndex2': ['mixamorigLeftHandIndex3'],
    'mixamorigLeftHandIndex3': ['mixamorigLeftHandIndex4'],
    'mixamorigLeftHandIndex4': ['mixamorigLeftHandIndex4_End'],
    'mixamorigLeftHandIndex4_End': [],
    'mixamorigLeftHandMiddle1': ['mixamorigLeftHandMiddle2'],
    'mixamorigLeftHandMiddle2': ['mixamorigLeftHandMiddle3'],
    'mixamorigLeftHandMiddle3': ['mixamorigLeftHandMiddle4'],
    'mixamorigLeftHandMiddle4': ['mixamorigLeftHandMiddle4_End'],
    'mixamorigLeftHandMiddle4_End': [],
    'mixamorigLeftHandRing1': ['mixamorigLeftHandRing2'],
    'mixamorigLeftHandRing2': ['mixamorigLeftHandRing3'],
    'mixamorigLeftHandRing3': ['mixamorigLeftHandRing4'],
    'mixamorigLeftHandRing4': ['mixamorigLeftHandRing4_End'],
    'mixamorigLeftHandRing4_End': [],
    'mixamorigLeftHandPinky1': ['mixamorigLeftHandPinky2'],
    'mixamorigLeftHandPinky2': ['mixamorigLeftHandPinky3'],
    'mixamorigLeftHandPinky3': ['mixamorigLeftHandPinky4'],
    'mixamorigLeftHandPinky4': ['mixamorigLeftHandPinky4_End'],
    'mixamorigLeftHandPinky4_End': [],
    'mixamorigRightUpLeg': ['mixamorigRightLeg'],
    'mixamorigRightLeg': ['mixamorigRightFoot'],
    'mixamorigRightFoot': ['mixamorigRightToeBase'],
    'mixamorigRightToeBase': ['mixamorigRightToe_End'],
    'mixamorigRightToe_End': [],
    'mixamorigLeftUpLeg': ['mixamorigLeftLeg'],
    'mixamorigLeftLeg': ['mixamorigLeftFoot'],
    'mixamorigLeftFoot': ['mixamorigLeftToeBase'],
    'mixamorigLeftToeBase': ['mixamorigLeftToe_End'],
    'mixamorigLeftToe_End': []
    }

        self.parent = {self.root: None}
        for parent, children in self.children.items():
            for child in children:
                self.parent[child] = parent
                
        self.left_joints = [
            joint for joint in self.keypoint2index
            if 'Left' in joint
        ]
        self.right_joints = [
            joint for joint in self.keypoint2index
            if 'Right' in joint
        ]

        # human3.6m坐标系(Z向上，Y向前，X向左)下的T-pose
        self.initial_directions = self._get_initial_directions()
        self.header =  self._build_bvh_header(self.initial_directions, 100.0)

        # SmartBody坐标系(Y向上，Z向前，X向右)下的T-pose
        # self.initial_directions = {
        #     'Hip': [0, 0, 0],
        #     'RightHip': [-1, 0, 0],
        #     'RightKnee': [0, -1, 0],
        #     'RightAnkle': [0, -1, 0],
        #     'RightAnkleEndSite': [0, 0, -1],
        #     'LeftHip': [1, 0, 0],
        #     'LeftKnee': [0, -1, 0],
        #     'LeftAnkle': [0, -1, 0],
        #     'LeftAnkleEndSite': [0, 0, -1],
        #     'Spine': [0, 1, 0],
        #     'Thorax': [0, 1, 0],
        #     'Neck': [0, 1, 0],
        #     'HeadEndSite': [0, 1, 0],
        #     'LeftShoulder': [1, 0, 0],
        #     'LeftElbow': [1, 0, 0],
        #     'LeftWrist': [1, 0, 0],
        #     'LeftWristEndSite': [1, 0, 0],
        #     'RightShoulder': [-1, 0, 0],
        #     'RightElbow': [-1, 0, 0],
        #     'RightWrist': [-1, 0, 0],
        #     'RightWristEndSite': [-1, 0, 0]
        # }

        # self.initial_directions = {
        #     'Hips': [0, 0, 0],
        #     'RightUpLeg': [-1, 0, 0],
        #     'RightLeg': [0, -1, 0],
        #     'RightFoot': [0, -1, 0],
        #     'RightFoot_End': [0, 0, 1],
        #     'LeftUpLeg': [1, 0, 0],
        #     'LeftLeg': [0, -1, 0],
        #     'LeftFoot': [0, -1, 0],
        #     'LeftFoot_End': [0, 0, 1],
        #     'Spine': [0, 1, 0],
        #     'Spine3': [0, 1, 0],
        #     'Neck': [0, 1, 0],
        #     'Head': [0, 1, 0],
        #     'LeftArm': [1, 0, 0],
        #     'LeftForeArm': [1, 0, 0],
        #     'LeftHand': [1, 0, 0],
        #     'LeftWristEndSite': [1, 0, 0],
        #     'RightArm': [-1, 0, 0],
        #     'RightForeArm': [-1, 0, 0],
        #     'RightHand': [-1, 0, 0],
        #     'RightWristEndSite': [-1, 0, 0]
        # }
    def _build_bvh_header(self, initial_offsets: Dict[str, np.ndarray], scale_factor: float) -> BvhHeader:
        """
        Builds a BvhHeader object containing a structured hierarchy of BvhNode objects.

        This function uses a recursive helper to traverse the skeleton definition
        and instantiate BvhNode for each joint, linking them together.

        Args:
            initial_offsets (Dict[str, np.ndarray]): A dictionary mapping joint names
            to their T-pose offset vectors.
            scale_factor (float): A factor to scale all offset values.

        Returns:
            BvhHeader: A complete header object with the root node and a
            dictionary of all nodes in the hierarchy.
        """
        print("Constructing BvhHeader object...")
        all_nodes: Dict[str, BvhNode] = {}

        def _create_node(joint_name: str, parent_node: BvhNode = None) -> BvhNode:
            """Recursively creates a BvhNode and all its descendants."""
            
            # 1. Determine node properties
            is_root = (parent_node is None)
            is_end_site = "End" in joint_name or "end" in joint_name.lower()
            
            # The rotation order for all standard joints is set to 'zyx'.
            # This is derived from the "Zrotation Yrotation Xrotation" channel order.
            # You could make this more flexible if needed (e.g., read from a config).
            rotation_order = None if is_end_site else 'zxy'

            # Calculate the scaled offset for this joint
            offset = initial_offsets.get(joint_name, np.array([0.0, 0.0, 0.0])) * scale_factor

            # 2. Create the current node instance
            current_node = BvhNode(
                name=joint_name,
                offset=offset,
                rotation_order=rotation_order,
                parent=parent_node,
                is_root=is_root,
                is_end_site=is_end_site
            )
            all_nodes[joint_name] = current_node
            
            # 3. Recursively create and link children
            child_names = self.children_of.get(joint_name, [])
            for child_name in child_names:
                child_node = _create_node(child_name, parent_node=current_node)
                current_node.children.append(child_node)
                
            return current_node

        # Start the recursive creation from the root joint
        root_node = _create_node(self.root_name, parent_node=None)
        
        # Create and return the final BvhHeader object
        header = BvhHeader(root=root_node, nodes=all_nodes)
        print("BvhHeader object constructed successfully.")
        return header
    def _get_mediapipe_map(self) -> Dict[str, str]:
        """
        Maps Mixamo joints to the mp_source keys. This version correctly defines
        the endpoints for arm, forearm, and hand bones using the pre-calculated
        virtual points.
        """
        target_map = {
            # Body and Head
            "mixamorigHips": "hips", "mixamorigSpine": "spine", "mixamorigSpine1": "spine1", "mixamorigSpine2": "spine2",
            "mixamorigNeck": "neck", "mixamorigHead": "nose", "mixamorigHeadTop_End": "derived_headtop",
            
            # Left Arm Chain - Uses the derived arm/forearm and virtual hand base
            "mixamorigLeftShoulder": "left_shoulder",
            "mixamorigLeftArm": "left_arm",
            "mixamorigLeftForeArm": "left_forearm",
            "mixamorigLeftHand": "virtual_left_hand_base",
            
            # Right Arm Chain
            "mixamorigRightShoulder": "right_shoulder",
            "mixamorigRightArm": "right_arm",
            "mixamorigRightForeArm": "right_forearm",
            "mixamorigRightHand": "virtual_right_hand_base",

            # Legs
            "mixamorigRightUpLeg": "right_hip", "mixamorigRightLeg": "right_knee", "mixamorigRightFoot": "right_ankle", "mixamorigRightToeBase": "right_foot_index", "mixamorigRightToe_End": "derived_right_toe_end",
            "mixamorigLeftUpLeg": "left_hip", "mixamorigLeftLeg": "left_knee", "mixamorigLeftFoot": "left_ankle", "mixamorigLeftToeBase": "left_foot_index", "mixamorigLeftToe_End": "derived_left_toe_end",
        }
        
        # Add finger mappings
        for hand in ["Left", "Right"]:
            hand_prefix = f"{hand.lower()}_hand_"
            for finger in ["Index", "Middle", "Ring", "Pinky"]:
                for i, j_type in enumerate(["mcp", "pip", "dip", "tip"], 1):
                    target_map[f"mixamorig{hand}Hand{finger}{i}"] = f"{hand_prefix}{finger.lower()}_{j_type}"
            
            for i, j_type in enumerate(["cmc", "mcp", "ip", "tip"], 1):
                target_map[f"mixamorig{hand}HandThumb{i}"] = f"{hand_prefix}thumb_{j_type}"
            
            for finger in ["Thumb", "Index", "Middle", "Ring", "Pinky"]:
                # Handle inconsistent End Site names from original utils
                end_name1 = f"mixamorig{hand}Hand{finger}4_End"
                end_name2 = f"mixamorig{hand}Hand{finger}4_END"
                tip_key = f"{hand_prefix}{finger.lower()}_tip"
                if end_name1 in self.joint_names: target_map[end_name1] = tip_key
                if end_name2 in self.joint_names: target_map[end_name2] = tip_key
                
        return target_map
    def get_initial_offset(self, poses_3d):
        # TODO: RANSAC
        bone_lens = {self.root: [0]}
        stack = [self.root]
        while stack:
            parent = stack.pop()
            p_idx = self.keypoint2index[parent]
            for child in self.children[parent]:
                if 'EndSite' in child:
                    bone_lens[child] = 0.4 * bone_lens[parent] 
                    continue
                stack.append(child)

                c_idx = self.keypoint2index[child]
                bone_lens[child] = np.linalg.norm(
                    poses_3d[:, p_idx] - poses_3d[:, c_idx],
                    axis=1
                )

        bone_len = {}
        for joint in self.keypoint2index:
            if 'Left' in joint or 'Right' in joint:
                base_name = joint.replace('Left', '').replace('Right', '')
                left_len = np.mean(bone_lens['Left' + base_name])
                right_len = np.mean(bone_lens['Right' + base_name])
                bone_len[joint] = (left_len + right_len) / 2
            else:
                bone_len[joint] = np.mean(bone_lens[joint])

        initial_offset = {}
        for joint, direction in self.initial_directions.items():
            direction = np.array(direction) / max(np.linalg.norm(direction), 1e-12)
            initial_offset[joint] = direction * bone_len[joint]

        return initial_offset

    def _get_initial_directions(self) -> Dict[str, np.ndarray]:
            """
            Defines the T-pose orientation for a Y-up Mixamo rig with corrected finger splays
            and a wider, more stable 'A-pose' for the legs.
            """
            # Coordinate System: Y-up, X-right, Z-inward (Right-Handed)
            # Left Arm/Hand points along +X
            # Right Arm/Hand points along -X

            # Helper function to perform the coordinate system swap
            def _swizzle_y_up_to_z_up(y_up_vector: List[float]) -> List[float]:
                """Converts a Y-up vector [x, y, z] to a Z-up vector [x, y, z]."""
                old_x, old_y, old_z = y_up_vector
                # new_X = old_X,  new_Y = -old_Z,  new_Z = old_Y
                return [old_x, -old_z, old_y]
            
            y_up_directions = {
                # --- TORSO ---
                'mixamorigHips': [0, 0, 0],
                'mixamorigSpine': [0, 1, 0],
                'mixamorigSpine1': [0, 1, 0],
                'mixamorigSpine2': [0, 1, 0],
                'mixamorigNeck': [0, 1, 0],
                'mixamorigHead': [0, 1, 0],
                'mixamorigHeadTop_End': [0, 1, 0],

                # --- LEFT ARM ---
                'mixamorigLeftShoulder': [1, 0, 0],
                'mixamorigLeftArm': [1, 0, 0],
                'mixamorigLeftForeArm': [1, 0, 0],
                'mixamorigLeftHand': [1, 0, 0],

                # --- LEFT HAND FINGERS ---
                'mixamorigLeftHandThumb1': [0.5, 0.2, 0.8], 'mixamorigLeftHandThumb2': [0.7, 0.1, 0.7], 'mixamorigLeftHandThumb3': [0.8, 0.0, 0.6], 'mixamorigLeftHandThumb4': [0.9, 0.0, 0.4], 'mixamorigLeftHandThumb4_End': [0.9, 0.0, 0.4],
                'mixamorigLeftHandIndex1': [1.0, 0.0, 0.2], 'mixamorigLeftHandIndex2': [1.0, 0.0, 0.2], 'mixamorigLeftHandIndex3': [1.0, 0.0, 0.2], 'mixamorigLeftHandIndex4': [1.0, 0.0, 0.2], 'mixamorigLeftHandIndex4_End': [1.0, 0.0, 0.2],
                'mixamorigLeftHandMiddle1': [1.0, 0.0, 0.0], 'mixamorigLeftHandMiddle2': [1.0, 0.0, 0.0], 'mixamorigLeftHandMiddle3': [1.0, 0.0, 0.0], 'mixamorigLeftHandMiddle4': [1.0, 0.0, 0.0], 'mixamorigLeftHandMiddle4_End': [1.0, 0.0, 0.0],
                'mixamorigLeftHandRing1': [1.0, 0.0, -0.2], 'mixamorigLeftHandRing2': [1.0, 0.0, -0.2], 'mixamorigLeftHandRing3': [1.0, 0.0, -0.2], 'mixamorigLeftHandRing4': [1.0, 0.0, -0.2], 'mixamorigLeftHandRing4_End': [1.0, 0.0, -0.2],
                'mixamorigLeftHandPinky1': [0.9, 0.0, -0.4], 'mixamorigLeftHandPinky2': [0.9, 0.0, -0.4], 'mixamorigLeftHandPinky3': [0.9, 0.0, -0.4], 'mixamorigLeftHandPinky4': [0.9, 0.0, -0.4], 'mixamorigLeftHandPinky4_End': [0.9, 0.0, -0.4],

                # --- RIGHT ARM ---
                'mixamorigRightShoulder': [-1, 0, 0],
                'mixamorigRightArm': [-1, 0, 0],
                'mixamorigRightForeArm': [-1, 0, 0],
                'mixamorigRightHand': [-1, 0, 0],

                # --- RIGHT HAND FINGERS ---
                'mixamorigRightHandThumb1': [-0.5, 0.2, 0.8], 'mixamorigRightHandThumb2': [-0.7, 0.1, 0.7], 'mixamorigRightHandThumb3': [-0.8, 0.0, 0.6], 'mixamorigRightHandThumb4': [-0.9, 0.0, 0.4], 'mixamorigRightHandThumb4_End': [-0.9, 0.0, 0.4],
                'mixamorigRightHandIndex1': [-1.0, 0.0, 0.2], 'mixamorigRightHandIndex2': [-1.0, 0.0, 0.2], 'mixamorigRightHandIndex3': [-1.0, 0.0, 0.2], 'mixamorigRightHandIndex4': [-1.0, 0.0, 0.2], 'mixamorigRightHandIndex4_End': [-1.0, 0.0, 0.2],
                'mixamorigRightHandMiddle1': [-1.0, 0.0, 0.0], 'mixamorigRightHandMiddle2': [-1.0, 0.0, 0.0], 'mixamorigRightHandMiddle3': [-1.0, 0.0, 0.0], 'mixamorigRightHandMiddle4': [-1.0, 0.0, 0.0], 'mixamorigRightHandMiddle4_End': [-1.0, 0.0, 0.0],
                'mixamorigRightHandRing1': [-1.0, 0.0, -0.2], 'mixamorigRightHandRing2': [-1.0, 0.0, -0.2], 'mixamorigRightHandRing3': [-1.0, 0.0, -0.2], 'mixamorigRightHandRing4': [-1.0, 0.0, -0.2], 'mixamorigRightHandRing4_End': [-1.0, 0.0, -0.2],
                'mixamorigRightHandPinky1': [-0.9, 0.0, -0.4], 'mixamorigRightHandPinky2': [-0.9, 0.0, -0.4], 'mixamorigRightHandPinky3': [-0.9, 0.0, -0.4], 'mixamorigRightHandPinky4': [-0.9, 0.0, -0.4], 'mixamorigRightHandPinky4_End': [-0.9, 0.0, -0.4],

                # --- LEGS  ---
                # 'mixamorigLeftUpLeg': [0.5, -1, 0],    # Increased outward splay
                # 'mixamorigLeftLeg': [0, -1, 0],
                # 'mixamorigLeftFoot': [0, 0, 1],
                # 'mixamorigLeftToeBase': [0, 0, 1],
                # 'mixamorigLeftToe_End': [0, 0, 1],
                # 'mixamorigRightUpLeg': [-0.5, -1, 0],   # Increased outward splay
                # 'mixamorigRightLeg': [0, -1, 0],
                # 'mixamorigRightFoot': [0, 0, 1],
                # 'mixamorigRightToeBase': [0, 0, 1],
                # 'mixamorigRightToe_End': [0, 0, 1],
            }
            
            z_up_directions = {}
            for name, vec_list in y_up_directions.items():
                z_up_directions[name] = _swizzle_y_up_to_z_up(vec_list)

            # --- Step 3: Normalize the final Z-up vectors ---
            final_directions = {}
            for name, vec_list in z_up_directions.items():
                vec = np.array(vec_list, dtype=float)
                norm = np.linalg.norm(vec)
                if norm > 1e-9:
                    final_directions[name] = vec / norm
                else:
                    final_directions[name] = vec
                    
            return final_directions


    def get_bvh_header(self, poses_3d):
        initial_offset = self.get_initial_offset(poses_3d)

        nodes = {}
        for joint in self.keypoint2index:
            is_root = joint == self.root
            is_end_site = 'EndSite' in joint
            nodes[joint] = bvh_helper.BvhNode(
                name=joint,
                offset=initial_offset[joint],
                rotation_order='zxy' if not is_end_site else '',
                is_root=is_root,
                is_end_site=is_end_site,
            )
        for joint, children in self.children.items():
            nodes[joint].children = [nodes[child] for child in children]
            for child in children:
                nodes[child].parent = nodes[joint]

        header = bvh_helper.BvhHeader(root=nodes[self.root], nodes=nodes)
        return header


    def pose2euler(self, pose, header):
        channel = []
        quats = {} # To store global quaternions for each joint
        eulers = {} # To store local euler angles for each joint

        exclude_legs = [ "left_hip",
        "right_hip",
        "left_knee",
         "right_knee",
         "left_ankle",
         "right_ankle",
         "left_heel",
         "right_heel",
         "left_foot_index",
         "right_foot_index"]

        # Use a stack for depth-first traversal of the skeleton
        stack = [header.root]
        
        # A shortcut to the keypoint-to-index map
        index = self.keypoint2index

        while stack:
            node = stack.pop()
            joint = node.name # e.g., 'hips', 'spine', 'left_shoulder'

            mainmap = self.map_to_mediapipe.get(joint, joint)
            
            # This check is crucial: only process joints defined in our skeleton header
            if mainmap not in index or mainmap in exclude_legs:
                continue

            joint_idx = index[mainmap]
            joint =  mainmap

            if node.is_root:
                # The root's channel data includes its 3D position
                # Assuming the root position is stored at the 'hips' index
                position_data = np.array([0.0, 0.0, 0.0], dtype=np.float32)                # print(f"Hips position data: {type(pose[index['hips']])}")
                channel.extend(position_data)

            # These variables will define the joint's local coordinate system
            x_dir, y_dir, z_dir, order = None, None, None, None

            # --- Define axes based on joint name ---

            if joint == 'hips':
                # X-axis: from right hip to left hip
                # Z-axis: from hips up to the spine
                x_dir = pose[index['left_hip']] - pose[index['right_hip']]
                z_dir = pose[index['spine']] - pose[joint_idx]
                order = 'zyx'

            # elif joint in ['right_hip', 'right_knee']:
            #     # Z-axis: along the bone (e.g., from knee to ankle)
            #     # X-axis: parallel to the hips' x-axis for stability
            #     child_name = node.children[0].name
            #     child_name  = self.map_to_mediapipe.get(child_name, child_name)
            #     child_idx = index[child_name]
            #     x_dir = pose[index['left_hip']] - pose[index['right_hip']]
            #     z_dir = pose[joint_idx] - pose[child_idx]
            #     order = 'zyx'   
                
            # elif joint in ['left_hip', 'left_knee']:
            #     child_name = node.children[0].name
            #     child_name  = self.map_to_mediapipe.get(child_name, child_name)

            #     child_idx = index[child_name]
            #     x_dir = pose[index['left_hip']] - pose[index['right_hip']]
            #     z_dir = pose[joint_idx] - pose[child_idx]
            #     order = 'zyx'

            elif joint == 'spine':
                # Z-axis points up to the next spine segment
                x_dir = pose[index['left_hip']] - pose[index['right_hip']]
                z_dir = pose[index['spine1']] - pose[joint_idx]
                order = 'zyx'
                
            elif joint == 'spine1':
                # Z-axis points up to the next spine segment
                x_dir = pose[index['left_hip']] - pose[index['right_hip']]
                z_dir = pose[index['spine2']] - pose[joint_idx]
                order = 'zyx'

            elif joint == 'spine2':
                # X-axis is now defined by the shoulders
                # Z-axis points up to the neck
                x_dir = pose[index['left_shoulder']] - pose[index['right_shoulder']]
                z_dir = pose[index['neck']] - pose[joint_idx]
                order = 'zyx'

            elif joint == 'neck':
                # Y-axis: up from the chest (spine2)
                # Z-axis: forward towards the nose
                x_dir = pose[index['left_shoulder']] - pose[index['right_shoulder']]
                z_dir = pose[index['nose']] - pose[joint_idx]
                order = 'zyx' # Common order for necks
            # elif joint == 'neck':
            #     # Y-axis: up from the chest (spine2)
            #     # Z-axis: forward towards the nose
            #     y_dir = pose[joint_idx] - pose[index['spine2']]
            #     z_dir = pose[index['nose']] - pose[joint_idx]
            #     order = 'zxy' # Common order for necks

            elif joint == 'left_shoulder':
                # X-axis: along the upper arm bone (shoulder to elbow)
                # Y-axis: defines the plane of arm swing (vector from elbow to wrist)
                x_dir = pose[index['left_arm']] - pose[joint_idx] # 'left_arm' is the elbow
                y_dir = (pose[index['left_arm']] - pose[index['left_forearm']]) # 'left_forearm' is the wrist
                order = 'xzy' # Allows for twist and swing

            elif joint == 'left_arm': # This is the Left Elbow
                # X-axis: along the forearm bone (elbow to wrist)
                # Y-axis: direction from shoulder, maintains alignment  
                x_dir = pose[index['left_forearm']] - pose[joint_idx]
                y_dir = (pose[joint_idx] - pose[index['left_shoulder']])   
                order = 'xzy'

            elif joint == 'right_shoulder':
                # Similar to left, but directions are flipped for a right-hand coord system
                x_dir = pose[joint_idx] - pose[index['right_arm']]
                y_dir = (pose[index['right_arm']] - pose[index['right_forearm']])
                order = 'xzy'

            elif joint == 'right_arm': # This is the Right Elbow
                x_dir = pose[joint_idx] - pose[index['right_forearm']]
                y_dir = (pose[joint_idx] - pose[index['right_shoulder']])
                order = 'xzy'
            if order:
                dcm = math3d.dcm_from_axis(x_dir, y_dir, z_dir, order)
                quats[joint] = math3d.dcm2quat(dcm)
            else:
                quats[joint] = quats[self.parent[joint]].copy()
            
            local_quat = quats[joint].copy()
            if node.parent:
                local_quat = math3d.quat_divide(
                    q=quats[joint], r=quats[node.parent.name]
                )
            
            euler = math3d.quat2euler(
                q=local_quat, order=node.rotation_order
            )
            euler = np.rad2deg(euler)
            eulers[joint] = euler
            channel.extend(euler)

            for child in node.children[::-1]:
                if not child.is_end_site:
                    stack.append(child)

        return channel

    def pose2euler_SmartBody(self, pose, header):
        channel = []
        quats = {}
        eulers = {}
        stack = [header.root]
        while stack:
            node = stack.pop()
            joint = node.name
            joint_idx = self.keypoint2index[joint]

            if node.is_root:
                channel.extend(pose[joint_idx])

            index = self.keypoint2index
            order = None
            if joint == 'Hip':
                x_dir = pose[index['LeftHip']] - pose[index['RightHip']]
                z_dir = None
                y_dir = pose[index['Spine']] - pose[joint_idx]
                order = 'yzx'
            elif joint in ['RightHip', 'RightKnee']:
                child_idx = self.keypoint2index[node.children[0].name]
                x_dir = pose[index['Hip']] - pose[index['RightHip']]
                z_dir = None
                y_dir = pose[joint_idx] - pose[child_idx]
                order = 'yzx'
            elif joint in ['LeftHip', 'LeftKnee']:
                child_idx = self.keypoint2index[node.children[0].name]
                x_dir = pose[index['LeftHip']] - pose[index['Hip']]
                z_dir = None
                y_dir = pose[joint_idx] - pose[child_idx]
                order = 'yzx'
            elif joint == 'Spine':
                x_dir = pose[index['LeftHip']] - pose[index['RightHip']]
                z_dir = None
                y_dir = pose[index['Thorax']] - pose[joint_idx]
                order = 'yzx'
            elif joint == 'Thorax':
                x_dir = pose[index['LeftShoulder']] - \
                        pose[index['RightShoulder']]
                z_dir = None
                y_dir = pose[joint_idx] - pose[index['Spine']]
                order = 'yzx'
            elif joint == 'Neck':
                x_dir = None
                z_dir = pose[index['Thorax']] - pose[joint_idx]
                y_dir = pose[index['HeadEndSite']] - pose[index['Thorax']]
                order = 'yxz'
            elif joint == 'LeftShoulder':
                x_dir = pose[index['LeftElbow']] - pose[joint_idx]
                z_dir = pose[index['LeftElbow']] - pose[index['LeftWrist']]
                y_dir = None
                order = 'xyz'
            elif joint == 'LeftElbow':
                x_dir = pose[index['LeftWrist']] - pose[joint_idx]
                z_dir = pose[joint_idx] - pose[index['LeftShoulder']]
                y_dir = None
                order = 'xyz'
            elif joint == 'RightShoulder':
                x_dir = pose[joint_idx] - pose[index['RightElbow']]
                z_dir = pose[index['RightElbow']] - pose[index['RightWrist']]
                y_dir = None
                order = 'xyz'
            elif joint == 'RightElbow':
                x_dir = pose[joint_idx] - pose[index['RightWrist']]
                z_dir = pose[joint_idx] - pose[index['RightShoulder']]
                y_dir = None
                order = 'xyz'
            if order:
                dcm = math3d.dcm_from_axis(x_dir, y_dir, z_dir, order)
                quats[joint] = math3d.dcm2quat(dcm)
            else:
                quats[joint] = quats[self.parent[joint]].copy()

            local_quat = quats[joint].copy()
            if node.parent:
                local_quat = math3d.quat_divide(
                    q=quats[joint], r=quats[node.parent.name]
                )

            euler = math3d.quat2euler(
                q=local_quat, order=node.rotation_order
            )
            euler = np.rad2deg(euler)
            eulers[joint] = euler
            channel.extend(euler)

            for child in node.children[::-1]:
                if not child.is_end_site:
                    stack.append(child)

        return channel

    def pose2euler_SmartBody_Modify(self, pose, header):
        channel = []
        quats = {}
        eulers = {}
        stack = [header.root]
        while stack:
            node = stack.pop()
            joint = node.name
            joint_idx = self.keypoint2index[joint]

            if node.is_root:
                channel.extend(pose[joint_idx])

            index = self.keypoint2index
            order = None
            if joint == 'Hips':
                x_dir = pose[index['LeftUpLeg']] - pose[index['RightUpLeg']]
                z_dir = None
                y_dir = pose[index['Spine']] - pose[joint_idx]
                order = 'yzx'
            elif joint in ['LeftUpLeg', 'RightLeg']:
                child_idx = self.keypoint2index[node.children[0].name]
                x_dir = pose[index['Hips']] - pose[index['RightUpLeg']]
                z_dir = None
                y_dir = pose[joint_idx] - pose[child_idx]
                order = 'yzx'
            elif joint in ['LeftUpLeg', 'LeftLeg']:
                child_idx = self.keypoint2index[node.children[0].name]
                x_dir = pose[index['LeftUpLeg']] - pose[index['Hips']]
                z_dir = None
                y_dir = pose[joint_idx] - pose[child_idx]
                order = 'yzx'
            elif joint == 'Spine':
                x_dir = pose[index['LeftUpLeg']] - pose[index['RightUpLeg']]
                z_dir = None
                y_dir = pose[index['Spine3']] - pose[joint_idx]
                order = 'yzx'
            elif joint == 'Spine3':
                x_dir = pose[index['LeftArm']] - \
                        pose[index['RightArm']]
                z_dir = None
                y_dir = pose[joint_idx] - pose[index['Spine']]
                order = 'yzx'
            elif joint == 'Neck':
                x_dir = None
                z_dir = pose[index['Spine3']] - pose[joint_idx]
                y_dir = pose[index['Head']] - pose[index['Spine3']]
                order = 'yxz'
            elif joint == 'LeftArm':
                x_dir = pose[index['LeftForeArm']] - pose[joint_idx]
                z_dir = pose[index['LeftForeArm']] - pose[index['LeftHand']]
                y_dir = None
                order = 'xyz'
            elif joint == 'LeftForeArm':
                x_dir = pose[index['LeftHand']] - pose[joint_idx]
                z_dir = pose[joint_idx] - pose[index['LeftArm']]
                y_dir = None
                order = 'xyz'
            elif joint == 'RightArm':
                x_dir = pose[joint_idx] - pose[index['RightForeArm']]
                z_dir = pose[index['RightForeArm']] - pose[index['RightHand']]
                y_dir = None
                order = 'xyz'
            elif joint == 'RightForeArm':
                x_dir = pose[joint_idx] - pose[index['RightHand']]
                z_dir = pose[joint_idx] - pose[index['RightArm']]
                y_dir = None
                order = 'xyz'
            if order:
                dcm = math3d.dcm_from_axis(x_dir, y_dir, z_dir, order)
                quats[joint] = math3d.dcm2quat(dcm)
            else:
                quats[joint] = quats[self.parent[joint]].copy()

            local_quat = quats[joint].copy()
            if node.parent:
                local_quat = math3d.quat_divide(
                    q=quats[joint], r=quats[node.parent.name]
                )

            euler = math3d.quat2euler(
                q=local_quat, order=node.rotation_order
            )
            euler = np.rad2deg(euler)
            eulers[joint] = euler
            channel.extend(euler)

            for child in node.children[::-1]:
                if not child.is_end_site:
                    stack.append(child)

        return channel



    def poses2bvh(self, poses_3d, header=None, output_file=None):
        if not header:
            header = self.header

        channels = []
        for frame, pose in enumerate(poses_3d):
            channels.append(self.pose2euler(pose, header))
            #channels.append(self.pose2euler_SmartBody(pose, header))
            #channels.append(self.pose2euler_SmartBody_Modify(pose, header))

        if output_file:
            bvh_helper.write_bvh(output_file, header, channels)
        
        return channels, header
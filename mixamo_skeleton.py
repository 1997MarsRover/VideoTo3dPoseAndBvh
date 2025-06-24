import json
import numpy as np
from typing import Dict, List, Any, Optional
import math3d
from scipy.spatial.transform import Rotation as R


class BvhNode(object):
    def __init__(
        self, name: str, offset: np.ndarray, rotation_order: str,
        children: List['BvhNode'] = None, parent: 'BvhNode' = None, 
        is_root: bool = False, is_end_site: bool = False
    ):
        if not is_end_site and rotation_order not in ['xyz', 'xzy', 'yxz', 'yzx', 'zxy', 'zyx']:
            raise ValueError(f"Rotation order '{rotation_order}' is invalid.")
        self.name = name
        self.offset = offset
        self.rotation_order = rotation_order
        # Ensure children is a new list if not provided
        self.children = children if children is not None else []
        self.parent = parent
        self.is_root = is_root
        self.is_end_site = is_end_site

class BvhHeader(object):
    def __init__(self, root: BvhNode, nodes: Dict[str, BvhNode]):
        self.root = root
        self.nodes = nodes

class MixamoSkeleton:
    """
    Represents a full Mixamo skeleton, including fingers and toes, and can
    generate a stable BVH hierarchy by analyzing an entire animation's worth of
    landmark data.
    """

    def __init__(self):
        """
        Initializes the skeleton by defining its structure, T-pose orientation,
        and mapping to MediaPipe landmark source keys.
        """
        self.joint_names: List[str] = []
        self.hierarchy: Dict[str, str] = {}
        self.children_of: Dict[str, List[str]] = {}
        self._parse_bvh_tree_structure(self._get_bvh_tree_structure())
        
        self.root_name: str = "mixamorigHips"
        self.initial_directions: Dict[str, np.ndarray] = self._get_initial_directions()
        self.map_to_mediapipe: Dict[str, str] = self._get_mediapipe_map()
        self.header =  self._build_bvh_header(self.initial_directions, 100.0)

        self.root = 'mixamorigHips'


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
        self.index2keypoint = {v: k for k, v in self.keypoint2index.items()}



    def load_poses_from_json(self, file_path: str, keypoint2index: Dict[str, int]) -> List[np.ndarray]:
        """
        Reads a JSON file containing per-frame 3D pose landmarks and converts them
        into a list of NumPy arrays.

        If a keypoint is not found for a given frame, its coordinates will be [0, 0, 0].

        Args:
            file_path (str): The path to the input JSON file.
            keypoint2index (Dict[str, int]): A dictionary mapping keypoint names
                                            (e.g., 'nose') to their desired integer index.

        Returns:
            List[np.ndarray]: A list where each element is a NumPy array representing
                            a single frame's pose. Each array has a shape of
                            (num_keypoints, 3) for X, Y, Z coordinates.
        """
        try:
            with open(file_path, 'r') as f:
                all_frames_data = json.load(f)
        except FileNotFoundError:
            print(f"Error: The file '{file_path}' was not found.")
            return []
        except json.JSONDecodeError:
            print(f"Error: The file '{file_path}' is not a valid JSON file.")
            return []

        all_poses = []
        num_keypoints = len(keypoint2index)

        # Iterate through each frame in the JSON list
        for idx, frame_data in enumerate(all_frames_data):
            landmarks = frame_data.get("person_absolute_world_landmarks")

            if not landmarks:
                print(f"Warning: Frame {frame_data.get('frame_number', 'N/A')} is missing 'person_absolute_world_landmarks'. Skipping.")
                continue
            
            # --- THIS IS THE MODIFIED LINE ---
            # Initialize with zeros. Missing keypoints will remain as [0, 0, 0].
            pose_3d = np.zeros((num_keypoints, 3), dtype=np.float32)

            # Populate the array using the keypoint2index map
            for keypoint_name, coords in landmarks.items():
                if keypoint_name in keypoint2index:
                    index =  keypoint2index[keypoint_name]
                    
                    x = coords.get('x')
                    y = coords.get('z')
                    z = coords.get('y')

                    if x is not None and y is not None and z is not None:
                        pose_3d[index] = [x, y, z]
            if idx == 0 or idx == 25:
                print(f"Frame {idx} pose: {len(pose_3d)}")  

            all_poses.append(pose_3d)

        return all_poses

    def build_stable_hierarchy_from_json(
        self,
        json_file_path: str,
        scale_factor: float = 100.0
    ) -> Optional[str]:
        """
        Main public method. Loads a landmark JSON file, calculates stable T-pose
        offsets, and returns a complete BVH HIERARCHY string.
        """
        print(f"Loading landmark data from '{json_file_path}'...")
        try:
            with open(json_file_path, 'r') as f:
                all_frames_data = json.load(f)
        except Exception as e:
            print(f"Error: Could not read or parse JSON file. {e}")
            return None
        
        if not all_frames_data:
            print("Error: JSON file contains no data.")
            return None

        avg_lengths = self._calculate_average_bone_lengths(all_frames_data)
        initial_offsets = self._calculate_initial_offsets(avg_lengths)
        hierarchy_string = self._build_hierarchy_string(initial_offsets, scale_factor)
        
        return hierarchy_string

    def _calculate_initial_offsets(self, average_lengths: Dict[str, float]) -> Dict[str, np.ndarray]:
        """Calculates T-pose offset vectors from average lengths and initial directions."""
        print("Calculating T-pose offsets from average bone lengths...")
        initial_offsets = {}
        for joint_name, direction in self.initial_directions.items():
            length = average_lengths.get(joint_name, 0.0)
            norm_direction = direction / (np.linalg.norm(direction) + 1e-9)
            initial_offsets[joint_name] = norm_direction * length
            
        initial_offsets[self.root_name] = np.array([0, 0, 0])
        return initial_offsets

    def _build_hierarchy_string(self, initial_offsets: Dict[str, np.ndarray], scale_factor: float) -> str:
        """Recursively builds the final HIERARCHY string from the T-pose offsets."""
        print("Constructing final BVH HIERARCHY string...")

        def _build_node_string(joint_name: str, indent_level: int) -> str:
            indent = "  " * indent_level
            children = self.children_of.get(joint_name, [])
            
            is_end_site = "End" in joint_name
            
            if indent_level == 0:
                node_str = f"HIERARCHY\n{indent}ROOT {joint_name}\n"
            elif is_end_site:
                node_str = f"{indent}End Site\n"
            else:
                node_str = f"{indent}JOINT {joint_name}\n"
            
            node_str += f"{indent}{{\n"
            
            offset = initial_offsets.get(joint_name, np.array([0.0, 0.0, 0.0])) * scale_factor
            offset_str = " ".join(map(lambda x: f"{x:.6f}", offset))
            node_str += f"{indent}  OFFSET {offset_str}\n"
            
            if not is_end_site:
                channels = "Xposition Yposition Zposition Zrotation Xrotation Yrotation" if indent_level == 0 else "Zrotation Xrotation Yrotation"
                node_str += f"{indent}  CHANNELS {len(channels.split())} {channels}\n"
                for child in children:
                    node_str += _build_node_string(child, indent_level + 1)
            
            node_str += f"{indent}}}\n"
            return node_str

        return _build_node_string(self.root_name, 0)
    
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

    def _parse_bvh_tree_structure(self, bvh_tree: Dict):
        """Recursively parses the user's tree structure to populate internal hierarchy maps."""
        def _recursive_parse(node_dict, parent_name):
            for name, content in node_dict.items():
                self.joint_names.append(name)
                self.children_of[name] = []
                self.hierarchy[name] = parent_name
                if parent_name != "WORLD":
                    self.children_of[parent_name].append(name)
                
                children = content.get("children", [])
                for child_dict in children:
                    _recursive_parse(child_dict, name)
        
        _recursive_parse(bvh_tree, "WORLD")
        
    def _get_bvh_tree_structure(self) -> Dict:
        """Defines the Mixamo skeleton hierarchy. From your formatting_utils.py."""
        # --- FIX 1: Verified this structure is correct. Fingers are children of Hand. ---
        return {
            "mixamorigHips": {
                "children": [
                    {"mixamorigSpine": {"children": [
                        {"mixamorigSpine1": {"children": [
                            {"mixamorigSpine2": {"children": [
                                {"mixamorigNeck": {"children": [
                                    {"mixamorigHead": {"children": [
                                        {"mixamorigHeadTop_End": {}}
                                    ]}}
                                ]}},
                                {"mixamorigRightShoulder": {"children": [
                                    {"mixamorigRightArm": {"children": [
                                        {"mixamorigRightForeArm": {"children": [
                                            {"mixamorigRightHand": {"children": [
                                                {"mixamorigRightHandThumb1": {"children": [{"mixamorigRightHandThumb2": {"children": [{"mixamorigRightHandThumb3": {"children": [{"mixamorigRightHandThumb4": {"children": [{"mixamorigRightHandThumb4_End": {}}]}}]}}]}}]}},
                                                {"mixamorigRightHandIndex1": {"children": [{"mixamorigRightHandIndex2": {"children": [{"mixamorigRightHandIndex3": {"children": [{"mixamorigRightHandIndex4": {"children": [{"mixamorigRightHandIndex4_End": {}}]}}]}}]}}]}},
                                                {"mixamorigRightHandMiddle1": {"children": [{"mixamorigRightHandMiddle2": {"children": [{"mixamorigRightHandMiddle3": {"children": [{"mixamorigRightHandMiddle4": {"children": [{"mixamorigRightHandMiddle4_End": {}}]}}]}}]}}]}},
                                                {"mixamorigRightHandRing1": {"children": [{"mixamorigRightHandRing2": {"children": [{"mixamorigRightHandRing3": {"children": [{"mixamorigRightHandRing4": {"children": [{"mixamorigRightHandRing4_End": {}}]}}]}}]}}]}},
                                                {"mixamorigRightHandPinky1": {"children": [{"mixamorigRightHandPinky2": {"children": [{"mixamorigRightHandPinky3": {"children": [{"mixamorigRightHandPinky4": {"children": [{"mixamorigRightHandPinky4_End": {}}]}}]}}]}}]}},
                                            ]}}
                                        ]}}
                                    ]}}
                                ]}},
                                {"mixamorigLeftShoulder": {"children": [
                                    {"mixamorigLeftArm": {"children": [
                                        {"mixamorigLeftForeArm": {"children": [
                                            {"mixamorigLeftHand": {"children": [
                                                {"mixamorigLeftHandThumb1": {"children": [{"mixamorigLeftHandThumb2": {"children": [{"mixamorigLeftHandThumb3": {"children": [{"mixamorigLeftHandThumb4": {"children": [{"mixamorigLeftHandThumb4_End": {}}]}}]}}]}}]}},
                                                {"mixamorigLeftHandIndex1": {"children": [{"mixamorigLeftHandIndex2": {"children": [{"mixamorigLeftHandIndex3": {"children": [{"mixamorigLeftHandIndex4": {"children": [{"mixamorigLeftHandIndex4_End": {}}]}}]}}]}}]}},
                                                {"mixamorigLeftHandMiddle1": {"children": [{"mixamorigLeftHandMiddle2": {"children": [{"mixamorigLeftHandMiddle3": {"children": [{"mixamorigLeftHandMiddle4": {"children": [{"mixamorigLeftHandMiddle4_End": {}}]}}]}}]}}]}},
                                                {"mixamorigLeftHandRing1": {"children": [{"mixamorigLeftHandRing2": {"children": [{"mixamorigLeftHandRing3": {"children": [{"mixamorigLeftHandRing4": {"children": [{"mixamorigLeftHandRing4_End": {}}]}}]}}]}}]}},
                                                {"mixamorigLeftHandPinky1": {"children": [{"mixamorigLeftHandPinky2": {"children": [{"mixamorigLeftHandPinky3": {"children": [{"mixamorigLeftHandPinky4": {"children": [{"mixamorigLeftHandPinky4_End": {}}]}}]}}]}}]}},
                                            ]}}
                                        ]}}
                                    ]}}
                                ]}},
                            ]}}
                        ]}}
                    ]}},
                    # {"mixamorigRightUpLeg": {"children": [{"mixamorigRightLeg": {"children": [{"mixamorigRightFoot": {"children": [{"mixamorigRightToeBase": {"children": [{"mixamorigRightToe_End": {}}]}}]}}]}}]}},
                    # {"mixamorigLeftUpLeg": {"children": [{"mixamorigLeftLeg": {"children": [{"mixamorigLeftFoot": {"children": [{"mixamorigLeftToeBase": {"children": [{"mixamorigLeftToe_End": {}}]}}]}}]}}]}},
                ]
            }
        }
    
    def pose2euler(self, pose: np.ndarray) -> list:
        """
        Converts a 3D pose (joint positions) into a list of Euler angles
        for a hierarchical skeleton.

        Args:
            pose (np.ndarray): A NumPy array of shape (num_keypoints, 3)
                               containing the 3D world coordinates for each joint.

        Returns:
            list: A flat list containing the root position and Euler angles
                  for all joints, in the order required by the skeleton hierarchy.
        """
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
        stack = [self.header.root]
        
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
                
            # --- Main Rotation Calculation ---
            
            if order:
                # If axes are defined, calculate the global rotation matrix and quaternion
                dcm = math3d.dcm_from_axis(x_dir, y_dir, z_dir, order)
                quats[joint] = math3d.dcm2quat(dcm)
            else:
                # For joints without specific logic (e.g., ankles, wrists),
                # they initially inherit the rotation of their parent.
                # This is a safe fallback for leaf nodes of the main skeleton.
                parent_joint = self.parent.get(joint)
                if parent_joint and parent_joint in quats:
                    quats[joint] = quats[parent_joint].copy()
                else:
                    # Failsafe for a joint with no parent in the dict
                    quats[joint] = np.array([1.0, 0.0, 0.0, 0.0]) # No rotation

            # Calculate the LOCAL rotation relative to the parent
            local_quat = quats[joint].copy()
            if node.parent and node.parent.name in quats:
                # local_rotation = global_rotation * inverse(parent_global_rotation)
                parent_quat = quats[node.parent.name]
                local_quat = math3d.quat_divide(q=quats[joint], r=parent_quat)

            # Convert the local quaternion to Euler angles in the specified order
            euler = math3d.quat2euler(q=local_quat, order=node.rotation_order)
            euler = np.rad2deg(euler)
            
            eulers[joint] = euler
            channel.extend(euler)

            # Add children to the stack to be processed next (in reverse order)
            for child in reversed(node.children):
                if not child.is_end_site:
                    stack.append(child)

        return channel

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



    def _calculate_average_bone_lengths(self, all_frames_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Analyzes all frames to find the average length of each bone.
        This version now assumes all virtual points, including hand_base,
        are pre-calculated and present in the input data.
        """
        print("Calculating average bone lengths over all frames...")
        bone_lengths_over_time = {joint: [] for joint in self.joint_names}

        for frame_data in all_frames_data:
            # The landmark dict from your JSON is at this key
            all_landmarks_this_frame = frame_data.get("person_absolute_world_landmarks", {})
            
            # --- REMOVED: Virtual hand base calculation is now done in formatting_utils.py ---
            # The code now directly reads the pre-calculated virtual points.
            
            joint_positions = {}
            for mixamo_joint, mp_key in self.map_to_mediapipe.items():
                if mp_key in all_landmarks_this_frame:
                    pos_data = all_landmarks_this_frame[mp_key]
                    # Ensure data is valid before converting to numpy array
                    if all(k in pos_data for k in ['x', 'y', 'z']):
                        joint_positions[mixamo_joint] = np.array([pos_data['x'], pos_data['y'], pos_data['z']])
            
            # Calculate the length of each bone for this frame
            for joint_name, parent_name in self.hierarchy.items():
                if parent_name != "WORLD" and joint_name in joint_positions and parent_name in joint_positions:
                    vector = joint_positions[joint_name] - joint_positions[parent_name]
                    length = np.linalg.norm(vector)
                    if length > 1e-5: # Avoid adding zero-length bones
                        bone_lengths_over_time[joint_name].append(length)

        # Average the collected bone lengths, using symmetry for stability
        average_bone_lengths = {}
        processed = set()
        for joint_name in self.joint_names:
            if joint_name in processed: continue
            
            if "Left" in joint_name:
                right_joint_name = joint_name.replace("Left", "Right")
                combined_lengths = bone_lengths_over_time.get(joint_name, []) + bone_lengths_over_time.get(right_joint_name, [])
                avg_len = np.mean(combined_lengths) if combined_lengths else 0.0
                average_bone_lengths[joint_name] = avg_len
                average_bone_lengths[right_joint_name] = avg_len
                processed.add(joint_name); processed.add(right_joint_name)
            else:
                lengths = bone_lengths_over_time.get(joint_name, [])
                average_bone_lengths[joint_name] = np.mean(lengths) if lengths else 0.0
                processed.add(joint_name)
                
        return average_bone_lengths


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
    


def normalize(v):
    norm = np.linalg.norm(v)
    return v / norm if norm > 0 else v


def compute_joint_rotation(child_pos, parent_pos, side_hint=None):
    y_axis = normalize(child_pos - parent_pos)
    if side_hint is None:
        side_hint = np.array([0, 1, 0]) if abs(y_axis[1]) < 0.9 else np.array([1, 0, 0])
    x_axis = normalize(np.cross(side_hint, y_axis))
    z_axis = normalize(np.cross(x_axis, y_axis))
    x_axis = normalize(np.cross(y_axis, z_axis))
    return np.stack([x_axis, y_axis, z_axis], axis=1)


class MotionCalculator:
    def __init__(self, skeleton, all_frames_data: List[Dict[str, Any]]):
        self.skeleton = skeleton
        self.all_frames_data = all_frames_data
        self.mediapipe_map = self.skeleton.map_to_mediapipe
        self.hierarchy = self.skeleton.hierarchy
        self.children_of = self.skeleton.children_of
        self.root_name = self.skeleton.root_name

    def _get_landmark_positions_for_frame(self, frame_data: Dict[str, Any]) -> Dict[str, np.ndarray]:
        positions = {}
        landmarks = frame_data.get("person_absolute_world_landmarks", {})
        for joint, mp_key in self.mediapipe_map.items():
            pos = landmarks.get(mp_key)
            if pos and all(k in pos for k in ['x', 'y', 'z']):
                positions[joint] = np.array([pos['x'], pos['z'], pos['y']])  
        return positions

    def calculate_motion_string(self, channels,scale_factor: float = 100.0) -> str:
        num_frames = len(self.all_frames_data)
        motion_lines = []

        for i, frame_data in enumerate(channels):
            print(f"\rProcessing frame {i+1}/{num_frames}", end="")
            # positions = self._get_landmark_positions_for_frame(frame_data)
            # global_rot = {}
            frame_channels = frame_data
            # stack = [(self.root_name, None)]

            # while stack:
            #     joint, parent = stack.pop()

            #     try:
            #         children = self.children_of.get(joint, [])
            #         if children:
            #             child = children[0]
            #             child_pos = positions[child]
            #             parent_pos = positions[joint]
            #             if 'Right' in joint:
            #                 side_hint = (positions[children[-1]] - parent_pos) * -1
            #             elif 'Left' in joint:
            #                 side_hint = (positions[children[-1]] - parent_pos)
            #             else:
            #                 side_hint = None
            #             dcm = compute_joint_rotation(child_pos, parent_pos, side_hint)
            #             global_rot[joint] = R.from_matrix(dcm)
            #         else:
            #             global_rot[joint] = global_rot.get(parent, R.identity())
            #     except Exception:
            #         global_rot[joint] = global_rot.get(parent, R.identity())

            #     if parent is None:
            #         pos = positions[joint] * scale_factor
            #         frame_channels.extend([pos[0], pos[1], pos[2]])
            #         local_rot = global_rot[joint]
            #     else:
            #         local_rot = global_rot[parent].inv() * global_rot[joint]

            #     eul = local_rot.as_euler('zyx', degrees=True)
            #     frame_channels.extend(eul.tolist())

            #     for child in reversed(self.children_of.get(joint, [])):
            #         if "End" not in child:
            #             stack.append((child, joint))

            motion_lines.append(" ".join(f"{v:.6f}" for v in frame_channels))

        print("\nMotion calculation complete.")
        header = f"MOTION\nFrames: {num_frames}\nFrame Time: {1/30.0:.8f}\n"
        return header + "\n".join(motion_lines)

if __name__ == '__main__':
     input_json_file = "video_absolute_world_landmarks_per_frame.json" 
     output_bvh_file = "final_output.bvh"
     bvh_scale_factor = 100.0

     print("--- Starting Full BVH Generation ---")
    
    #   --- Part 1: Generate the HIERARCHY (same as before) ---
     print("--- Step 1: Building Stable HIERARCHY ---")
     mixamo_rig = MixamoSkeleton()
     keypoint_dict = mixamo_rig.keypoint2index

     poses_3D = mixamo_rig.load_poses_from_json(input_json_file, keypoint_dict)
     all_channels = []

     for pose in poses_3D:
         channels = mixamo_rig.pose2euler(pose)
         all_channels.append(channels)
     
     print(all_channels)
    #  print(poses_3D)
     bvh_hierarchy_string = mixamo_rig.build_stable_hierarchy_from_json(
         json_file_path=input_json_file,
         scale_factor=bvh_scale_factor
     )
     if not bvh_hierarchy_string:
         print("\n Hierarchy generation failed. Aborting.")
     else:
         print("\n Hierarchy built successfully.")

        

    #      # --- Part 2: Calculate the MOTION data ---
         print("\n--- Step 2: Calculating MOTION Data ---")
        #   Load the data again (or pass it from the hierarchy builder)
         with open(input_json_file, 'r') as f:
             all_frames_data = json.load(f)

         motion_builder = MotionCalculator(mixamo_rig, all_frames_data)
         bvh_motion_string = motion_builder.calculate_motion_string(all_channels,scale_factor=bvh_scale_factor)

        #   --- Part 3: Combine and Save ---
         print("\n--- Step 3: Combining and Saving BVH File ---")
         full_bvh_content = bvh_hierarchy_string + bvh_motion_string
         with open(output_bvh_file, 'w') as f:
             f.write(full_bvh_content)
        
         print(f"\n Successfully saved complete BVH file to: {output_bvh_file}")
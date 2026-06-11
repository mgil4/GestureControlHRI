"""
gesture_system.launch.py

Launches all nodes with the correct configuration for the selected mode.
''ros2 launch gesture_hri_pkg gesture_system.launch.py'' command initialises the entire pipeline.

USE:
- Mode 1: Robot command control (default)
ros2 launch gesture_hri_pkg gesture_system.launch.py mode:=robot
- Mode 2A: Sign-to-speech (TTS)
ros2 launch gesture_hri_pkg gesture_system.launch.py mode:=text
- Mode 2B: Sign-to-robot conversation (LLM + TTS)
ros2 launch gesture_hri_pkg gesture_system.launch.py mode:=llm backend:=ollama ollama_model:=llama3
"""

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo, OpaqueFunction
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

def _build_nodes(context, *args, **kwargs):
   mode        = LaunchConfiguration("mode").perform(context)
   image_topic = LaunchConfiguration("image_topic").perform(context)
   pkg         = LaunchConfiguration("pkg").perform(context)
   show_preview = LaunchConfiguration("show_preview").perform(context)
 
   config_dir = PathJoinSubstitution([
       FindPackageShare("gesture_hri_pkg"),
       "config"
   ])
 
   perception_yaml = PathJoinSubstitution([config_dir, "hand_gesture_recognition.yaml"])
   decision_yaml   = PathJoinSubstitution([config_dir, "gesture_decision.yaml"])
   controller_yaml = PathJoinSubstitution([config_dir, "robot_controller.yaml"])
 
   log = LogInfo(msg=f"[gesture_system] mode='{mode}'")
 
   perception_node = Node(
       package=pkg,
       executable="hand_gesture_recognition",
       name="hand_gesture_recognition",
       output="screen",
       parameters=[perception_yaml,
           {
               "image_topic": image_topic,
               "show_preview": (show_preview.lower() == "true"),               
           }
       ],
   )
 
   decision_node = Node(
       package=pkg,
       executable="gesture_decision",
       name="gesture_decision",
       output="screen",
       parameters=[decision_yaml,
           {
               "mode": mode
           }
       ],
   )
 
   controller_node = Node(
       package=pkg,
       executable="robot_controller",
       name="robot_controller",
       output="screen",
       parameters=[controller_yaml,
           {
               "mode": mode
           }
       ],
   )
 
   nodes = [
       log,
       perception_node,
       decision_node,
       controller_node,
   ]
 
   if mode == "llm":
       llm_yaml = PathJoinSubstitution([config_dir, "llm_client.yaml"])
       llm_node = Node(
           package=pkg,
           executable="llm_client",
           name="llm_client",
           output="screen",
           parameters=[llm_yaml],
       )
       nodes.append(llm_node)
 
   return nodes
 
def generate_launch_description():
   args = [
       DeclareLaunchArgument(
           "mode",
           default_value="robot",
           description="robot | text | llm"
       ),
       DeclareLaunchArgument(
           "image_topic",
           default_value="/head_front_camera/color/image_raw"
       ),
       DeclareLaunchArgument(
           "pkg",
           default_value="gesture_hri_pkg"
       ),
       DeclareLaunchArgument(
           "show_preview",
           default_value="gesture_hri_pkg"
       ),
   ]
 
   return LaunchDescription(args + [OpaqueFunction(function=_build_nodes)])


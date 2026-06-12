#!/usr/bin/env python3
"""
robot_controller.py

Single node that handles all three interaction modes by reading the ``mode``
parameter at startup and activating only the relevant internal handler.
No separate node files are needed per mode.

SUBSCRIBED TOPIC:
  /gesture/action  (std_msgs/String)
    JSON produced by gesture_decision.py. The ``mode`` field in each
    message determines which internal handler processes it.

MODE 1: robot   (Robot Command Mode)
Executes TIAGo Pro movement commands derived from numeric gestures.

MODE 2A: text (Sign-to-Speech)
Receives completed sentences (action == "SPEAK") and sends them to the
TIAGo Pro TTS action server.

MODE 2B: llm (Sign-to-Robot Conversation)
Receives completed sentences (action == "LLM_QUERY"), forwards them to a
language model via the ``llm_client.py`` helper node on the
/llm/query (String) topic / /llm/response (String) topic, waits for the
response, then sends it to the TTS backend.

All three modes share the same TTS backend code.
"""

import json
import threading
import time

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from std_msgs.msg import String, Header
from geometry_msgs.msg import Twist
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration

# PAL TTS action
try:
    from tts_msgs.action import TTS as TTSAction
    _HAS_PAL_TTS = True
except ImportError:
    _HAS_PAL_TTS = False


# play_motion2 action
try:
    from play_motion2_msgs.action import PlayMotion2
    _HAS_PLAY_MOTION = True
except ImportError:
    _HAS_PLAY_MOTION = False

# Actions: motion parameters table
_ROBOT_ACTION_TABLE: dict[str, dict] = {
    "STOP":              {"twist": (0.0, 0.0, 0.0), "motion": None},
    "MOVE_FORWARD":      {"twist": (1.0, 0.0, 0.0), "motion": None},
    "MOVE_BACK":         {"twist": (-1.0, 0.0, 0.0), "motion": None},
    "TURN_LEFT":         {"twist": (0.0, 0.0, 1.0), "motion": None},
    "TURN_RIGHT":        {"twist": (0.0, 0.0, -1.0), "motion": None},
    "MOVE_FORWARD_FAST": {"twist": (1.0, 0.0, 0.0), "motion": None, "fast": True},
    "SPIN":              {"twist": (0.0, 0.0, 1.0), "motion": None, "spin": True},
    "LOOK_UP":           {"twist": None,             "motion": None, "head": "up"},
    "LOOK_DOWN":         {"twist": None,             "motion": None, "head": "down"},
    "HOME_POSE":         {"twist": None,             "motion": "home"},
    "EMERGENCY_STOP":    {"twist": (0.0, 0.0, 0.0), "motion": "home"},
    "IDLE":              {"twist": None,             "motion": None},
}

# Node
class RobotControllerNode(Node):

    def __init__(self):
        super().__init__("robot_controller")

        # Parameters
        self._declare_params()
        self._read_params()

        # Shared ROS interfaces
        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        self._action_sub = self.create_subscription(
            String, "/gesture/action", self._action_callback, qos
        )

        # Mode-specific setup
        if self._mode == "robot":
            self._setup_robot_mode()
        elif self._mode in ("text", "llm"):
            self._setup_text_mode()
            if self._mode == "llm":
                self._setup_llm_mode()

        self.get_logger().info(
            f"Robot Controller ready mode={self._mode}\n"
            + self._mode_summary()
        )

    # Parameter helpers
    def _declare_params(self):
        self.declare_parameter("mode",                 "robot")
        self.declare_parameter("linear_speed",         0.3)
        self.declare_parameter("angular_speed",        0.5)
        self.declare_parameter("fast_multiplier",      1.8)
        self.declare_parameter("cmd_vel_duration",     1.5)
        self.declare_parameter("cmd_vel_rate_hz",      10.0)
        self.declare_parameter("head_pan_default",     0.0)
        # BUG FIX: sign convention in declare_parameter now matches robot_controller.yaml.
        # YAML sets head_tilt_up=0.4 and head_tilt_down=-0.4 (TIAGo joint convention).
        # The old code defaults (-0.4 / 0.4) were opposite, causing LOOK_UP to tilt down
        # and vice versa when the YAML was NOT loaded.  Now both sources agree.
        self.declare_parameter("head_tilt_up",         0.4)
        self.declare_parameter("head_tilt_down",      -0.4)
        self.declare_parameter("head_move_duration",   1.0)
        self.declare_parameter("tts_backend",          "auto")
        self.declare_parameter("llm_response_timeout", 30.0)

    def _read_params(self):
        self._mode               = self.get_parameter("mode").value
        self._linear_speed       = self.get_parameter("linear_speed").value
        self._angular_speed      = self.get_parameter("angular_speed").value
        self._fast_multiplier    = self.get_parameter("fast_multiplier").value
        self._cmd_vel_duration   = self.get_parameter("cmd_vel_duration").value
        self._cmd_vel_rate_hz    = self.get_parameter("cmd_vel_rate_hz").value
        self._head_pan_default   = self.get_parameter("head_pan_default").value
        self._head_tilt_up       = self.get_parameter("head_tilt_up").value
        self._head_tilt_down     = self.get_parameter("head_tilt_down").value
        self._head_move_duration = self.get_parameter("head_move_duration").value
        self._tts_backend        = self.get_parameter("tts_backend").value
        self._llm_timeout        = self.get_parameter("llm_response_timeout").value

    def _mode_summary(self) -> str:
        if self._mode == "robot":
            return (
                f"  linear_speed: {self._linear_speed} m/s\n"
                f"  angular_speed: {self._angular_speed} rad/s\n"
                f"  cmd_vel_dur: {self._cmd_vel_duration} s\n"
                f"  play_motion2: {'available' if _HAS_PLAY_MOTION else 'NOT FOUND: motions disabled'}"
            )
        return (
            f"  tts_backend: {self._tts_backend}\n"
            f"  pal_tts: {'available' if _HAS_PAL_TTS else 'NOT FOUND: using fallback'}"
        )

    # MODE 1  –  Robot setup
    def _setup_robot_mode(self):
        self._cmd_vel_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self._head_pub    = self.create_publisher(
            JointTrajectory, "/head_controller/joint_trajectory", 10
        )
        if _HAS_PLAY_MOTION:
            self._play_motion_client = ActionClient(self, PlayMotion2, "/play_motion2")
        else:
            self._play_motion_client = None
            self.get_logger().warn(
                "play_motion2_msgs not installed: predefined motions (wave, home) disabled"
            )
        self._motion_lock   = threading.Lock()
        self._twist_cancel  = threading.Event()

    # MODE 2A / 2B: Text / LLM setup
    def _setup_text_mode(self):
        self._active_tts = self._resolve_tts_backend()
        self.get_logger().info(f"[TTS] Backend resolved: {self._active_tts}")

        if _HAS_PAL_TTS and self._active_tts == "pal":
            self._say_client = ActionClient(self, TTSAction, "/tts_engine/tts")
            self.get_logger().info("[TTS] PAL action client created -> /tts_engine/tts")
        else:
            self._say_client = None
            self.get_logger().error(
                "[TTS] PAL TTS unavailable. Install tts_msgs or check your ROS environment."
            )

    def _setup_llm_mode(self):
        self._llm_query_pub = self.create_publisher(String, "/llm/query", 10)
        self._llm_response_sub = self.create_subscription(
            String, "/llm/response", self._llm_response_callback, 10
        )
        # _llm_response stores the latest reply so the timeout callback can check it.
        self._llm_response: str = ""
        # One-shot timer reference; None when no query is in flight.
        self._llm_timeout_timer = None

    def _resolve_tts_backend(self) -> str:
        if self._tts_backend == "pal":
            if _HAS_PAL_TTS:
                return "pal"
            self.get_logger().warn(
                "tts_backend=pal but tts_msgs not installed: falling back to gtts"
            )
            return "gtts"
        if self._tts_backend == "gtts":
            return "gtts"
        return "pal" if _HAS_PAL_TTS else "gtts"

    # Main callback
    def _action_callback(self, msg: String):
        try:
            data = json.loads(msg.data)
            self.get_logger().info(f"RAW action received: {msg.data}")
        except json.JSONDecodeError as e:
            self.get_logger().error(f"Bad JSON on /gesture/action: {e}")
            return

        action     = data.get("action", "IDLE")
        mode       = data.get("mode",   self._mode)
        confidence = data.get("confidence", 0.0)

        self.get_logger().debug(
            f"Received action={action}  mode={mode}  conf={confidence:.2f}"
        )

        if mode == "robot":
            self._handle_robot_action(action, data)
        elif mode == "text" and action == "SPEAK":
            text = data.get("text", "").strip()
            if text:
                self.get_logger().info(f"[TTS] Speaking: '{text}'")
                threading.Thread(
                    target=self._speak, args=(text,), daemon=True
                ).start()
        elif mode == "llm" and action == "LLM_QUERY":
            text = data.get("text", "").strip()
            if text:
                self._handle_llm_query(text)

    # MODE 1: Robot action handlers
    def _handle_robot_action(self, action: str, data: dict):
        if action not in _ROBOT_ACTION_TABLE:
            self.get_logger().warn(f"Unknown robot action: '{action}'")
            return

        spec        = _ROBOT_ACTION_TABLE[action]
        twist_scale = spec.get("twist")
        motion_name = spec.get("motion")
        head_cmd    = spec.get("head")
        is_fast     = spec.get("fast",  False)
        is_spin     = spec.get("spin",  False)

        self.get_logger().info(
            f"\n                                                  "
            f"[ROBOT] Executing action:      {action}\n"
        )

        if twist_scale == (0.0, 0.0, 0.0):
            self._publish_stop()

        if twist_scale is not None and twist_scale != (0.0, 0.0, 0.0):
            self._twist_cancel.set()
            lx, ly, az = twist_scale
            lin      = self._linear_speed  * (self._fast_multiplier if is_fast else 1.0)
            ang      = self._angular_speed * (self._fast_multiplier if is_fast else 1.0)
            duration = self._cmd_vel_duration * (2.0 if is_spin else 1.0)
            threading.Thread(
                target=self._publish_twist_burst,
                args=(lx * lin, ly * lin, az * ang, duration),
                daemon=True,
            ).start()

        if head_cmd is not None:
            tilt = self._head_tilt_up if head_cmd == "up" else self._head_tilt_down
            self._publish_head_goal(pan=self._head_pan_default, tilt=tilt)

        if motion_name is not None:
            self._send_play_motion(motion_name)

    def _publish_stop(self):
        self._cmd_vel_pub.publish(Twist())
        self.get_logger().info("[ROBOT] Base stopped")

    def _publish_twist_burst(self, lx: float, ly: float, az: float, duration: float):
        with self._motion_lock:
            self._twist_cancel.clear()
            twist           = Twist()
            twist.linear.x  = lx
            twist.linear.y  = ly
            twist.angular.z = az

            rate_sec = 1.0 / max(self._cmd_vel_rate_hz, 1.0)
            steps    = int(duration / rate_sec)

            for _ in range(steps):
                if self._twist_cancel.is_set():
                    break
                self._cmd_vel_pub.publish(twist)
                time.sleep(rate_sec)

            self._publish_stop()

    def _publish_head_goal(self, pan: float, tilt: float):
        traj             = JointTrajectory()
        traj.joint_names = ["head_1_joint", "head_2_joint"]

        point            = JointTrajectoryPoint()
        point.positions  = [pan, tilt]
        point.velocities = [0.0, 0.0]

        secs  = int(self._head_move_duration)
        nsecs = int((self._head_move_duration - secs) * 1e9)
        point.time_from_start = Duration(sec=secs, nanosec=nsecs)

        traj.points = [point]
        self._head_pub.publish(traj)
        self.get_logger().info(f"[ROBOT] Head: pan={pan:.2f} tilt={tilt:.2f}")

    def _send_play_motion(self, motion_name: str):
        if not _HAS_PLAY_MOTION or self._play_motion_client is None:
            self.get_logger().warn(
                f"play_motion2 unavailable: cannot execute motion '{motion_name}'"
            )
            return

        if not self._play_motion_client.wait_for_server(timeout_sec=2.0):
            self.get_logger().warn(
                f"play_motion2 server not ready: skipping motion '{motion_name}'"
            )
            return

        goal               = PlayMotion2.Goal()
        goal.motion_name   = motion_name
        goal.skip_planning = False

        self.get_logger().info(f"[ROBOT] play_motion2: '{motion_name}'")
        self._play_motion_client.send_goal_async(
            goal,
            feedback_callback=lambda fb: None,
        ).add_done_callback(
            lambda future: self._play_motion_done(future, motion_name)
        )

    def _play_motion_done(self, future, motion_name: str):
        try:
            handle = future.result()
            handle.get_result_async().add_done_callback(
                lambda res_future: self.get_logger().info(
                    f"[ROBOT] play_motion2 '{motion_name}' finished "
                    f"success={res_future.result().result.success}"
                )
            )
        except Exception as e:
            self.get_logger().error(f"[ROBOT] play_motion2 '{motion_name}' error: {e}")

    # TTS (shared by mode 2A and 2B)
    def _speak(self, text: str):
        if self._active_tts == "pal":
            self._speak_pal(text)
        else:
            self.get_logger().error("[TTS] No TTS available: active_tts is not 'pal'")

    def _speak_pal(self, text: str):
        """
        Send text to the PAL TTS action server (/tts_engine/tts).

        BUG FIX: the original implementation called spin_until_future_complete()
        from a background thread.  rclpy's single-threaded executor does not
        support being driven from two threads simultaneously, so the TTS goal
        future would silently never complete, leaving the node in a hung state.

        Fix: the call is now fully callback-driven.  _speak_pal() returns
        immediately after send_goal_async(); the result arrives via
        _tts_goal_accepted_callback → _tts_result_callback, both executed
        on the main executor thread with no risk of deadlock.

        NOTE: because _speak_pal now returns before speech finishes, callers
        that need to sequence actions after TTS should use the result callback
        rather than relying on _speak() blocking.
        """
        if self._say_client is None or not _HAS_PAL_TTS:
            self.get_logger().error(
                "[TTS] PAL client not initialised — tts_msgs not installed or "
                "_setup_text_mode was not called"
            )
            return

        if not self._say_client.wait_for_server(timeout_sec=3.0):
            self.get_logger().error(
                "[TTS] /tts_engine/tts action server not available after 3s"
            )
            return

        goal        = TTSAction.Goal()
        goal.input  = text
        goal.locale = "en_US"

        self.get_logger().info(f"[TTS/PAL] Sending goal: '{text}'")
        send_future = self._say_client.send_goal_async(goal)
        send_future.add_done_callback(self._tts_goal_accepted_callback)
        # Returns immediately; speech continues via callbacks on the executor thread.

    def _tts_goal_accepted_callback(self, future):
        """Called by the executor when the TTS server accepts or rejects the goal."""
        try:
            goal_handle = future.result()
        except Exception as e:
            self.get_logger().error(f"[TTS/PAL] Goal send failed: {e}")
            return

        if not goal_handle.accepted:
            self.get_logger().error("[TTS/PAL] Goal rejected by server")
            return

        self.get_logger().info("[TTS/PAL] Goal accepted, waiting for result...")
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self._tts_result_callback)

    def _tts_result_callback(self, future):
        """Called by the executor when TTS finishes speaking."""
        try:
            future.result()
            self.get_logger().info("[TTS/PAL] Speech completed")
        except Exception as e:
            self.get_logger().error(f"[TTS/PAL] Result error: {e}")

    # MODE 2B: LLM
    def _handle_llm_query(self, text: str):
        """
        Publishes the user's sentence to /llm/query and arms a timeout timer.
        The response arrives asynchronously via _llm_response_callback.

        BUG FIX: the original implementation used threading.Event.wait() to
        block a background thread until the LLM response arrived, then called
        spin_until_future_complete() inside _speak_pal() from that same thread.
        This caused a deadlock because spin_until_future_complete() tries to
        drive the executor from a non-executor thread while the real executor
        is already spinning — TTS goal callbacks never fired.

        Fix: _handle_llm_query() now returns immediately after publishing the
        query.  The response callback (_llm_response_callback) is called on
        the executor thread and directly calls _speak() in a short-lived
        daemon thread (only to avoid blocking the executor during
        wait_for_server).  A ROS timer replaces the threading.Event timeout.
        """
        self._llm_response = ""

        # Cancel any previously armed timeout from a prior query.
        if self._llm_timeout_timer is not None:
            self._llm_timeout_timer.cancel()
            self._llm_timeout_timer = None

        msg      = String()
        msg.data = text
        self._llm_query_pub.publish(msg)
        self.get_logger().info(f"[LLM] Query published: '{text}'")

        # Arm a one-shot timeout timer on the ROS executor — safe from any thread.
        self._llm_timeout_timer = self.create_timer(
            self._llm_timeout, self._llm_timeout_callback
        )

    def _llm_timeout_callback(self):
        """Fires on the executor thread if no LLM response arrives in time."""
        if self._llm_timeout_timer is not None:
            self._llm_timeout_timer.cancel()
            self._llm_timeout_timer = None

        if not self._llm_response:
            self.get_logger().warn(
                f"[LLM] No response received within {self._llm_timeout}s"
            )
            threading.Thread(
                target=self._speak,
                args=("I did not receive a response. Please try again.",),
                daemon=True,
            ).start()

    def _llm_response_callback(self, msg: String):
        """
        Receives the LLM reply from llm_client.py on the executor thread.
        Cancels the timeout timer and speaks the response.

        If the response is empty (model produced nothing even after retry),
        cancel the timer immediately and speak the fallback rather than
        silently returning and leaving the 30s timeout ticking.
        """
        response = msg.data.strip()

        # Always cancel the timer as soon as any message arrives.
        if self._llm_timeout_timer is not None:
            self._llm_timeout_timer.cancel()
            self._llm_timeout_timer = None

        if not response:
            self.get_logger().warn("[LLM] Received empty response from llm_client")
            threading.Thread(
                target=self._speak,
                args=("I am sorry, I could not generate a response. Please try again.",),
                daemon=True,
            ).start()
            return

        self._llm_response = response  # mark as answered

        self.get_logger().info(f"[LLM] Response received, speaking...")
        # Short-lived daemon thread: avoids blocking the executor during
        # wait_for_server inside _speak_pal, while _speak_pal itself is now
        # fully async so no further blocking occurs after that call returns.
        threading.Thread(target=self._speak, args=(response,), daemon=True).start()

    # Cleanup
    def destroy_node(self):
        self.get_logger().info("Shutting down Robot Controller node")
        if self._mode == "robot":
            try:
                self._publish_stop()
            except Exception:
                pass
        if self._llm_timeout_timer is not None:
            try:
                self._llm_timeout_timer.cancel()
            except Exception:
                pass
        super().destroy_node()


# Entry point
def main(args=None):
    rclpy.init(args=args)
    node = RobotControllerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

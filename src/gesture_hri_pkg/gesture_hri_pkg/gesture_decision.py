#!/usr/bin/env python3
"""
gesture_decision.py

Central reasoning layer between the perception node and the execution node.
Its behaviour is entirely controlled by the ``mode`` ROS 2 parameter; NO
code changes are needed to switch between experimental conditions.

SUBSCRIBED TOPIC:
  /gesture/result  (std_msgs/String)  from hand_gesture_recognition.py
    JSON schema (produced by the updated perception node):
    {
      "label":      "A",          # majority-voted label
      "confidence": 0.87,         # aggregated confidence
      "num_hands":  1,
      "hands":      [{"label": "A", "confidence": 0.92, "hand_index": 0}],
      "timestamp":  1234567890.1
    }

PUBLISHED TOPICS:
  /gesture/action        (std_msgs/String)  confirmed commands / sentences
  /gesture/action_verbose (std_msgs/String) every-frame debug stream

MODES:
MODE 1  –  robot   (Robot Command Mode)
  Only numeric gesture labels (e.g. "0".."9") are considered valid.
  Each number maps to a predefined robot action.
  Alphabetic gestures are silently ignored.
  Two-hand combos are supported via COMBO_MAP.
  Published payload:
    { "mode": "robot", "action": "MOVE_FORWARD", "gesture": "1",
      "confidence": 0.91, "trigger": "single", "held_frames": 5 }

MODE 2A  –  text   (Sign-to-Speech Translation Mode)
  Only alphabetic gesture labels (A-Z) are processed.
  The node maintains an internal text buffer.
  Special control gestures:
    START_GESTURE  →  clears the buffer, marks sentence start
    SPACE_GESTURE  →  appends a space character
    END_GESTURE    →  publishes the completed sentence, clears buffer
    DELETE_GESTURE →  removes the last character (backspace)
  Published payload when sentence is completed:
    { "mode": "text", "action": "SPEAK",
      "text": "hello world", "confidence": 0.88 }
  Intermediate buffer state is published on /gesture/action_verbose so
  you can display it on a screen without firing TTS prematurely.

MODE 2B  –  llm    (Sign-to-Robot Conversation Mode)
  Identical sentence-building logic to Mode 2A.
  The only difference is the published action tag:
    { "mode": "llm", "action": "LLM_QUERY",
      "text": "hello world", "confidence": 0.88 }
  The robot_controller node uses this to route to the LLM instead of TTS.
"""

import json
import time
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from std_msgs.msg import String

# MODE 1: Robot gesture
# Helpers
def _is_alpha(label: str) -> bool:
    """Returns True if the label is a single uppercase/lowercase letter A-Z."""
    return len(label) == 1 and label.upper() in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

def _is_numeric(label: str) -> bool:
    """Returns True if the label is a single digit 0-9."""
    return label in "0123456789"

# Node
class GestureDecisionNode(Node):
    """
    Mode-driven decision node.
    A single reusable file whose entire behaviour is controlled by the
    ``mode`` ROS 2 parameter.  No separate node files are needed for each
    interaction mode.
    """
    VALID_MODES = ("robot", "text", "llm")

    def __init__(self):
        super().__init__("gesture_decision")

        # Declare & read parameters
        self._declare_params()
        self._read_params()
        self._validate_mode()

        # Shared debounce / timing state
        self._candidate_label  : str | None = None
        self._candidate_frames : int        = 0
        self._active_action    : str | None = None
        self._last_fire_time   : float      = 0.0
        self._last_hand_time   : float      = time.time()
        self._idle_sent        : bool       = False
        self._waiting_for_release : bool    = False
        
        # Letter predicted shown but not yet committed to buffer 
        self._pending_letter     : str | None = None
        self._pending_confidence : float      = 0.0

        # Mode 2A/2B  –  text buffer state
        self._text_buffer      : list[str]  = []   # accumulated characters
        self._sentence_started : bool       = False
        self._buffer_conf_sum  : float      = 0.0  # for mean confidence
        self._buffer_conf_n    : int        = 0
        
        self._llm_busy = False
        self._llm_reply_sub = self.create_subscription(String, "/llm/response", self._llm_reply_callback, 10)

        # QoS
        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        # ROS interfaces
        self._gesture_sub = self.create_subscription(
            String, "/gesture/result", self._gesture_callback, qos
        )
        self._action_pub   = self.create_publisher(String, "/gesture/action",         10)
        self._verbose_pub  = self.create_publisher(String, "/gesture/action_verbose", 10)

        self.get_logger().info(
            f"GestureDecisionNode ready\n"
            f"  mode             : {self._mode}\n"
            f"  debounce_frames  : {self._debounce_frames}\n"
            f"  cooldown_sec     : {self._cooldown_sec}\n"
            f"  idle_timeout_sec : {self._idle_timeout_sec}\n"
            f"  min_confidence   : {self._min_confidence}\n"
            + (
                f"  control gestures : START={self._start_gesture} "
                f"SPACE={self._space_gesture} "
                f"END={self._end_gesture} "
                f"DELETE={self._delete_gesture}"
                f"COMMIT={self._commit_gesture}"
                if self._mode in ("text", "llm") else
                f"  robot map        : {list(self._robot_gesture_map.keys())}"
            )
        )

    # Parameter helpers
    def _declare_params(self):
        self.declare_parameter("mode",                "robot")
        self.declare_parameter("debounce_frames",     3)
        self.declare_parameter("cooldown_sec",        1.5)
        self.declare_parameter("idle_timeout_sec",    2.0)
        self.declare_parameter("publish_every_frame", False)
        self.declare_parameter("min_confidence",      0.60)
        # Text / LLM control gesture labels
        self.declare_parameter("start_gesture",       "5")
        self.declare_parameter("space_gesture",       "3")
        self.declare_parameter("end_gesture",         "7")
        self.declare_parameter("delete_gesture",      "8")
        self.declare_parameter("commit_gesture",      "5")
        # Robot gesture map (flat parallel lists)
        self.declare_parameter("gesture_labels", ["0","a","s","m","n","o","e","t"])
        self.declare_parameter("gesture_actions", ["STOP","STOP","STOP","STOP","STOP","STOP","STOP","STOP"])
        self.declare_parameter("gesture_priority", [True,True,True,True,True,True,True,True])
        # Robot combo map (flat parallel lists)
        self.declare_parameter("combo_labels",   ["5+1","5+d","5+2","5+v","5+u","5+3","5+4","5+b","3+1","3+d","3+2","3+v","3+u"])
        self.declare_parameter("combo_actions",  ["MOVE_FORWARD","MOVE_FORWARD","MOVE_BACK","MOVE_BACK","MOVE_BACK","TURN_LEFT","TURN_RIGHT","TURN_RIGHT","LOOK_UP","LOOK_UP","LOOK_DOWN","LOOK_DOWN","LOOK_DOWN"])
        self.declare_parameter("combo_priority", [False,False,False,False,False,False,False,False,False,False,False,False,False])

    def _read_params(self):
        self._mode                = self.get_parameter("mode").value
        self._debounce_frames     = self.get_parameter("debounce_frames").value
        self._cooldown_sec        = self.get_parameter("cooldown_sec").value
        self._idle_timeout_sec    = self.get_parameter("idle_timeout_sec").value
        self._publish_every_frame = self.get_parameter("publish_every_frame").value
        self._min_confidence      = self.get_parameter("min_confidence").value
        self._start_gesture       = self.get_parameter("start_gesture").value
        self._space_gesture       = self.get_parameter("space_gesture").value
        self._end_gesture         = self.get_parameter("end_gesture").value
        self._delete_gesture      = self.get_parameter("delete_gesture").value
        self._commit_gesture      = self.get_parameter("commit_gesture").value
        
        # Rebuild gesture map from parallel lists
        labels   = self.get_parameter("gesture_labels").value
        actions  = self.get_parameter("gesture_actions").value
        priority = self.get_parameter("gesture_priority").value
        self._robot_gesture_map = {
        	lbl: {"action": act, "priority": pri}
        	for lbl, act, pri in zip(labels, actions, priority)
    	}
        # Rebuild combo map
        c_labels   = self.get_parameter("combo_labels").value
        c_actions  = self.get_parameter("combo_actions").value
        c_priority = self.get_parameter("combo_priority").value
        self._robot_combo_map = {
                frozenset(lbl.split("+")): {"action": act, "priority": pri}
                for lbl, act, pri in zip(c_labels, c_actions, c_priority)
        }

    def _validate_mode(self):
        if self._mode not in self.VALID_MODES:
            self.get_logger().fatal(
                f"Invalid mode '{self._mode}'. Must be one of {self.VALID_MODES}"
            )
            raise ValueError(f"Invalid mode: {self._mode}")
            
    def _llm_reply_callback(self, msg: String):
        self._llm_busy = False
        self.get_logger().info("LLM reply received, gesture input re-enabled.")

    # In robot mode: alphabetic aliases ("b" is treated as "4")
    _ROBOT_ALIASES: dict[str, str] = {
        "b": "4",
        "d": "1",
        "f": "9",
        "o": "0",
        "v": "2",
        "w": "6",
    }

    # In text/llm mode: numeric aliases ("4" is treated as "b")
    _TEXT_ALIASES: dict[str, str] = {
        "4": "b",
        "1": "d",
        "9": "f",
        "0": "o",
        "2": "v",
        "6": "w",
    }

    # Main callback  (fires for every message on /gesture/result)
    def _gesture_callback(self, msg: String):
        try:
            data = json.loads(msg.data)
        except json.JSONDecodeError as e:
            self.get_logger().error(f"Bad JSON on /gesture/result: {e}")
            return

        # Extract top-level voted result from updated perception node
        label      = data.get("label", "")
        confidence = float(data.get("confidence", 0.0))
        num_hands  = int(data.get("num_hands", 0))
        hands      = data.get("hands", [])       # per-hand list, may be empty

        # Reject low-confidence frames early
        if confidence < self._min_confidence:
            label = ""
            
        # Normalize visually equivalent labels based on mode
        if label and self._mode in ("text", "llm"):
            _TEXT_ALIASES = {
                "4": "b", "1": "d", "9": "f",
                "0": "o", "2": "v", "6": "w",
            }
            label = _TEXT_ALIASES.get(label, label)
            # Also remap in hands list so verbose output is consistent
            for h in hands:
                h["label"] = _TEXT_ALIASES.get(h["label"], h["label"])

        # Normalize hands with aliases
        if self._mode == "robot":
            for h in hands:
                h["label"] = self._ROBOT_ALIASES.get(h["label"].lower(), h["label"])

        has_gesture = bool(label)

        # Track last hand presence (for idle timeout)
        if has_gesture:
            self.get_logger().info(f"Predicted label='{label}' conf={confidence:.2f}")
            self._last_hand_time = time.time()
            self._idle_sent = False

        # Route to mode handler
        if self._mode == "robot":
            self._handle_robot_mode(label, confidence, hands, has_gesture)
        else:
            # text and llm share the same sentence-building logic
            self._handle_text_mode(label, confidence, has_gesture, num_hands=num_hands, hands=hands)


    # MODE 1  –  Robot Command Mode
    def _handle_robot_mode(
        self,
        label: str,
        confidence: float,
        hands: list[dict],
        has_gesture: bool,
    ):
        """
        Processes gestures in robot command mode.
        Only numeric labels are considered valid inputs.
        Alphabetic labels are silently ignored.
        Two-hand combos are checked first (using the per-hand list).
        Debounce + cooldown logic applies to non-priority gestures.
        """
        # Try two-hand combo first
        resolved = None
        if len(hands) >= 2:
            label_set = frozenset(h["label"] for h in hands[:2])
            if label_set in self._robot_combo_map:
                mapping = self._robot_combo_map[label_set]
                conf    = max(h["confidence"] for h in hands[:2])
                resolved = {
                    "action":    mapping["action"],
                    "priority":  mapping["priority"],
                    "trigger":   "combo",
                    "gesture":   "+".join(sorted(label_set)),
                    "confidence": conf,
                }
                self.get_logger().info(f"Combo detected: {resolved['gesture']} {mapping['action']}") 


        # Fall back to single-hand numeric gesture
        #if resolved is None and has_gesture:
        #    if label in self._robot_gesture_map:
        #        mapping  = self._robot_gesture_map[label]
        #        resolved = {
        #            "action":    mapping["action"],
        #            "priority":  mapping["priority"],
        #            "trigger":   "single",
        #            "gesture":   label,
        #            "confidence": confidence,
        #        }

        # Verbose publish
        verbose = self._build_robot_payload(resolved, mode="robot")
        self._maybe_publish_verbose(verbose)

        # Debounce / cooldown / fire
        action_payload = None

        if resolved is None:
            # No valid gesture – reset debounce, check idle
            self._candidate_label  = None
            self._candidate_frames = 0
            action_payload = self._check_idle()
        else:
            action    = resolved["action"]
            priority  = resolved["priority"]

            if priority:
                # Fires immediately regardless of debounce or cooldown
                action_payload         = verbose
                self._candidate_label  = None
                self._candidate_frames = 0
                self._active_action    = action
                self._last_fire_time   = time.time()
            else:
                # Accumulate debounce counter
                if resolved["gesture"] == self._candidate_label:
                    self._candidate_frames += 1
                else:
                    self._candidate_label  = resolved["gesture"]
                    self._candidate_frames = 1

                if self._candidate_frames >= self._debounce_frames:
                    now       = time.time()
                    cool_ok   = (now - self._last_fire_time) >= self._cooldown_sec
                    new_action = action != self._active_action

                    if cool_ok or new_action:
                        action_payload       = verbose
                        self._active_action  = action
                        self._last_fire_time = now

        # Publish confirmed action
        if action_payload is not None:
            self._publish_action(action_payload)
            #self._llm_busy = True
        elif self._publish_every_frame:
            self._publish_action(verbose)

    # MODE 2A / 2B: Text / LLM (sentence building)
    def _handle_text_mode(self, label: str, confidence: float, has_gesture: bool, num_hands: int = 1, hands: list | None = None,):
        """
        A debounced letter is held as 'pending' and shown in the log.
        It is only appended to the buffer when the second hand
        simultaneously shows the commit_gesture.
        Control gestures (START, SPACE, DELETE, END) fire immediately
        on debounce, they do not require the commit hand.
        """
        if hands is None:
            hands = []
        
        if self._llm_busy:
            return
            
        if not has_gesture:
            self._candidate_label     = None
            self._candidate_frames    = 0
            self._waiting_for_release = False
            self._pending_letter      = None
            self._pending_confidence  = 0.0
            self._maybe_publish_idle()
            return

        is_control = label in (
            self._start_gesture,
            self._space_gesture,
            self._end_gesture,
            self._delete_gesture,
        )
        is_letter = _is_alpha(label)
        
        if not is_letter and not is_control:
            self._candidate_label  = None
            self._candidate_frames = 0
            return

        # Lockout until sign changes after a fire
        if self._waiting_for_release:
            if label == self._candidate_label:
                # Check if commit hand appeared while holding, commit pending
                if (self._pending_letter is not None
                        and num_hands >= 2
                        and any(h["label"] == self._commit_gesture
                                for h in hands)):
                    self._commit_pending_letter()
                return
            else:
                self._waiting_for_release = False
                self._candidate_frames    = 0
                self._pending_letter      = None
                self._pending_confidence  = 0.0

        # Debounce accumulation 
        if label == self._candidate_label:
            self._candidate_frames += 1
        else:
            self._candidate_label  = label
            self._candidate_frames = 1
            self._pending_letter   = None   # new sign cancels previous pending

        self._publish_buffer_state(label, confidence)

        if self._candidate_frames < self._debounce_frames:
            return

        # Gesture confirmed 
        self._waiting_for_release = True
        self._candidate_frames    = 0

        # Control gestures fire immediately without commit hand
        if label == self._start_gesture:
            self._text_buffer      = []
            self._sentence_started = True
            self._buffer_conf_sum  = 0.0
            self._buffer_conf_n    = 0
            self._pending_letter   = None
            self.get_logger().info(
                "\n                                                  "
                "Text buffer cleared. SENTENCE STARTED\n"
            )
            self._publish_buffer_state(label, confidence)
            return

        elif label == self._space_gesture:
            if self._sentence_started:
                self._text_buffer.append(" ")
                self._accumulate_confidence(confidence)
                self.get_logger().info(
                    f"\n                                                  "
                    f"SPACE inserted. Text: '{self._buffer_text()}'\n"
                )
            return

        elif label == self._delete_gesture:
            if self._text_buffer:
                self._text_buffer.pop()
                self.get_logger().info(
                    f"\n                                                  "
                    f"DELETE. Text: '{self._buffer_text()}'\n"
                )
            return

        elif label == self._end_gesture:
            sentence = self._buffer_text().strip()
            if sentence and self._sentence_started:
                avg_conf = (
                    self._buffer_conf_sum / self._buffer_conf_n
                    if self._buffer_conf_n > 0 else 0.0
                )
                action_tag = "LLM_QUERY" if self._mode == "llm" else "SPEAK"
                payload = {
                    "mode":       self._mode,
                    "action":     action_tag,
                    "text":       sentence,
                    "confidence": round(avg_conf, 4),
                }
                self._publish_action(payload)
                self._llm_busy = True
                self.get_logger().info(
                    "\n                                                  TEXT SENT.\n"
                )
            self._text_buffer      = []
            self._sentence_started = False
            self._buffer_conf_sum  = 0.0
            self._buffer_conf_n    = 0
            self._pending_letter   = None
            return

        # Letter confirmed, hold as pending, wait for commit hand 
        if is_letter:
            if not self._sentence_started:
                self.get_logger().warn(
                    "\n                                                  "
                    "START gesture needed. SHOW YOUR OPEN PALM.\n"
                )
                return

            self._pending_letter     = label.upper()
            self._pending_confidence = confidence
            self.get_logger().info(
                f"\n                                                  "
                f"Detecting: '{self._pending_letter}'. "
                f"Show commit sign (open palm) "
                f"with second hand to add the letter\n"
            )

            # If commit hand is already visible in this same frame, commit now
            if (num_hands >= 2 and any(h["label"] == self._commit_gesture for h in hands)):
                self._commit_pending_letter()
	
    def _commit_pending_letter(self):
        """Append the pending letter to the buffer and clear pending state."""
        if self._pending_letter is None:
            return
        self._text_buffer.append(self._pending_letter)
        self._accumulate_confidence(self._pending_confidence)
        self.get_logger().info(
            f"\n                                                  "
            f"ADDED '{self._pending_letter}'. "
            f"Text: '{self._buffer_text()}'\n")
        self._pending_letter     = None
        self._pending_confidence = 0.0

    # Text buffer helpers
    def _buffer_text(self) -> str:
        return "".join(self._text_buffer)

    def _accumulate_confidence(self, conf: float):
        self._buffer_conf_sum += conf
        self._buffer_conf_n   += 1

    def _publish_buffer_state(self, current_label: str, confidence: float):
        """Publish current buffer state on verbose topic (for live UI display)."""
        if self._verbose_pub.get_subscription_count() == 0:
            return
        payload = {
            "mode":            self._mode,
            "action":          "BUFFER_UPDATE",
            "buffer":          self._buffer_text(),
            "current_gesture": current_label,
            "confidence":      round(confidence, 4),
            "sentence_active": self._sentence_started,
        }
        self._raw_publish(self._verbose_pub, payload)

    # Idle detection (shared across modes)
    def _check_idle(self) -> dict | None:
        """Returns an IDLE payload if the timeout has elapsed, else None."""
        elapsed = time.time() - self._last_hand_time
        if elapsed >= self._idle_timeout_sec and not self._idle_sent:
            self._idle_sent    = True
            self._active_action = "IDLE"
            return self._idle_payload()
        return None

    def _maybe_publish_idle(self):
        payload = self._check_idle()
        if payload is not None:
            self._publish_action(payload)

    def _idle_payload(self) -> dict:
        return {
            "mode":       self._mode,
            "action":     "IDLE",
            "trigger":    "timeout",
            "gesture":    "None",
            "confidence": 0.0,
        }

    # Payload builders
    def _build_robot_payload(self, resolved: dict | None, mode: str) -> dict:
        held = self._candidate_frames if resolved else 0
        if resolved is None:
            return {
                "mode":        mode,
                "action":      self._active_action or "IDLE",
                "trigger":     "none",
                "gesture":     "None",
                "confidence":  0.0,
                "held_frames": 0,
            }
        return {
            "mode":        mode,
            "action":      resolved["action"],
            "trigger":     resolved["trigger"],
            "gesture":     resolved["gesture"],
            "confidence":  round(resolved["confidence"], 4),
            "held_frames": held,
        }

    # Publish helpers
    def _maybe_publish_verbose(self, payload: dict):
        if self._verbose_pub.get_subscription_count() > 0:
            self._raw_publish(self._verbose_pub, payload)

    def _publish_action(self, payload: dict):
        self._raw_publish(self._action_pub, payload)
        #self.get_logger().info(f"\nACTION: {json.dumps(payload)}\n")

    def _raw_publish(self, publisher, payload: dict):
        msg      = String()
        msg.data = json.dumps(payload)
        publisher.publish(msg)

# Entry point
def main(args=None):
    rclpy.init(args=args)
    node = GestureDecisionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

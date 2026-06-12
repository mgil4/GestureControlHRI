#!/usr/bin/env python3
"""
llm_client.py

Standalone node that sits between robot_controller.py and a language model.
Uses Ollama with Qwen3:0.6b (small and fast LLM).

SUBSCRIBED TOPIC
  /llm/query  (std_msgs/String)
    Plain text sentence from robot_controller.py (mode 2B).

PUBLISHED TOPICS
  /llm/response  (std_msgs/String)  Plain text response from the model
  /llm/status  (std_msgs/String)    JSON: {"state": "thinking"|"ready"|"error", "detail": "…"}

INSTALL
curl -fsSL https://ollama.com/install.sh | sh 
ollama pull qwen3:0.6b 
pip install ollama

"""

import json
import threading
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

# /no_think suppresses Qwen3's chain-of-thought <think> blocks.
# Without it the model prepends lengthy reasoning text that would
# be read aloud verbatim by the TTS engine.
_DEFAULT_SYSTEM_PROMPT = (
   "You are TIAGo, a friendly and helpful mobile robot that gives conversation. "
   "You will receive as input the acronym of an airport (3-letter IATA code)."
   "You should answer giving a list of things to visit at the city where it belongs, don't wait for an answer nor offer more help."
   "Avoid markdown formatting and emojis, respond in plain spoken English only, "
   "since your response will be read aloud by a text-to-speech engine."
)
 
class LLMClientNode(Node):
 
    def __init__(self):
        super().__init__("llm_client")

        self._declare_params()
        self._read_params()

        # ROS interfaces
        self._query_sub    = self.create_subscription(String, "/llm/query",    self._query_callback, 10)
        self._response_pub = self.create_publisher(  String, "/llm/response",  10)
        self._status_pub   = self.create_publisher(  String, "/llm/status",    10)

        # Prevent overlapping Ollama calls
        self._busy_lock = threading.Lock()

        # Deduplication: track the last query processed so the same string
        # arriving twice in a row does not fire a second LLM call.
        # Initialised to a sentinel that will never match a real query.
        self._last_query: str = ""

        self.get_logger().info(
            f"LLMClientNode ready\n"
            f"  model         : {self._ollama_model}\n"
            f"  ollama_host   : {self._ollama_host}\n"
            f"  max_tokens    : {self._max_tokens}\n"
            f"  temperature   : {self._temperature}"
        )

    def _declare_params(self):
        self.declare_parameter("ollama_host",   "http://localhost:11434")
        self.declare_parameter("ollama_model",  "qwen3:0.6b")
        self.declare_parameter("max_tokens",    200)
        self.declare_parameter("temperature",   0.7)
        # history_turns is declared so the YAML value is accepted without warnings,
        # but conversation history is not used in this version.
        self.declare_parameter("history_turns", 0)
        self.declare_parameter("system_prompt", _DEFAULT_SYSTEM_PROMPT)

    def _read_params(self):
        self._ollama_host   = self.get_parameter("ollama_host").value
        self._ollama_model  = self.get_parameter("ollama_model").value
        self._max_tokens    = self.get_parameter("max_tokens").value
        self._temperature   = self.get_parameter("temperature").value
        self._system_prompt = self.get_parameter("system_prompt").value

    def _query_callback(self, msg: String):
        """
        Receives a query string from robot_controller.py and spawns a
        background thread to call Ollama.

        BUG FIX: previously _last_query was always reset to "" before the
        comparison, making the dedup guard ineffective and preventing
        _last_query from ever storing the actual query text.
        """
        query = msg.data.strip()

        # Reject empty strings or exact repeats of the last processed query.
        if not query or query == self._last_query:
            return

        self._last_query = query  # store AFTER the guard, not before

        self.get_logger().info(
            f"\n                                                         "
            f"[LLM] QUERY RECEIVED: '{query}'\n"
        )

        # Run in background so the ROS executor is not blocked
        threading.Thread(target=self._process_query, args=(query,), daemon=True).start()

    def _process_query(self, query: str):
        with self._busy_lock:
            self._publish_status("thinking", query)
            t0 = time.time()
            try:
                messages      = self._build_messages(query)
                response_text = self._call_ollama(messages)
                elapsed       = time.time() - t0

                self.get_logger().info(
                    f"\n                                                         "
                    f"[LLM] RESPONSE ({elapsed:.1f}s): '{response_text}'"
                )
                self._publish_response(response_text)
                self._publish_status("ready", f"Done in {elapsed:.1f}s")

            except Exception as e:
                self.get_logger().error(f"[LLM] Error: {e}")
                self._publish_status("error", str(e))
                self._publish_response(
                    "I am sorry, I had trouble generating a response. Please try again."
                )

    def _build_messages(self, query: str) -> list[dict]:
        return [
            {"role": "system", "content": self._system_prompt},
            {"role": "user",   "content": query},
        ]

    def _call_ollama(self, messages: list[dict]) -> str:
        try:
            import ollama
        except ImportError:
            raise RuntimeError("ollama package not installed. Run: pip install ollama")

        client = ollama.Client(host=self._ollama_host)
        resp   = client.chat(
            model=self._ollama_model,
            messages=messages,
            options={
                "temperature": self._temperature,
                "num_predict": self._max_tokens,
            },
        )
        return resp["message"]["content"].strip()

    def _publish_response(self, text: str):
        msg      = String()
        msg.data = text
        self._response_pub.publish(msg)

    def _publish_status(self, state: str, detail: str = ""):
        msg      = String()
        msg.data = json.dumps({"state": state, "detail": detail})
        self._status_pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = LLMClientNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

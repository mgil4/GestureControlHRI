#!/usr/bin/env python3
"""
llm_client.py

Standalone node that sits between robot_controller.py and a language model.
Uses Ollama with Qwen3:0.6b (small and fast LLM).

SUBSCRIBED TOPIC
  /llm/query  (std_msgs/String)
    Plain text sentence from robot_controller.py (mode 2B).

PUBLISHED TOPIC
  /llm/response  (std_msgs/String)
    Plain text response from the language model.

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
from collections import deque

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

# /no_think suppresses Qwen3's chain-of-thought <think> blocks.
# Without it the model prepends lengthy reasoning text that would
# be read aloud verbatim by the TTS engine.
_DEFAULT_SYSTEM_PROMPT = (
   "/no_think "
   "You are TIAGo, a friendly and helpful mobile robot assistant. "
   "You are having a conversation with a person who is communicating "
   "with you using sign language. You should answer to their message."
   "Keep your responses concise (1-3 sentences), warm, and clear. "
   "Avoid markdown formatting, respond in plain spoken English only, "
   "since your response will be read aloud by a text-to-speech engine."
)
 
class LLMClientNode(Node):
 
   def __init__(self):
       super().__init__("llm_client")
 
       self._declare_params()
       self._read_params()
 
       # Conversation history — ring buffer of {"role": ..., "content": ...} dicts
       max_messages = self._history_turns * 2  # one user + one assistant per turn
       self._history: deque = deque(maxlen=max_messages if max_messages > 0 else None)
       self._history_enabled = self._history_turns > 0
 
       # ROS interfaces
       self._query_sub    = self.create_subscription(String, "/llm/query",    self._query_callback, 10)
       self._response_pub = self.create_publisher(  String, "/llm/response",  10)
       self._status_pub   = self.create_publisher(  String, "/llm/status",    10)
 
       # Prevent overlapping Ollama calls
       self._busy_lock = threading.Lock()
 
       self.get_logger().info(
           f"LLMClientNode ready\n"
           f"  model         : {self._ollama_model}\n"
           f"  ollama_host   : {self._ollama_host}\n"
           f"  max_tokens    : {self._max_tokens}\n"
           f"  temperature   : {self._temperature}\n"
           f"  history_turns : {self._history_turns}"
       )
 
   def _declare_params(self):
       self.declare_parameter("ollama_host",    "http://localhost:11434")
       self.declare_parameter("ollama_model",   "qwen3:0.6b")
       self.declare_parameter("max_tokens",     200)
       self.declare_parameter("temperature",    0.7)
       self.declare_parameter("history_turns",  6)
       self.declare_parameter("system_prompt",  _DEFAULT_SYSTEM_PROMPT)
 
   def _read_params(self):
       self._ollama_host   = self.get_parameter("ollama_host").value
       self._ollama_model  = self.get_parameter("ollama_model").value
       self._max_tokens    = self.get_parameter("max_tokens").value
       self._temperature   = self.get_parameter("temperature").value
       self._history_turns = self.get_parameter("history_turns").value
       self._system_prompt = self.get_parameter("system_prompt").value
 
   def _query_callback(self, msg: String):
       query = msg.data.strip()
       if not query:
           return
       self.get_logger().info(f"[LLM] Query received: '{query}'")
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
 
               if self._history_enabled:
                   self._history.append({"role": "user",      "content": query})
                   self._history.append({"role": "assistant",  "content": response_text})
 
               self.get_logger().info(f"[LLM] Response ({elapsed:.1f}s): '{response_text}'")
               self._publish_response(response_text)
               self._publish_status("ready", f"Done in {elapsed:.1f}s")
 
           except Exception as e:
               self.get_logger().error(f"[LLM] Error: {e}")
               self._publish_status("error", str(e))
               self._publish_response(
                   "I am sorry, I had trouble generating a response. Please try again."
               )
 
   def _build_messages(self, query: str) -> list[dict]:
       messages = [{"role": "system", "content": self._system_prompt}]
       if self._history_enabled:
           messages.extend(list(self._history))
       messages.append({"role": "user", "content": query})
       return messages
 
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


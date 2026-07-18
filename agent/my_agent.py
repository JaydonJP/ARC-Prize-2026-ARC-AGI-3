from __future__ import annotations

import json
import logging
import os
import textwrap
from typing import Any, Optional

import openai
from arcengine import FrameData, GameAction, GameState
from openai import OpenAI as OpenAIClient

# When run inside the ARC-AGI-3-Agents framework (locally or on Kaggle)
from agents.agent import Agent

logger = logging.getLogger(__name__)

class MyAgent(Agent):
    """An LLM-driven agent optimized for offline Kaggle evaluation with Qwen 2.5 Coder."""

    MAX_ACTIONS = 80
    MODEL = "qwen-coder"
    MESSAGE_LIMIT = 20

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.messages: list[dict[str, Any]] = []
        self.episodic_memory: list[str] = []
        self.token_counter = 0

    @property
    def name(self) -> str:
        return f"{super().name}.{self.MODEL}.agentic"

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        return latest_frame.state is GameState.WIN

    def _get_client(self):
        # Point to the local vLLM server running in the Kaggle notebook background
        base_url = os.environ.get("OPENAI_BASE_URL", "http://localhost:8000/v1")
        # Ensure we have a dummy API key for vLLM
        api_key = os.environ.get("OPENAI_API_KEY", "test-key-123")
        return OpenAIClient(api_key=api_key, base_url=base_url)

    def choose_action(
        self, frames: list[FrameData], latest_frame: FrameData
    ) -> GameAction:
        client = self._get_client()

        # Handle RESET manually
        if latest_frame.state in (GameState.NOT_PLAYED, GameState.GAME_OVER):
            self.messages = [] # clear context on reset
            self.episodic_memory.append(f"Level ended with state: {latest_frame.state.name}. Score: {latest_frame.levels_completed}")
            return GameAction.RESET

        # Build prompt representing the current grid state and episodic memory
        system_prompt = self.build_system_prompt()
        user_prompt = self.build_user_prompt(latest_frame)

        if not self.messages:
            self.messages.append({"role": "system", "content": system_prompt})
        
        self.messages.append({"role": "user", "content": user_prompt})

        # Trim messages if they get too long, preserving the system prompt
        if len(self.messages) > self.MESSAGE_LIMIT:
            self.messages = [self.messages[0]] + self.messages[-(self.MESSAGE_LIMIT-1):]

        tools = self.build_tools()

        try:
            response = client.chat.completions.create(
                model=self.MODEL,
                messages=self.messages,
                tools=tools,
                tool_choice="required",
                temperature=0.2, # Low temperature for more deterministic/logical play
            )
        except Exception as e:
            logger.warning(f"LLM API Error: {e}. Falling back to Random Action.")
            import random
            return random.choice([a for a in GameAction if a is not GameAction.RESET])

        message = response.choices[0].message
        self.messages.append(message.model_dump())

        if not message.tool_calls:
            # Fallback if the model didn't call a tool
            logger.warning("No tool call returned by LLM.")
            import random
            return random.choice([a for a in GameAction if a is not GameAction.RESET])

        tool_call = message.tool_calls[0]
        action_name = tool_call.function.name
        arguments = tool_call.function.arguments

        # Track the action for the next turn
        self.episodic_memory.append(f"Action taken: {action_name} with args {arguments}")

        try:
            data = json.loads(arguments) if arguments else {}
        except:
            data = {}

        action = GameAction.from_name(action_name)
        action.set_data(data)
        
        # Add reasoning metadata for the scorecard logs
        action.reasoning = {
            "model": self.MODEL,
            "chosen_action": action_name,
            "args": data
        }
        
        return action

    def build_system_prompt(self) -> str:
        return textwrap.dedent(
            """
            You are an expert AI agent playing an interactive grid-based reasoning game (ARC-AGI-3).
            Your goal is to WIN the game by completing all levels. 
            You must carefully analyze the visual grid representations (2D arrays) provided on each turn.
            
            Game dynamics:
            - The grid contains integers from 0 to 15, representing colors/objects.
            - You can move using ACTION1 (Up), ACTION2 (Down), ACTION3 (Left), ACTION4 (Right).
            - ACTION5 (Interact/Enter) and ACTION6 (Click) may also be available.
            - If your action causes no change in the grid, you probably hit a wall or took an invalid action.
            
            Memory Strategy:
            You must learn from previous steps. If you notice a repeated pattern of "no change", try a different direction.
            """
        )

    def build_user_prompt(self, latest_frame: FrameData) -> str:
        grid_repr = self.pretty_print_3d(latest_frame.frame)
        memory = "\n".join(self.episodic_memory[-5:]) # Show last 5 episodic events
        
        return textwrap.dedent(
            f"""
            # Current State: {latest_frame.state.name}
            # Levels Completed: {latest_frame.levels_completed}
            
            # Recent Memory (Last 5 events):
            {memory if memory else "None"}

            # Current Grid Observation:
            {grid_repr}

            # Instructions:
            Analyze the grid and the memory. Determine what changed since the last action.
            Call exactly one tool (action) to proceed.
            """
        )

    def pretty_print_3d(self, array_3d: list[list[list[Any]]]) -> str:
        lines = []
        for i, block in enumerate(array_3d):
            lines.append(f"Grid Layer {i}:")
            for row in block:
                # Format to align single digits with double digits for readable matrix
                formatted_row = "[" + ", ".join(f"{val:2d}" for val in row) + "]"
                lines.append(f"  {formatted_row}")
            lines.append("")
        return "\n".join(lines)

    def build_tools(self) -> list[dict[str, Any]]:
        empty_params = {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        }
        functions = [
            {"name": GameAction.ACTION1.name, "description": "Move Up", "parameters": empty_params},
            {"name": GameAction.ACTION2.name, "description": "Move Down", "parameters": empty_params},
            {"name": GameAction.ACTION3.name, "description": "Move Left", "parameters": empty_params},
            {"name": GameAction.ACTION4.name, "description": "Move Right", "parameters": empty_params},
            {"name": GameAction.ACTION5.name, "description": "Interact/Spacebar", "parameters": empty_params},
            {
                "name": GameAction.ACTION6.name,
                "description": "Click/Point",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "x": {"type": "integer", "description": "X coordinate (0-63)"},
                        "y": {"type": "integer", "description": "Y coordinate (0-63)"},
                    },
                    "required": ["x", "y"],
                    "additionalProperties": False,
                },
            },
        ]
        
        tools = []
        for f in functions:
            tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": f["name"],
                        "description": f["description"],
                        "parameters": f.get("parameters", {}),
                    },
                }
            )
        return tools

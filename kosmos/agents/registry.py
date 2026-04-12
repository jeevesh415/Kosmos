"""
Agent registry for discovering and managing agents.

Provides centralized management of all active agents in the system.

Reserved for multi-agent message-passing architecture (future work).
Not yet integrated into the main research loop.

Async Architecture (Issue #66 fix):
- _route_message() and send_message() are now async
- Sync wrappers provided for backwards compatibility
"""

from typing import Dict, List, Optional, Type, Any
from kosmos.agents.base import BaseAgent, AgentMessage, AgentStatus
import logging
from datetime import datetime, timezone
import asyncio


logger = logging.getLogger(__name__)


class AgentRegistry:
    """
    Central registry for managing agents.

    Provides:
    - Agent registration and discovery
    - Message routing between agents
    - Lifecycle management of multiple agents
    - Health monitoring

    Example:
        ```python
        registry = AgentRegistry()

        # Register agents
        hypothesis_gen = HypothesisGeneratorAgent()
        registry.register(hypothesis_gen)

        # Get agent by ID
        agent = registry.get_agent(hypothesis_gen.agent_id)

        # Send message between agents
        registry.send_message(
            from_agent_id=hypothesis_gen.agent_id,
            to_agent_id=lit_analyzer.agent_id,
            content={"query": "dark matter"}
        )

        # Check system health
        health = registry.get_system_health()
        ```
    """

    def __init__(self):
        """Initialize agent registry."""
        self._agents: Dict[str, BaseAgent] = {}
        self._agent_types: Dict[str, List[str]] = {}  # type -> [agent_ids]
        self._message_history: List[AgentMessage] = []
        self._max_history_size = 1000

        logger.info("Agent registry initialized")

    # ========================================================================
    # AGENT REGISTRATION
    # ========================================================================

    def register(self, agent: BaseAgent) -> str:
        """
        Register an agent.

        Args:
            agent: Agent instance to register

        Returns:
            str: Agent ID

        Raises:
            ValueError: If agent with same ID already registered
        """
        if agent.agent_id in self._agents:
            raise ValueError(f"Agent {agent.agent_id} already registered")

        self._agents[agent.agent_id] = agent

        # Track by type
        if agent.agent_type not in self._agent_types:
            self._agent_types[agent.agent_type] = []
        self._agent_types[agent.agent_type].append(agent.agent_id)

        # Set up message routing callback so agent.send_message() delivers messages
        agent.set_message_router(self._route_message)

        logger.info(f"Registered agent {agent.agent_type} ({agent.agent_id})")
        return agent.agent_id

    def unregister(self, agent_id: str):
        """
        Unregister an agent.

        Args:
            agent_id: ID of agent to unregister
        """
        if agent_id not in self._agents:
            logger.warning(f"Agent {agent_id} not found in registry")
            return

        agent = self._agents[agent_id]

        # Remove from type tracking
        if agent.agent_type in self._agent_types:
            self._agent_types[agent.agent_type].remove(agent_id)
            if not self._agent_types[agent.agent_type]:
                del self._agent_types[agent.agent_type]

        # Stop agent if running
        if agent.is_running():
            agent.stop()

        del self._agents[agent_id]
        logger.info(f"Unregistered agent {agent_id}")

    def get_agent(self, agent_id: str) -> Optional[BaseAgent]:
        """
        Get agent by ID.

        Args:
            agent_id: Agent ID

        Returns:
            BaseAgent: Agent instance or None if not found
        """
        return self._agents.get(agent_id)

    def get_agents_by_type(self, agent_type: str) -> List[BaseAgent]:
        """
        Get all agents of a specific type.

        Args:
            agent_type: Agent type name

        Returns:
            List[BaseAgent]: List of agents
        """
        agent_ids = self._agent_types.get(agent_type, [])
        return [self._agents[aid] for aid in agent_ids]

    def list_agents(self) -> List[Dict[str, Any]]:
        """
        List all registered agents.

        Returns:
            List[dict]: Agent information
        """
        return [agent.get_status() for agent in self._agents.values()]

    def list_agent_types(self) -> List[str]:
        """
        List all registered agent types.

        Returns:
            List[str]: Agent type names
        """
        return list(self._agent_types.keys())

    # ========================================================================
    # LIFECYCLE MANAGEMENT
    # ========================================================================

    def start_agent(self, agent_id: str):
        """
        Start a specific agent.

        Args:
            agent_id: Agent ID
        """
        agent = self.get_agent(agent_id)
        if not agent:
            raise ValueError(f"Agent {agent_id} not found")

        agent.start()
        logger.info(f"Started agent {agent_id}")

    def stop_agent(self, agent_id: str):
        """
        Stop a specific agent.

        Args:
            agent_id: Agent ID
        """
        agent = self.get_agent(agent_id)
        if not agent:
            raise ValueError(f"Agent {agent_id} not found")

        agent.stop()
        logger.info(f"Stopped agent {agent_id}")

    def start_all(self):
        """Start all registered agents."""
        logger.info("Starting all agents")
        for agent in self._agents.values():
            if not agent.is_running():
                agent.start()

    def stop_all(self):
        """Stop all registered agents."""
        logger.info("Stopping all agents")
        for agent in self._agents.values():
            if agent.is_running():
                agent.stop()

    def pause_agent(self, agent_id: str):
        """Pause a specific agent."""
        agent = self.get_agent(agent_id)
        if agent:
            agent.pause()

    def resume_agent(self, agent_id: str):
        """Resume a specific agent."""
        agent = self.get_agent(agent_id)
        if agent:
            agent.resume()

    # ========================================================================
    # MESSAGE ROUTING
    # ========================================================================

    async def _route_message(self, message: AgentMessage):
        """
        Internal async callback for routing messages from agents.

        This is set as the message_router on agents when they register,
        allowing agent.send_message() to automatically deliver messages.

        Args:
            message: Message to route to target agent
        """
        to_agent = self.get_agent(message.to_agent)

        if not to_agent:
            logger.error(f"Cannot route message: target agent {message.to_agent} not found")
            return

        # Deliver to recipient asynchronously
        await to_agent.receive_message(message)

        # Store in history
        self._message_history.append(message)
        if len(self._message_history) > self._max_history_size:
            self._message_history.pop(0)

        logger.debug(f"Routed message from {message.from_agent} to {message.to_agent}")

    def _route_message_sync(self, message: AgentMessage):
        """
        Synchronous wrapper for _route_message (backwards compatibility).
        """
        try:
            loop = asyncio.get_running_loop()
            future = asyncio.run_coroutine_threadsafe(
                self._route_message(message), loop
            )
            return future.result(timeout=30)
        except RuntimeError:
            return asyncio.run(self._route_message(message))

    async def send_message(
        self,
        from_agent_id: str,
        to_agent_id: str,
        content: Dict[str, Any],
        message_type: str = "request",
        correlation_id: Optional[str] = None
    ) -> AgentMessage:
        """
        Route message from one agent to another asynchronously.

        Args:
            from_agent_id: Sender agent ID
            to_agent_id: Recipient agent ID
            content: Message content
            message_type: Type of message
            correlation_id: Optional correlation ID

        Returns:
            AgentMessage: The sent message
        """
        from_agent = self.get_agent(from_agent_id)
        to_agent = self.get_agent(to_agent_id)

        if not from_agent:
            raise ValueError(f"From agent {from_agent_id} not found")
        if not to_agent:
            raise ValueError(f"To agent {to_agent_id} not found")

        # Create and send message asynchronously
        from kosmos.agents.base import MessageType
        message = await from_agent.send_message(
            to_agent=to_agent_id,
            content=content,
            message_type=MessageType(message_type),
            correlation_id=correlation_id
        )

        # Note: message is already delivered via _route_message callback
        # Store in history (redundant with _route_message, but kept for direct sends)
        self._message_history.append(message)
        if len(self._message_history) > self._max_history_size:
            self._message_history.pop(0)

        return message

    def send_message_sync(
        self,
        from_agent_id: str,
        to_agent_id: str,
        content: Dict[str, Any],
        message_type: str = "request",
        correlation_id: Optional[str] = None
    ) -> AgentMessage:
        """
        Synchronous wrapper for send_message (backwards compatibility).
        """
        try:
            loop = asyncio.get_running_loop()
            future = asyncio.run_coroutine_threadsafe(
                self.send_message(from_agent_id, to_agent_id, content, message_type, correlation_id),
                loop
            )
            return future.result(timeout=30)
        except RuntimeError:
            return asyncio.run(
                self.send_message(from_agent_id, to_agent_id, content, message_type, correlation_id)
            )

    async def broadcast_message(
        self,
        from_agent_id: str,
        content: Dict[str, Any],
        target_types: Optional[List[str]] = None
    ) -> List[AgentMessage]:
        """
        Broadcast message to multiple agents asynchronously.

        Args:
            from_agent_id: Sender agent ID
            content: Message content
            target_types: Optional list of agent types to target (None = all)

        Returns:
            List[AgentMessage]: Sent messages
        """
        from_agent = self.get_agent(from_agent_id)
        if not from_agent:
            raise ValueError(f"From agent {from_agent_id} not found")

        messages = []

        # Determine target agents
        if target_types:
            targets = []
            for agent_type in target_types:
                targets.extend(self.get_agents_by_type(agent_type))
        else:
            targets = [a for a in self._agents.values() if a.agent_id != from_agent_id]

        # Send to all targets concurrently
        send_tasks = [
            self.send_message(
                from_agent_id=from_agent_id,
                to_agent_id=target.agent_id,
                content=content,
                message_type="notification"
            )
            for target in targets
        ]
        results = await asyncio.gather(*send_tasks, return_exceptions=True)

        messages = []
        for i, result in enumerate(results):
            if isinstance(result, BaseException):
                logger.error(f"Failed to send broadcast to {targets[i].agent_id}: {result}")
            else:
                messages.append(result)

        logger.info(f"Broadcast message from {from_agent_id} to {len(messages)}/{len(targets)} agents")
        return messages

    def broadcast_message_sync(
        self,
        from_agent_id: str,
        content: Dict[str, Any],
        target_types: Optional[List[str]] = None
    ) -> List[AgentMessage]:
        """
        Synchronous wrapper for broadcast_message (backwards compatibility).
        """
        try:
            loop = asyncio.get_running_loop()
            future = asyncio.run_coroutine_threadsafe(
                self.broadcast_message(from_agent_id, content, target_types),
                loop
            )
            return future.result(timeout=60)
        except RuntimeError:
            return asyncio.run(
                self.broadcast_message(from_agent_id, content, target_types)
            )

    def get_message_history(
        self,
        agent_id: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get message history.

        Args:
            agent_id: Optional agent ID to filter by (None = all)
            limit: Max number of messages to return

        Returns:
            List[dict]: Message history
        """
        messages = self._message_history

        if agent_id:
            messages = [
                m for m in messages
                if m.from_agent == agent_id or m.to_agent == agent_id
            ]

        # Return most recent messages
        messages = messages[-limit:]
        return [m.to_dict() for m in messages]

    # ========================================================================
    # HEALTH MONITORING
    # ========================================================================

    def get_system_health(self) -> Dict[str, Any]:
        """
        Get overall system health.

        Returns:
            dict: Health status of all agents
        """
        total_agents = len(self._agents)
        healthy_agents = sum(1 for a in self._agents.values() if a.is_healthy())
        running_agents = sum(1 for a in self._agents.values() if a.is_running())

        agent_health = {
            agent_id: {
                "status": agent.status,
                "is_healthy": agent.is_healthy(),
                "message_queue_length": len(agent.message_queue),
                "errors": agent.errors_encountered
            }
            for agent_id, agent in self._agents.items()
        }

        return {
            "system_healthy": healthy_agents == total_agents,
            "total_agents": total_agents,
            "healthy_agents": healthy_agents,
            "running_agents": running_agents,
            "unhealthy_agents": total_agents - healthy_agents,
            "agent_health": agent_health,
            "message_history_size": len(self._message_history),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

    def get_agent_statistics(self) -> Dict[str, Any]:
        """
        Get statistics for all agents.

        Returns:
            dict: Agent statistics
        """
        return {
            "total_agents": len(self._agents),
            "agents_by_type": {
                agent_type: len(agent_ids)
                for agent_type, agent_ids in self._agent_types.items()
            },
            "total_messages_sent": sum(a.messages_sent for a in self._agents.values()),
            "total_messages_received": sum(a.messages_received for a in self._agents.values()),
            "total_tasks_completed": sum(a.tasks_completed for a in self._agents.values()),
            "total_errors": sum(a.errors_encountered for a in self._agents.values()),
        }

    # ========================================================================
    # UTILITY
    # ========================================================================

    def clear(self):
        """Clear all agents from registry."""
        self.stop_all()
        self._agents.clear()
        self._agent_types.clear()
        self._message_history.clear()
        logger.info("Cleared agent registry")

    def __len__(self) -> int:
        """Return number of registered agents."""
        return len(self._agents)

    def __contains__(self, agent_id: str) -> bool:
        """Check if agent is registered."""
        return agent_id in self._agents


# Singleton registry instance
_registry: Optional[AgentRegistry] = None


def get_registry(reset: bool = False) -> AgentRegistry:
    """
    Get or create agent registry singleton.

    Args:
        reset: If True, create new registry instance

    Returns:
        AgentRegistry: Registry instance
    """
    global _registry
    if _registry is None or reset:
        _registry = AgentRegistry()
    return _registry

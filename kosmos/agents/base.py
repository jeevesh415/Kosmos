"""
Base agent class and communication protocol.

All agents (HypothesisGenerator, ExperimentDesigner, DataAnalyst, etc.) inherit from this base.

Async Architecture (Issue #66 fix):
- send_message(), receive_message(), process_message() are now async
- Uses asyncio.Queue for message queue
- Sync wrappers provided for backwards compatibility
"""

from typing import Dict, Any, Optional, List, Callable, Awaitable, Union
from enum import Enum
from datetime import datetime, timezone
from pydantic import BaseModel, Field
import logging
import uuid
import json
import asyncio


logger = logging.getLogger(__name__)


class AgentStatus(str, Enum):
    """Agent lifecycle status."""
    CREATED = "created"
    STARTING = "starting"
    RUNNING = "running"
    IDLE = "idle"
    WORKING = "working"
    PAUSED = "paused"
    STOPPED = "stopped"
    ERROR = "error"


class MessageType(str, Enum):
    """Types of inter-agent messages."""
    REQUEST = "request"
    RESPONSE = "response"
    NOTIFICATION = "notification"
    ERROR = "error"


class AgentMessage(BaseModel):
    """
    Message for inter-agent communication.

    Example:
        ```python
        msg = AgentMessage(
            type=MessageType.REQUEST,
            from_agent="hypothesis_generator",
            to_agent="literature_analyzer",
            content={"query": "dark matter papers"},
            correlation_id="exp-123"
        )
        ```
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    type: MessageType
    from_agent: str
    to_agent: str
    content: Dict[str, Any]
    correlation_id: Optional[str] = None  # For tracking request/response pairs
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "type": self.type,
            "from_agent": self.from_agent,
            "to_agent": self.to_agent,
            "content": self.content,
            "correlation_id": self.correlation_id,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata
        }

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict())


class AgentState(BaseModel):
    """Agent state for persistence."""
    agent_id: str
    agent_type: str
    status: AgentStatus
    data: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class BaseAgent:
    """
    Base class for all Kosmos agents.

    Provides:
    - Lifecycle management (start, stop, pause, resume)
    - Message passing for inter-agent communication
    - State persistence
    - Health checks
    - Logging

    Subclasses should implement:
    - process_message(): Handle incoming messages
    - execute(): Main agent logic
    """

    def __init__(
        self,
        agent_id: Optional[str] = None,
        agent_type: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize agent.

        Args:
            agent_id: Unique agent ID (auto-generated if None)
            agent_type: Agent type name (uses class name if None)
            config: Optional configuration dict
        """
        self.agent_id = agent_id or str(uuid.uuid4())
        self.agent_type = agent_type or self.__class__.__name__
        self.config = config or {}

        self.status = AgentStatus.CREATED
        self.created_at = datetime.now(timezone.utc)
        self.updated_at = datetime.now(timezone.utc)

        # Message handling - async queue for proper async processing
        self.message_queue: List[AgentMessage] = []  # Legacy sync queue (for compatibility)
        self._async_message_queue: asyncio.Queue[AgentMessage] = asyncio.Queue()
        self.message_handlers: Dict[str, Callable] = {}
        # Message router can be sync or async callable
        self._message_router: Optional[
            Union[Callable[[AgentMessage], None], Callable[[AgentMessage], Awaitable[None]]]
        ] = None

        # State management
        self.state_data: Dict[str, Any] = {}

        # Statistics
        self.messages_received = 0
        self.messages_sent = 0
        self.tasks_completed = 0
        self.errors_encountered = 0

        logger.info(f"Agent {self.agent_type} ({self.agent_id}) created")

    # ========================================================================
    # LIFECYCLE MANAGEMENT
    # ========================================================================

    def start(self):
        """Start the agent."""
        if self.status != AgentStatus.CREATED:
            logger.warning(f"Agent {self.agent_id} already started")
            return

        self.status = AgentStatus.STARTING
        logger.info(f"Starting agent {self.agent_id}")

        try:
            self._on_start()
            self.status = AgentStatus.RUNNING
            logger.info(f"Agent {self.agent_id} started successfully")
        except Exception as e:
            self.status = AgentStatus.ERROR
            logger.error(f"Failed to start agent {self.agent_id}: {e}")
            raise

    def stop(self):
        """Stop the agent."""
        logger.info(f"Stopping agent {self.agent_id}")

        try:
            self._on_stop()
            self.status = AgentStatus.STOPPED
            logger.info(f"Agent {self.agent_id} stopped")
        except Exception as e:
            logger.error(f"Error stopping agent {self.agent_id}: {e}")
            raise

    def pause(self):
        """Pause the agent."""
        if self.status != AgentStatus.RUNNING:
            logger.warning(f"Cannot pause agent {self.agent_id} in status {self.status}")
            return

        self.status = AgentStatus.PAUSED
        self._on_pause()
        logger.info(f"Agent {self.agent_id} paused")

    def resume(self):
        """Resume the agent."""
        if self.status != AgentStatus.PAUSED:
            logger.warning(f"Cannot resume agent {self.agent_id} in status {self.status}")
            return

        self.status = AgentStatus.RUNNING
        self._on_resume()
        logger.info(f"Agent {self.agent_id} resumed")

    def is_running(self) -> bool:
        """Check if agent is running."""
        return self.status in (AgentStatus.RUNNING, AgentStatus.WORKING, AgentStatus.IDLE)

    def is_healthy(self) -> bool:
        """
        Health check.

        Subclasses can override for custom health checks.
        """
        return self.status in [AgentStatus.RUNNING, AgentStatus.IDLE, AgentStatus.WORKING]

    def get_status(self) -> Dict[str, Any]:
        """
        Get agent status information.

        Returns:
            dict: Status including state, statistics, health
        """
        return {
            "agent_id": self.agent_id,
            "agent_type": self.agent_type,
            "status": self.status,
            "is_healthy": self.is_healthy(),
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "statistics": {
                "messages_received": self.messages_received,
                "messages_sent": self.messages_sent,
                "tasks_completed": self.tasks_completed,
                "errors_encountered": self.errors_encountered,
            },
            "message_queue_length": len(self.message_queue),
        }

    # ========================================================================
    # MESSAGE PASSING
    # ========================================================================

    async def send_message(
        self,
        to_agent: str,
        content: Dict[str, Any],
        message_type: MessageType = MessageType.REQUEST,
        correlation_id: Optional[str] = None
    ) -> AgentMessage:
        """
        Send message to another agent asynchronously.

        Args:
            to_agent: Target agent ID
            content: Message content
            message_type: Type of message
            correlation_id: Optional correlation ID

        Returns:
            AgentMessage: The sent message
        """
        message = AgentMessage(
            type=message_type,
            from_agent=self.agent_id,
            to_agent=to_agent,
            content=content,
            correlation_id=correlation_id
        )

        self.messages_sent += 1
        # Log agent message if enabled
        try:
            from kosmos.config import get_config
            if get_config().logging.log_agent_messages:
                logger.debug(
                    "[MSG] %s -> %s: type=%s, correlation_id=%s, content_preview=%.100s",
                    self.agent_id,
                    to_agent,
                    message_type.value,
                    correlation_id or message.id,
                    str(content)[:100]
                )
        except Exception:
            pass  # Config not available

        # Use message router if available to actually deliver the message
        if self._message_router is not None:
            try:
                # Handle both sync and async routers
                result = self._message_router(message)
                if asyncio.iscoroutine(result):
                    await result
                logger.debug(f"Message routed from {self.agent_id} to {to_agent}")
            except Exception as e:
                logger.error(f"Failed to route message from {self.agent_id} to {to_agent}: {e}")

        return message

    def send_message_sync(
        self,
        to_agent: str,
        content: Dict[str, Any],
        message_type: MessageType = MessageType.REQUEST,
        correlation_id: Optional[str] = None
    ) -> AgentMessage:
        """
        Synchronous wrapper for send_message (backwards compatibility).

        Use this when calling from synchronous code that cannot be converted to async.
        """
        try:
            loop = asyncio.get_running_loop()
            # We're in an async context, use run_coroutine_threadsafe
            import concurrent.futures
            future = asyncio.run_coroutine_threadsafe(
                self.send_message(to_agent, content, message_type, correlation_id),
                loop
            )
            return future.result(timeout=30)
        except RuntimeError:
            # No running loop, create one
            return asyncio.run(
                self.send_message(to_agent, content, message_type, correlation_id)
            )

    async def receive_message(self, message: AgentMessage):
        """
        Receive message from another agent asynchronously.

        Args:
            message: Incoming message
        """
        self.messages_received += 1
        self.message_queue.append(message)  # Legacy sync queue
        await self._async_message_queue.put(message)  # Async queue

        # Log received message if enabled
        try:
            from kosmos.config import get_config
            if get_config().logging.log_agent_messages:
                logger.debug(
                    "[MSG] %s <- %s: type=%s, msg_id=%s",
                    self.agent_id,
                    message.from_agent,
                    message.type.value,
                    message.id
                )
        except Exception:
            pass  # Config not available

        # Process message
        try:
            await self.process_message(message)
        except Exception as e:
            self.errors_encountered += 1
            logger.error(f"Error processing message in {self.agent_id}: {e}")
            # Send error response
            if message.type == MessageType.REQUEST:
                await self.send_message(
                    to_agent=message.from_agent,
                    content={"error": str(e)},
                    message_type=MessageType.ERROR,
                    correlation_id=message.id
                )

    def receive_message_sync(self, message: AgentMessage):
        """
        Synchronous wrapper for receive_message (backwards compatibility).
        """
        try:
            loop = asyncio.get_running_loop()
            future = asyncio.run_coroutine_threadsafe(
                self.receive_message(message), loop
            )
            return future.result(timeout=30)
        except RuntimeError:
            return asyncio.run(self.receive_message(message))

    async def process_message(self, message: AgentMessage):
        """
        Process incoming message asynchronously.

        Subclasses should override this to implement message handling logic.

        Args:
            message: Message to process
        """
        logger.warning(f"Agent {self.agent_id} received message but process_message() not implemented")

    def process_message_sync(self, message: AgentMessage):
        """
        Synchronous wrapper for process_message (backwards compatibility).
        """
        try:
            loop = asyncio.get_running_loop()
            future = asyncio.run_coroutine_threadsafe(
                self.process_message(message), loop
            )
            return future.result(timeout=30)
        except RuntimeError:
            return asyncio.run(self.process_message(message))

    def register_message_handler(self, message_type: str, handler: Callable):
        """
        Register handler for specific message type.

        Args:
            message_type: Type of message to handle
            handler: Callable that takes AgentMessage
        """
        self.message_handlers[message_type] = handler
        logger.debug(f"Registered handler for {message_type} in agent {self.agent_id}")

    def set_message_router(
        self,
        router: Union[Callable[[AgentMessage], None], Callable[[AgentMessage], Awaitable[None]]]
    ):
        """
        Set the message router callback for delivering messages.

        When set, send_message() will use this callback to actually deliver
        messages to their target agents. This enables proper message routing
        through a central registry or message broker.

        Args:
            router: Callable that takes an AgentMessage and delivers it.
                   Can be sync or async - async routers will be awaited.
        """
        self._message_router = router
        logger.debug(f"Message router set for agent {self.agent_id}")

    # ========================================================================
    # STATE PERSISTENCE
    # ========================================================================

    def get_state(self) -> AgentState:
        """
        Get current agent state for persistence.

        Returns:
            AgentState: Current state
        """
        self.updated_at = datetime.now(timezone.utc)
        return AgentState(
            agent_id=self.agent_id,
            agent_type=self.agent_type,
            status=self.status,
            data=self.state_data,
            created_at=self.created_at,
            updated_at=self.updated_at
        )

    def restore_state(self, state: AgentState):
        """
        Restore agent from saved state.

        Args:
            state: Saved state to restore
        """
        self.agent_id = state.agent_id
        self.agent_type = state.agent_type
        self.status = state.status
        self.state_data = state.data
        self.created_at = state.created_at
        self.updated_at = state.updated_at

        logger.info(f"Restored agent {self.agent_id} from saved state")

    def save_state_data(self, key: str, value: Any):
        """Save data to agent state."""
        self.state_data[key] = value
        self.updated_at = datetime.now(timezone.utc)

    def get_state_data(self, key: str, default: Any = None) -> Any:
        """Retrieve data from agent state."""
        return self.state_data.get(key, default)

    # ========================================================================
    # EXECUTION
    # ========================================================================

    def execute(self, task: Union[Dict[str, Any], "AgentMessage"]) -> Union[Dict[str, Any], "AgentMessage"]:
        """
        Execute agent task.

        Subclasses should override this to implement main agent logic.
        Accepts either a Dict task specification or an AgentMessage,
        depending on the agent's communication pattern.

        Args:
            task: Task specification (Dict) or AgentMessage

        Returns:
            Task result as Dict or AgentMessage
        """
        raise NotImplementedError(f"execute() not implemented for {self.agent_type}")

    # ========================================================================
    # LIFECYCLE HOOKS (for subclasses to override)
    # ========================================================================

    def _on_start(self):
        """Hook called when agent starts. Override in subclasses."""
        pass

    def _on_stop(self):
        """Hook called when agent stops. Override in subclasses."""
        pass

    def _on_pause(self):
        """Hook called when agent pauses. Override in subclasses."""
        pass

    def _on_resume(self):
        """Hook called when agent resumes. Override in subclasses."""
        pass

    # ========================================================================
    # UTILITY
    # ========================================================================

    def __repr__(self) -> str:
        return f"<{self.agent_type} id={self.agent_id} status={self.status}>"

    def __str__(self) -> str:
        return f"{self.agent_type}({self.agent_id})"

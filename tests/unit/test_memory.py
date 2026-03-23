"""Unit tests for vector_os_nano.core.memory — SessionMemory."""
from __future__ import annotations

import time

import pytest


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def make_execution_result(
    success: bool = True,
    status: str = "completed",
    steps: list[tuple[str, str, str]] | None = None,
    failed_step_name: str | None = None,
    failure_reason: str | None = None,
):
    """Build an ExecutionResult from simple tuples (step_id, skill_name, status)."""
    from vector_os_nano.core.types import ExecutionResult, StepTrace, TaskStep

    trace = []
    if steps:
        for step_id, skill_name, step_status in steps:
            error = ""
            if step_status not in ("success", "ok"):
                error = f"{skill_name} not found" if "not found" not in step_status else step_status
            trace.append(
                StepTrace(
                    step_id=step_id,
                    skill_name=skill_name,
                    status=step_status,
                    duration_sec=1.0,
                    error=error,
                )
            )

    failed_step = None
    if failed_step_name:
        failed_step = TaskStep(step_id="failed_s", skill_name=failed_step_name)

    return ExecutionResult(
        success=success,
        status=status,
        steps_completed=sum(1 for t in trace if t.status in ("success", "ok")),
        steps_total=len(trace),
        failed_step=failed_step,
        failure_reason=failure_reason,
        trace=trace,
    )


@pytest.fixture
def mem():
    from vector_os_nano.core.memory import SessionMemory
    return SessionMemory(max_entries=50)


# ---------------------------------------------------------------------------
# Basic add / retrieve
# ---------------------------------------------------------------------------


class TestAddUserMessage:
    def test_add_user_message(self, mem):
        mem.add_user_message("hello")
        assert len(mem.entries) == 1
        entry = mem.entries[0]
        assert entry.role == "user"
        assert entry.content == "hello"
        assert entry.entry_type == "chat"

    def test_add_user_message_task_type(self, mem):
        mem.add_user_message("pick the cup", entry_type="task")
        entry = mem.entries[0]
        assert entry.entry_type == "task"
        assert entry.role == "user"

    def test_add_user_message_sets_timestamp(self, mem):
        before = time.time()
        mem.add_user_message("hello")
        after = time.time()
        assert before <= mem.entries[0].timestamp <= after


class TestAddAssistantMessage:
    def test_add_assistant_message(self, mem):
        mem.add_assistant_message("I am ready.")
        assert len(mem.entries) == 1
        entry = mem.entries[0]
        assert entry.role == "assistant"
        assert entry.content == "I am ready."

    def test_add_assistant_message_entry_type(self, mem):
        mem.add_assistant_message("done", entry_type="task")
        assert mem.entries[0].entry_type == "task"


# ---------------------------------------------------------------------------
# Task result
# ---------------------------------------------------------------------------


class TestAddTaskResultSuccess:
    def test_summary_contains_instruction(self, mem):
        result = make_execution_result(
            success=True,
            status="completed",
            steps=[("s1", "scan", "success"), ("s2", "pick", "success")],
        )
        mem.add_task_result("pick the red cup", result)
        entry = mem.entries[0]
        assert "pick the red cup" in entry.content

    def test_summary_shows_completed(self, mem):
        result = make_execution_result(success=True, status="completed")
        mem.add_task_result("pick the mug", result)
        assert "completed" in mem.entries[0].content

    def test_summary_shows_step_trace(self, mem):
        result = make_execution_result(
            success=True,
            status="completed",
            steps=[
                ("s1", "scan", "success"),
                ("s2", "detect", "success"),
                ("s3", "pick", "success"),
                ("s4", "home", "success"),
            ],
        )
        mem.add_task_result("pick the red cup", result)
        content = mem.entries[0].content
        assert "scan(ok)" in content
        assert "detect(ok)" in content
        assert "pick(ok)" in content
        assert "home(ok)" in content
        assert "->" in content

    def test_task_result_role_is_task_result(self, mem):
        result = make_execution_result(success=True)
        mem.add_task_result("pick the mug", result)
        assert mem.entries[0].role == "task_result"

    def test_task_result_entry_type(self, mem):
        result = make_execution_result(success=True)
        mem.add_task_result("pick the mug", result)
        assert mem.entries[0].entry_type == "task_result"

    def test_metadata_contains_success(self, mem):
        result = make_execution_result(success=True)
        mem.add_task_result("pick the mug", result)
        assert mem.entries[0].metadata["success"] is True


class TestAddTaskResultFailure:
    def test_summary_shows_failed(self, mem):
        result = make_execution_result(
            success=False,
            status="failed",
            steps=[("s1", "scan", "success"), ("s2", "detect", "execution_failed")],
            failed_step_name="detect",
            failure_reason="object not found",
        )
        mem.add_task_result("pick the blue ball", result)
        content = mem.entries[0].content
        assert "failed" in content

    def test_summary_shows_failed_step_name(self, mem):
        result = make_execution_result(
            success=False,
            status="failed",
            steps=[("s1", "scan", "success"), ("s2", "detect", "execution_failed")],
            failed_step_name="detect",
        )
        mem.add_task_result("pick the blue ball", result)
        assert "detect" in mem.entries[0].content

    def test_summary_shows_failed_step_in_trace(self, mem):
        result = make_execution_result(
            success=False,
            status="failed",
            steps=[("s1", "scan", "success"), ("s2", "detect", "execution_failed")],
        )
        mem.add_task_result("pick the blue ball", result)
        content = mem.entries[0].content
        assert "scan(ok)" in content
        assert "detect(failed" in content

    def test_metadata_contains_failure(self, mem):
        result = make_execution_result(success=False, status="failed")
        mem.add_task_result("pick the blue ball", result)
        assert mem.entries[0].metadata["success"] is False


class TestTaskResultWithWorldDiff:
    def test_world_diff_included_in_summary(self, mem):
        result = make_execution_result(success=True, status="completed")
        world_diff = {"gripper_state": "closed", "held_object": "red_cup"}
        mem.add_task_result("pick the red cup", result, world_diff=world_diff)
        content = mem.entries[0].content
        assert "gripper_state=closed" in content
        assert "held_object=red_cup" in content

    def test_world_diff_stored_in_metadata(self, mem):
        result = make_execution_result(success=True)
        world_diff = {"held_object": "mug"}
        mem.add_task_result("pick the mug", result, world_diff=world_diff)
        assert mem.entries[0].metadata["world_diff"] == world_diff


class TestTaskResultWithoutWorldDiff:
    def test_no_world_diff_says_no_changes(self, mem):
        result = make_execution_result(success=True, status="completed")
        mem.add_task_result("scan", result, world_diff=None)
        assert "no changes" in mem.entries[0].content

    def test_no_world_diff_no_metadata_key(self, mem):
        result = make_execution_result(success=True)
        mem.add_task_result("scan", result)
        assert "world_diff" not in mem.entries[0].metadata


# ---------------------------------------------------------------------------
# get_llm_history
# ---------------------------------------------------------------------------


class TestGetLLMHistoryFormat:
    def test_returns_list_of_dicts(self, mem):
        mem.add_user_message("hello")
        history = mem.get_llm_history()
        assert isinstance(history, list)
        assert all(isinstance(h, dict) for h in history)

    def test_dict_has_role_and_content_keys(self, mem):
        mem.add_user_message("hello")
        history = mem.get_llm_history()
        assert "role" in history[0]
        assert "content" in history[0]

    def test_only_user_and_assistant_roles(self, mem):
        from vector_os_nano.core.memory import MemoryEntry
        # Inject a system entry directly to test filtering
        mem._entries.append(
            MemoryEntry(role="system", content="system prompt", timestamp=time.time())
        )
        mem.add_user_message("hi")
        history = mem.get_llm_history()
        roles = {h["role"] for h in history}
        assert "system" not in roles
        assert roles.issubset({"user", "assistant"})

    def test_task_result_mapped_to_assistant(self, mem):
        result = make_execution_result(success=True)
        mem.add_task_result("pick the cup", result)
        history = mem.get_llm_history()
        assert history[0]["role"] == "assistant"


class TestGetLLMHistoryMaxTurns:
    def test_returns_at_most_max_turns(self, mem):
        for i in range(30):
            mem.add_user_message(f"message {i}")
        history = mem.get_llm_history(max_turns=10)
        assert len(history) == 10

    def test_returns_most_recent_entries(self, mem):
        for i in range(30):
            mem.add_user_message(f"message {i}")
        history = mem.get_llm_history(max_turns=5)
        # Most recent 5 messages should be 25..29
        assert history[-1]["content"] == "message 29"
        assert history[0]["content"] == "message 25"

    def test_fewer_entries_than_max_turns_returns_all(self, mem):
        mem.add_user_message("only one")
        history = mem.get_llm_history(max_turns=20)
        assert len(history) == 1


class TestGetLLMHistoryCrossMode:
    def test_chat_task_chat_task_continuity(self, mem):
        """Verify all four mode transitions are preserved in history."""
        mem.add_user_message("hello", entry_type="chat")
        mem.add_assistant_message("hi there", entry_type="chat")

        result1 = make_execution_result(success=True, status="completed")
        mem.add_user_message("pick the red cup", entry_type="task")
        mem.add_task_result("pick the red cup", result1)

        mem.add_user_message("nice work", entry_type="chat")
        mem.add_assistant_message("thank you", entry_type="chat")

        result2 = make_execution_result(success=True, status="completed")
        mem.add_user_message("now put it on the left", entry_type="task")
        mem.add_task_result("now put it on the left", result2)

        history = mem.get_llm_history(max_turns=20)
        # Should have 8 entries: 2 chat + user task + task_result + 2 chat + user task + task_result
        assert len(history) == 8
        assert history[0]["content"] == "hello"
        assert history[-1]["role"] == "assistant"  # last task_result

    def test_task_result_content_in_history(self, mem):
        result = make_execution_result(success=True, status="completed")
        mem.add_task_result("pick the cup", result)
        history = mem.get_llm_history()
        assert "pick the cup" in history[0]["content"]


# ---------------------------------------------------------------------------
# get_last_task_context
# ---------------------------------------------------------------------------


class TestGetLastTaskContext:
    def test_returns_none_when_no_tasks(self, mem):
        mem.add_user_message("hello")
        assert mem.get_last_task_context() is None

    def test_returns_metadata_after_one_task(self, mem):
        result = make_execution_result(success=True, status="completed")
        mem.add_task_result("pick the mug", result)
        ctx = mem.get_last_task_context()
        assert ctx is not None
        assert ctx["instruction"] == "pick the mug"
        assert ctx["success"] is True

    def test_returns_most_recent_task_after_two(self, mem):
        r1 = make_execution_result(success=True, status="completed")
        mem.add_task_result("pick the mug", r1)

        r2 = make_execution_result(success=False, status="failed")
        mem.add_task_result("put it on the left", r2)

        ctx = mem.get_last_task_context()
        assert ctx["instruction"] == "put it on the left"
        assert ctx["success"] is False

    def test_chat_between_tasks_does_not_affect_context(self, mem):
        r1 = make_execution_result(success=True)
        mem.add_task_result("pick the mug", r1)
        mem.add_user_message("great job")
        mem.add_assistant_message("thank you")
        ctx = mem.get_last_task_context()
        assert ctx["instruction"] == "pick the mug"


# ---------------------------------------------------------------------------
# Bounded size / trim
# ---------------------------------------------------------------------------


class TestBoundedSize:
    def test_bounded_size_max_50(self):
        from vector_os_nano.core.memory import SessionMemory
        mem = SessionMemory(max_entries=50)
        for i in range(100):
            mem.add_user_message(f"msg {i}")
        assert len(mem.entries) == 50

    def test_trim_removes_oldest(self):
        from vector_os_nano.core.memory import SessionMemory
        mem = SessionMemory(max_entries=5)
        for i in range(7):
            mem.add_user_message(f"msg {i}")
        entries = mem.entries
        assert len(entries) == 5
        # oldest 2 (msg 0, msg 1) should be gone
        contents = [e.content for e in entries]
        assert "msg 0" not in contents
        assert "msg 1" not in contents
        assert "msg 6" in contents

    def test_exact_max_entries_no_trim(self):
        from vector_os_nano.core.memory import SessionMemory
        mem = SessionMemory(max_entries=3)
        mem.add_user_message("a")
        mem.add_user_message("b")
        mem.add_user_message("c")
        assert len(mem.entries) == 3


# ---------------------------------------------------------------------------
# clear
# ---------------------------------------------------------------------------


class TestClear:
    def test_clear_removes_all(self, mem):
        mem.add_user_message("hello")
        mem.add_assistant_message("hi")
        mem.clear()
        assert len(mem.entries) == 0

    def test_clear_on_empty_is_safe(self, mem):
        mem.clear()
        assert len(mem.entries) == 0

    def test_add_after_clear(self, mem):
        mem.add_user_message("hello")
        mem.clear()
        mem.add_user_message("new start")
        assert len(mem.entries) == 1
        assert mem.entries[0].content == "new start"


# ---------------------------------------------------------------------------
# entries returns copy
# ---------------------------------------------------------------------------


class TestEntriesReturnsCopy:
    def test_modifying_returned_list_does_not_affect_internal(self, mem):
        mem.add_user_message("hello")
        snapshot = mem.entries
        snapshot.clear()
        assert len(mem.entries) == 1

    def test_entries_order_preserved(self, mem):
        for msg in ("first", "second", "third"):
            mem.add_user_message(msg)
        contents = [e.content for e in mem.entries]
        assert contents == ["first", "second", "third"]


# ---------------------------------------------------------------------------
# MemoryEntry is frozen (immutable)
# ---------------------------------------------------------------------------


class TestMemoryEntryFrozen:
    def test_cannot_set_role(self):
        from vector_os_nano.core.memory import MemoryEntry
        entry = MemoryEntry(role="user", content="hello", timestamp=time.time())
        with pytest.raises((AttributeError, TypeError)):
            entry.role = "assistant"  # type: ignore[misc]

    def test_cannot_set_content(self):
        from vector_os_nano.core.memory import MemoryEntry
        entry = MemoryEntry(role="user", content="hello", timestamp=time.time())
        with pytest.raises((AttributeError, TypeError)):
            entry.content = "changed"  # type: ignore[misc]

    def test_default_entry_type(self):
        from vector_os_nano.core.memory import MemoryEntry
        entry = MemoryEntry(role="user", content="hi", timestamp=0.0)
        assert entry.entry_type == "chat"

    def test_default_metadata_is_empty(self):
        from vector_os_nano.core.memory import MemoryEntry
        entry = MemoryEntry(role="user", content="hi", timestamp=0.0)
        assert entry.metadata == {}

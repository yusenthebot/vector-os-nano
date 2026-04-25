# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Multi-layer permission system for Vector CLI's agentic harness.

Public exports:
    PermissionContext — stateful permission checker for a single agent session

Check order (mirrors Claude Code's hasPermissionsToUseTool):
    1. no_permission flag  → allow everything immediately
    2. Deny rules          → immediate deny
    3. Tool check_permissions() returning "deny" → deny (bypasses all modes)
    4. Session always-allow → allow
    5. Read-only auto-allow
    6. Default             → ask
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from vector_os_nano.vcli.tools.base import PermissionResult


# ---------------------------------------------------------------------------
# PermissionContext
# ---------------------------------------------------------------------------


@dataclass
class PermissionContext:
    """Stateful per-session permission checker.

    Attributes:
        deny_tools:    Tool names that are always denied.
        session_allow: Tool names the user approved with "always" this session.
        no_permission: When True, every tool is allowed (--no-permission flag).
    """

    deny_tools: set[str] = field(default_factory=set)
    session_allow: set[str] = field(default_factory=set)
    no_permission: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check(
        self,
        tool: Any,
        params: dict[str, Any],
        tool_context: Any = None,
    ) -> PermissionResult:
        """Return the effective PermissionResult for *tool* with *params*.

        Args:
            tool:         Any object with a ``name`` attribute. Optionally
                          implements ``check_permissions()``, ``is_read_only()``.
            params:       The parameters that will be passed to ``tool.execute()``.
            tool_context: Optional ``ToolContext`` forwarded to tool methods.

        Returns:
            A ``PermissionResult`` with behavior ``"allow"``, ``"deny"``, or
            ``"ask"``.
        """
        # 0. no-permission mode — allow everything unconditionally
        if self.no_permission:
            return PermissionResult("allow")

        tool_name: str = getattr(tool, "name", "") or getattr(tool, "__tool_name__", "")

        # 1. Deny rules (highest priority — overrides session_allow too)
        if tool_name in self.deny_tools:
            return PermissionResult("deny", f"Tool '{tool_name}' is denied")

        # 2. Tool-specific safety check (can deny or allow explicitly)
        tool_perm: PermissionResult | None = None
        if hasattr(tool, "check_permissions"):
            tool_perm = tool.check_permissions(params, tool_context)
            if tool_perm.behavior == "deny":
                return tool_perm
            if tool_perm.behavior == "allow":
                return tool_perm

        # 3. Session always-allow
        if tool_name in self.session_allow:
            return PermissionResult("allow")

        # 4. Read-only auto-allow
        if hasattr(tool, "is_read_only") and tool.is_read_only(params):
            return PermissionResult("allow")

        # 5. Propagate tool-specific "ask" (reuse result from step 2)
        if tool_perm is not None and tool_perm.behavior == "ask":
            return tool_perm

        # 6. Default: ask
        return PermissionResult("ask", f"Allow {tool_name}?")

    def add_always_allow(self, tool_name: str) -> None:
        """Add *tool_name* to the session always-allow set.

        Called when the user responds "a" (always) to a permission prompt.
        """
        self.session_allow.add(tool_name)

    def add_deny(self, tool_name: str) -> None:
        """Add *tool_name* to the permanent deny set for this session."""
        self.deny_tools.add(tool_name)

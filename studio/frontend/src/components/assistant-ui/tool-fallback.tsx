// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"use client";

import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import { Spinner } from "@/components/ui/spinner";
// eslint-disable-next-line no-restricted-imports -- the feature barrel imports this component
import { useChatPreferencesStore } from "@/features/chat/stores/chat-preferences-store";
import { useDetachThreadFromBottom } from "@/components/assistant-ui/use-intent-aware-autoscroll";
import { useCollapseScrollLock } from "@/hooks/use-collapse-scroll-lock";
import {
  formatMcpToolName,
  mcpServerFromProvenance,
  mcpToolFromProvenance,
  splitMcpToolName,
} from "@/features/chat/utils/mcp-tool-name";
// eslint-disable-next-line no-restricted-imports -- the feature barrel imports this component
import { useToolAwaitingApproval } from "@/features/chat/tool-approval";
import { copyToClipboard } from "@/lib/copy-to-clipboard";
import { McpAppFrame } from "@/features/chat/mcp-apps/mcp-app-frame";
import {
  type McpUiToolResult,
  isMcpUiToolResult,
} from "@/features/chat/mcp-apps/mcp-ui";
import { sandboxSessionIdFor } from "@/components/assistant-ui/sandbox-files";
import { useChatProjectScope } from "@/features/chat/chat-project-scope";
import { stripAnsi, stringifyToolResult } from "@/lib/strip-ansi";
import { cn } from "@/lib/utils";
import {
  type ToolCallMessagePartComponent,
  type ToolCallMessagePartStatus,
  useAuiState,
} from "@assistant-ui/react";
import {
  AlertCircleIcon,
  ChevronDownIcon,
  LoaderIcon,
  XCircleIcon,
} from "lucide-react";
import { Tick02Icon } from "@/lib/tick-icon";
import { Copy01Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import {
  type CSSProperties,
  type ComponentProps,
  type ElementType,
  memo,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { IconActionButton } from "./icon-action-button";
import {
  isToolCallCancelled,
  isToolCallRunning,
  toolArgText,
  toolFallbackLabel,
} from "./tool-arg-text";
import {
  syncToolActivityPreference,
  toolActivityOpen,
} from "./tool-activity-open-state";
import { ToolResultOutput } from "./tool-result-output";

const ANIMATION_DURATION = 200;
const COPY_RESET_MS = 2000;

export type ToolFallbackRootProps = Omit<
  ComponentProps<typeof Collapsible>,
  "open" | "onOpenChange"
> & {
  open?: boolean;
  onOpenChange?: (open: boolean) => void;
  defaultOpen?: boolean;
  /**
   * Parked on an allow/deny decision. Pins the card open above `open` and the
   * collapse preference, so what is being approved stays readable. Groups do
   * the same with `hasPendingConfirmation`.
   */
  awaitingApproval?: boolean;
};

function ToolFallbackRoot({
  className,
  open: controlledOpen,
  onOpenChange: controlledOnOpenChange,
  defaultOpen = false,
  awaitingApproval = false,
  children,
  ...props
}: ToolFallbackRootProps) {
  const collapsibleRef = useRef<HTMLDivElement>(null);
  const visibility = useChatPreferencesStore((state) => state.toolVisibility);
  const [uncontrolledState, setUncontrolledState] = useState(() => ({
    visibility,
    active: defaultOpen,
    override: null as boolean | null,
  }));
  const syncedUncontrolledState = syncToolActivityPreference(
    uncontrolledState,
    visibility,
    defaultOpen,
  );
  if (syncedUncontrolledState !== uncontrolledState) {
    setUncontrolledState(syncedUncontrolledState);
  }
  const lockScroll = useCollapseScrollLock(collapsibleRef, ANIMATION_DURATION);

  const isControlled = controlledOpen !== undefined;
  const isOpen =
    awaitingApproval ||
    (isControlled ? controlledOpen : toolActivityOpen(syncedUncontrolledState));

  // Opening by hand grows the card downward; see the same note in reasoning.tsx.
  const detachFromBottom = useDetachThreadFromBottom();
  const messageRunning = useAuiState(
    ({ message }) => message.status?.type === "running",
  );
  const handleOpenChange = useCallback(
    (open: boolean) => {
      if (!open) {
        lockScroll();
      } else if (!messageRunning) {
        detachFromBottom();
      }
      if (!isControlled) {
        setUncontrolledState({ ...syncedUncontrolledState, override: open });
      }
      controlledOnOpenChange?.(open);
    },
    [
      syncedUncontrolledState,
      lockScroll,
      isControlled,
      controlledOnOpenChange,
      detachFromBottom,
      messageRunning,
    ],
  );

  return (
    <Collapsible
      ref={collapsibleRef}
      data-slot="tool-fallback-root"
      open={isOpen}
      onOpenChange={handleOpenChange}
      className={cn(
        "aui-tool-fallback-root group/tool-fallback-root w-full",
        className,
      )}
      style={
        {
          "--animation-duration": `${ANIMATION_DURATION}ms`,
        } as CSSProperties
      }
      {...props}
    >
      {children}
    </Collapsible>
  );
}

type ToolStatus = ToolCallMessagePartStatus["type"];

// The shared app tick is icon data, not a component; wrap it to slot into the
// status map alongside the lucide icons.
function CompleteTickIcon(props: Omit<ComponentProps<typeof HugeiconsIcon>, "icon">) {
  return <HugeiconsIcon icon={Tick02Icon} strokeWidth={2} {...props} />;
}

const statusIconMap: Record<ToolStatus, ElementType> = {
  running: LoaderIcon,
  complete: CompleteTickIcon,
  incomplete: XCircleIcon,
  "requires-action": AlertCircleIcon,
};

function ToolFallbackTrigger({
  toolName,
  mcpServer,
  mcpTool,
  status,
  icon: ToolIcon,
  awaitingApproval = false,
  failed = false,
  className,
  ...props
}: ComponentProps<typeof CollapsibleTrigger> & {
  // Straight off the wire: provider SSE is relayed verbatim, and a non-string
  // name matches nothing in thread.tsx's by_name map, which is exactly why it
  // lands HERE, where formatMcpToolName calls `.startsWith` on it.
  toolName: unknown;
  mcpServer?: string;
  mcpTool?: string;
  status?: ToolCallMessagePartStatus;
  icon?: ElementType;
  /** Parked on Allow/Deny: still "running", but nothing runs until the user answers. */
  awaitingApproval?: boolean;
  /** Completed with an error result, which the status alone reports as success. */
  failed?: boolean;
}) {
  const statusType = status?.type ?? "complete";
  const isRunning = isToolCallRunning(status);
  const isCancelled = isToolCallCancelled(status);
  const isFailed = failed && !isRunning && !isCancelled;

  const StatusIcon = isFailed ? AlertCircleIcon : statusIconMap[statusType];
  const label = awaitingApproval
    ? "Waiting for approval"
    : isFailed
      ? "Tool failed"
      : toolFallbackLabel(status);
  const name = toolArgText(toolName);
  const displayName = formatMcpToolName(name, mcpServer, mcpTool) ?? name;

  return (
    <CollapsibleTrigger
      data-slot="tool-fallback-trigger"
      className={cn(
        // Brightens on hover like the Thinking trigger. The icon inherits this; the label
        // sets its own colour and picks it up through the group below.
        "aui-tool-fallback-trigger group/trigger flex w-full cursor-pointer items-center gap-2 text-sm transition-colors hover:text-foreground",
        className,
      )}
      {...props}
    >
      {isRunning ? (
        <Spinner className="aui-tool-fallback-trigger-icon" />
      ) : ToolIcon && !isFailed ? (
        <ToolIcon
          data-slot="tool-fallback-trigger-icon"
          className={cn(
            "aui-tool-fallback-trigger-icon size-4 shrink-0",
            isCancelled && "text-muted-foreground",
          )}
        />
      ) : (
        <StatusIcon
          data-slot="tool-fallback-trigger-icon"
          className={cn(
            "aui-tool-fallback-trigger-icon size-4 shrink-0",
            isCancelled && "text-muted-foreground",
            isFailed && "text-destructive",
          )}
        />
      )}
      <span
        data-slot="tool-fallback-trigger-label"
        className={cn(
          "aui-tool-fallback-trigger-label-wrapper relative min-w-0 text-left leading-none text-muted-foreground transition-colors",
          // A cancelled row stays muted: the strikethrough is the point, not the name.
          isCancelled
            ? "text-muted-foreground line-through"
            : "group-hover/trigger:text-foreground",
        )}
      >
        <span
          className={cn(
            "block truncate leading-normal",
            "group-data-[state=open]/trigger:overflow-visible group-data-[state=open]/trigger:whitespace-normal group-data-[state=open]/trigger:break-words",
          )}
        >
          {label}:{" "}
          <span className="font-medium">{displayName}</span>
        </span>
        {isRunning && (
          <span
            aria-hidden={true}
            data-slot="tool-fallback-trigger-shimmer"
            className={cn(
              "aui-tool-fallback-trigger-shimmer shimmer pointer-events-none absolute inset-0 block truncate leading-normal motion-reduce:animate-none",
              "group-data-[state=open]/trigger:overflow-visible group-data-[state=open]/trigger:whitespace-normal group-data-[state=open]/trigger:break-words",
            )}
          >
            {label}:{" "}
            <span className="font-medium">{displayName}</span>
          </span>
        )}
      </span>
      <ChevronDownIcon
        data-slot="tool-fallback-trigger-chevron"
        className={cn(
          "aui-tool-fallback-trigger-chevron mr-1 size-3.5 shrink-0 self-center",
          "transition-[transform,opacity] duration-(--animation-duration) ease-out",
          "group-data-[state=closed]/trigger:-rotate-90",
          "group-data-[state=open]/trigger:rotate-0",
        )}
      />
    </CollapsibleTrigger>
  );
}

function ToolFallbackContent({
  className,
  children,
  ...props
}: ComponentProps<typeof CollapsibleContent>) {
  return (
    <CollapsibleContent
      data-slot="tool-fallback-content"
      className={cn(
        "aui-tool-fallback-content relative overflow-hidden text-sm outline-none",
        "group/collapsible-content ease-out",
        "data-[state=closed]:animate-collapsible-up",
        "data-[state=open]:animate-collapsible-down",
        "data-[state=closed]:fill-mode-forwards",
        "data-[state=closed]:pointer-events-none",
        "data-[state=open]:duration-(--animation-duration)",
        "data-[state=closed]:duration-(--animation-duration)",
        className,
      )}
      {...props}
    >
      <div className="mt-2 flex flex-col gap-2 pl-5">{children}</div>
    </CollapsibleContent>
  );
}

// Indented when the arguments parse; otherwise (still streaming, or not JSON) shown as sent.
function prettyToolArgs(argsText: string): string {
  try {
    return JSON.stringify(JSON.parse(argsText), null, 2);
  } catch {
    return argsText;
  }
}

function ToolFallbackArgs({
  argsText,
  className,
  ...props
}: ComponentProps<"div"> & {
  argsText?: string;
}) {
  const prettyArgs = useMemo(
    () => (argsText ? prettyToolArgs(argsText) : ""),
    [argsText],
  );
  if (!argsText) {
    return null;
  }

  return (
    <div
      data-slot="tool-fallback-args"
      className={cn("aui-tool-fallback-args", className)}
      {...props}
    >
      <p className="aui-tool-fallback-args-header font-semibold">Arguments:</p>
      <pre className="aui-tool-fallback-args-value mt-1 whitespace-pre-wrap break-words font-mono text-xs">
        {prettyArgs}
      </pre>
    </div>
  );
}

// CopyBtn in tool-code-cell.tsx draws the same button, but importing it would pull streamdown
// into every card that falls back here.
function ToolFallbackCopyButton({ text }: { text: string }) {
  const [copied, setCopied] = useState(false);
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    return () => {
      if (timer.current) {
        clearTimeout(timer.current);
      }
    };
  }, []);

  const copy = useCallback(async () => {
    if (await copyToClipboard(text)) {
      setCopied(true);
      if (timer.current) {
        clearTimeout(timer.current);
      }
      timer.current = setTimeout(() => setCopied(false), COPY_RESET_MS);
    }
  }, [text]);

  return (
    <IconActionButton label={copied ? "Copied" : "Copy result"} onClick={copy}>
      {copied ? (
        <HugeiconsIcon icon={Tick02Icon} strokeWidth={2} className="size-3" />
      ) : (
        <HugeiconsIcon icon={Copy01Icon} className="size-3" />
      )}
    </IconActionButton>
  );
}

interface McpImageResult {
  text: string;
  images: { data: string; mimeType: string }[];
}

function isMcpImageResult(val: unknown): val is McpImageResult {
  if (typeof val !== "object" || val === null) {
    return false;
  }
  const v = val as { text?: unknown; images?: unknown };
  return (
    typeof v.text === "string" &&
    Array.isArray(v.images) &&
    v.images.length > 0 &&
    v.images.every(
      (img: unknown) =>
        typeof img === "object" &&
        img !== null &&
        typeof (img as { data?: unknown }).data === "string" &&
        typeof (img as { mimeType?: unknown }).mimeType === "string",
    )
  );
}

/** Outside ToolFallbackContent so it stays on screen with the card collapsed. */
function ToolFallbackMcpApp({
  toolName,
  result,
  argsText,
}: {
  toolName: string;
  result: McpUiToolResult;
  argsText?: string;
}) {
  const threadId = useAuiState(({ threadListItem }) => threadListItem.remoteId);
  // The provider's project (the store's lags a thread switch): the adapter keys the run's session on it.
  const projectId = useChatProjectScope();
  const parts = splitMcpToolName(toolName);
  if (!parts) return null;
  return (
    <McpAppFrame
      serverId={parts.serverId}
      toolName={parts.tool}
      ui={result.ui}
      argsText={argsText}
      resultImages={result.images}
      threadId={threadId}
      sessionId={sandboxSessionIdFor(threadId, projectId)}
    />
  );
}

function ToolFallbackResult({
  result,
  failed = false,
  className,
  ...props
}: ComponentProps<"div"> & {
  result?: unknown;
  /** An "Error: ..." result, shown in the error colour. */
  failed?: boolean;
}) {
  if (result === undefined) {
    return null;
  }

  const imageResult = isMcpImageResult(result) ? result : null;
  // Colourised CLIs (ls --color, grep --color, npm, cargo, pytest) emit SGR escapes that a plain
  // <pre> cannot style; strip them so the pane stays readable (#7962).
  const resultText = imageResult
    ? stripAnsi(imageResult.text)
    : stringifyToolResult(result);

  return (
    <div
      data-slot="tool-fallback-result"
      className={cn("aui-tool-fallback-result pt-2", className)}
      {...props}
    >
      <div className="flex items-center justify-between">
        <p className="aui-tool-fallback-result-header font-semibold">Result:</p>
        {resultText ? <ToolFallbackCopyButton text={resultText} /> : null}
      </div>
      {/* Tailed and height-capped like the terminal card; Copy above takes the full text. */}
      {resultText ? (
        <div
          className={cn(
            "aui-tool-fallback-result-content",
            failed && "text-destructive",
          )}
        >
          <ToolResultOutput text={resultText} />
        </div>
      ) : null}
      {imageResult ? (
        <div className="mt-2 flex flex-col gap-2">
          {imageResult.images.map((img, i) => (
            <img
              key={i}
              src={`data:${img.mimeType};base64,${img.data}`}
              alt={`Tool result ${i + 1}`}
              loading="lazy"
              className="max-w-full rounded border border-border"
            />
          ))}
        </div>
      ) : null}
    </div>
  );
}

function ToolFallbackError({
  status,
  className,
  ...props
}: ComponentProps<"div"> & {
  status?: ToolCallMessagePartStatus;
}) {
  if (status?.type !== "incomplete") {
    return null;
  }

  const error = status.error;
  const errorText = error
    ? typeof error === "string"
      ? error
      : JSON.stringify(error)
    : null;

  if (!errorText) {
    return null;
  }

  const isCancelled = status.reason === "cancelled";
  const headerText = isCancelled ? "Cancelled reason:" : "Error:";
  const tone = isCancelled ? "text-muted-foreground" : "text-destructive";

  return (
    <div
      data-slot="tool-fallback-error"
      className={cn("aui-tool-fallback-error", className)}
      {...props}
    >
      <p className={cn("aui-tool-fallback-error-header font-semibold", tone)}>
        {headerText}
      </p>
      <p className={cn("aui-tool-fallback-error-reason", tone)}>{errorText}</p>
    </div>
  );
}

const ToolFallbackImpl: ToolCallMessagePartComponent = ({
  toolCallId,
  toolName,
  argsText,
  result,
  status,
  ...rest
}) => {
  // Allow/Deny confirmation controls are rendered uniformly for every tool
  // card (built-in and fallback) by the `withToolConfirmation` wrapper in
  // thread.tsx; this renderer only reads whether it is parked on them.
  const provenance = (rest as { provenance?: unknown }).provenance;
  const isCancelled = isToolCallCancelled(status);
  // The arguments being approved live inside the content while Allow/Deny
  // render outside it, so a parked call stays open whatever the preference.
  const awaitingApproval = useToolAwaitingApproval(toolCallId);
  // The backend hands MCP failures back as a completed call whose result
  // starts "Error:", so the result is the only place the failure shows.
  const failed = typeof result === "string" && /^\s*Error:/.test(result);
  // A widget result's pane shows the text (and images) the model saw, never its UI seed.
  const widget = isMcpUiToolResult(result, toolName) ? result : null;
  const shown = widget?.images?.length
    ? { text: widget.text, images: widget.images }
    : (widget?.text ?? result);

  return (
    <ToolFallbackRoot
      className={cn(isCancelled && "bg-muted/30")}
      defaultOpen={isToolCallRunning(status)}
      awaitingApproval={awaitingApproval}
    >
      <ToolFallbackTrigger
        toolName={toolName}
        mcpServer={mcpServerFromProvenance(provenance)}
        mcpTool={mcpToolFromProvenance(provenance)}
        status={status}
        awaitingApproval={awaitingApproval}
        failed={failed}
      />
      {!isCancelled && widget && (
        <ToolFallbackMcpApp
          toolName={toolName}
          result={widget}
          argsText={argsText}
        />
      )}
      <ToolFallbackContent>
        <ToolFallbackError status={status} />
        <ToolFallbackArgs
          argsText={argsText}
          className={cn(isCancelled && "opacity-60")}
        />
        {!isCancelled && <ToolFallbackResult result={shown} failed={failed} />}
      </ToolFallbackContent>
    </ToolFallbackRoot>
  );
};

const ToolFallback = memo(
  ToolFallbackImpl,
) as unknown as ToolCallMessagePartComponent & {
  Root: typeof ToolFallbackRoot;
  Trigger: typeof ToolFallbackTrigger;
  Content: typeof ToolFallbackContent;
  Args: typeof ToolFallbackArgs;
  Result: typeof ToolFallbackResult;
  Error: typeof ToolFallbackError;
};

ToolFallback.displayName = "ToolFallback";
ToolFallback.Root = ToolFallbackRoot;
ToolFallback.Trigger = ToolFallbackTrigger;
ToolFallback.Content = ToolFallbackContent;
ToolFallback.Args = ToolFallbackArgs;
ToolFallback.Result = ToolFallbackResult;
ToolFallback.Error = ToolFallbackError;

export {
  ToolFallback,
  ToolFallbackRoot,
  ToolFallbackTrigger,
  ToolFallbackContent,
  ToolFallbackArgs,
  ToolFallbackResult,
  ToolFallbackError,
};

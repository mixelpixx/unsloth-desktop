// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  Delete02Icon,
  Edit03Icon,
  PlusSignIcon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { RefreshCwIcon, UploadIcon } from "lucide-react";
import {
  type ChangeEvent,
  useCallback,
  useEffect,
  useRef,
  useState,
} from "react";
import { toast } from "sonner";

import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Spinner } from "@/components/ui/spinner";
import { Switch } from "@/components/ui/switch";
import { subscribeToMcpServerMutationSettlements } from "./api/mcp-server-mutation-tracker";
import {
  type McpCapabilities,
  type McpServerConfig,
  createMcpServer,
  decodeMcpStdioCommand,
  deleteMcpServer,
  encodeMcpStdioCommand,
  getMcpCapabilities,
  importMcpServers,
  listMcpServers,
  refreshMcpServerTools,
  testMcpServer,
  updateMcpServer,
} from "./api/mcp-servers-api";
import {
  type McpStdioSnapshot,
  createMcpStdioSnapshot,
  resolveMcpStdioUrl,
} from "./mcp-server-form";
import { BlenderMcpSetup } from "./blender-mcp-setup";
import { parseMcpConfigFile } from "./utils/mcp-config-file";

type HeaderRow = { id: string; key: string; value: string };
type ArgumentRow = { id: string; value: string };
type FormTransport = "unknown" | "http" | "stdio";

type FormState = {
  displayName: string;
  url: string;
  transport: FormTransport;
  arguments: ArgumentRow[];
  stdioSnapshot: McpStdioSnapshot | null;
  headers: HeaderRow[];
  credentialTransport: Exclude<FormTransport, "unknown"> | null;
  useOauth: boolean;
  // Local programs only; blank = start in the backend's own working directory.
  cwd: string;
};

const EMPTY_FORM: FormState = {
  displayName: "",
  url: "",
  transport: "unknown",
  arguments: [],
  stdioSnapshot: null,
  headers: [],
  credentialTransport: null,
  useOauth: false,
  cwd: "",
};

// What to send for the working directory: only a local program has one, and blank clears it.
function cwdForTransport(form: FormState): string | null {
  return form.transport === "stdio" ? form.cwd.trim() || null : null;
}

function newRowId(): string {
  return `r_${Math.random().toString(36).slice(2, 10)}`;
}

function headersFromObject(headers: Record<string, string>): HeaderRow[] {
  return Object.entries(headers).map(([k, v]) => ({
    id: newRowId(),
    key: k,
    value: v,
  }));
}

function argumentsFromStrings(arguments_: readonly string[]): ArgumentRow[] {
  return arguments_.map((value) => ({ id: newRowId(), value }));
}

function argumentsToStrings(rows: readonly ArgumentRow[]): string[] {
  return rows.map((row) => row.value);
}

function headersToObject(
  rows: HeaderRow[],
): Record<string, string> | undefined {
  const out: Record<string, string> = {};
  for (const row of rows) {
    const key = row.key.trim();
    if (!key) continue;
    out[key] = row.value;
  }
  return Object.keys(out).length > 0 ? out : undefined;
}

// A non-HTTP address is a local stdio command. Case-insensitive to match the backend's is_stdio(),
// so all layers split http-vs-command identically.
function isHttpAddress(value: string): boolean {
  const trimmed = value.trim().toLowerCase();
  return trimmed.startsWith("http://") || trimmed.startsWith("https://");
}

function transportFromAddress(
  value: string,
  credentialTransport: FormState["credentialTransport"] = null,
  typing = false,
): FormTransport {
  const trimmed = value.trim().toLowerCase();
  if (!trimmed) {
    return "unknown";
  }
  if (isHttpAddress(value)) {
    return "http";
  }
  // Mid-keystroke "h", "htt", "https:/" is a URL being typed: flipping to the local-program form (and
  // its "local programs are off" alert) for a few keystrokes reads as a glitch. Blur still resolves it.
  if (
    (typing || credentialTransport === "http") &&
    ("http://".startsWith(trimmed) || "https://".startsWith(trimmed))
  ) {
    return "unknown";
  }
  return "stdio";
}

function formWithAddress(
  form: FormState,
  url: string,
  preservePartialHttp: boolean,
): FormState {
  const transport = transportFromAddress(
    url,
    preservePartialHttp ? form.credentialTransport : null,
    preservePartialHttp,
  );
  const nextCredentialTransport =
    transport === "unknown" ? form.credentialTransport : transport;
  const transportChanged =
    transport !== "unknown" &&
    form.credentialTransport !== null &&
    form.credentialTransport !== transport;
  return {
    ...form,
    url,
    transport,
    headers: transportChanged ? [] : form.headers,
    credentialTransport: nextCredentialTransport,
    useOauth: transport === "stdio" ? false : form.useOauth,
  };
}

function isValidAddress(value: string): boolean {
  const trimmed = value.trim();
  if (!trimmed) return false;
  if (isHttpAddress(trimmed)) {
    try {
      const parsed = new URL(trimmed);
      return parsed.protocol === "http:" || parsed.protocol === "https:";
    } catch {
      return false;
    }
  }
  // The backend owns stdio parsing and validation. In particular, the browser must not split an
  // executable or duplicate platform-specific quoting rules.
  return true;
}

// A remote URL can carry an API key in its query string or userinfo; the list only needs to say which
// server a row is, so it shows origin + path and marks a hidden query.
function displayAddress(url: string): string {
  if (!isHttpAddress(url)) return url;
  try {
    const parsed = new URL(url.trim());
    return `${parsed.origin}${parsed.pathname}${parsed.search ? "?…" : ""}`;
  } catch {
    return url;
  }
}

const KNOWN_LAUNCHERS = new Set([
  "npx", "npm", "pnpm", "bunx", "bun", "node", "deno", "uvx", "uv", "pipx",
  "python", "python3", "py", "docker", "podman", "dotnet", "java", "go", "cargo",
]);

// Only gates whether a pasted command line is split into program + arguments (by the backend's own
// parser). A wrong "no" leaves the line as one executable, exactly as before, so stay conservative: an
// unquoted "C:\Program Files\..." splits to "C:\Program", which is not a program, and stays whole.
function looksLikeProgram(token: string): boolean {
  const name = (token.split(/[\\/]/).pop() ?? token).toLowerCase();
  if (/\.(exe|cmd|bat|ps1|py|js|mjs|cjs|sh)$/.test(name)) return true;
  return KNOWN_LAUNCHERS.has(name);
}

function ArgumentsEditor({
  rows,
  onChange,
  disabled,
}: {
  rows: ArgumentRow[];
  onChange: (rows: ArgumentRow[]) => void;
  disabled: boolean;
}) {
  const update = (id: string, value: string) =>
    onChange(rows.map((row) => (row.id === id ? { ...row, value } : row)));
  const add = () => onChange([...rows, { id: newRowId(), value: "" }]);
  const remove = (id: string) => onChange(rows.filter((row) => row.id !== id));

  return (
    <div className="grid gap-2">
      <div className="flex items-center justify-between">
        <Label className="text-sm">Arguments</Label>
        <Button
          type="button"
          variant="ghost"
          size="sm"
          onClick={add}
          disabled={disabled}
        >
          <HugeiconsIcon icon={PlusSignIcon} className="size-3.5" />
          Add argument
        </Button>
      </div>
      {rows.length === 0 ? (
        <div className="text-xs text-muted-foreground">
          Optional. Each row is one argument; row order is preserved.
        </div>
      ) : (
        <div className="flex flex-col gap-2">
          {rows.map((row, index) => (
            <div key={row.id} className="flex items-center gap-2">
              <Input
                data-reload-snapshot-sensitive={true}
                value={row.value}
                disabled={disabled}
                aria-label={`Argument ${index + 1}`}
                placeholder={index === 0 ? "e.g. -y" : undefined}
                onChange={(event) => update(row.id, event.target.value)}
              />
              <Button
                type="button"
                variant="ghost"
                size="icon"
                onClick={() => remove(row.id)}
                disabled={disabled}
                aria-label={`Remove argument ${index + 1}`}
              >
                <HugeiconsIcon icon={Delete02Icon} className="size-3.5" />
              </Button>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function HeadersEditor({
  rows,
  onChange,
  stdio,
  disabled,
}: {
  rows: HeaderRow[];
  onChange: (rows: HeaderRow[]) => void;
  // stdio servers reuse this editor for environment variables instead of headers.
  stdio: boolean;
  disabled: boolean;
}) {
  const update = (id: string, patch: Partial<HeaderRow>) =>
    onChange(rows.map((row) => (row.id === id ? { ...row, ...patch } : row)));
  const add = () => onChange([...rows, { id: newRowId(), key: "", value: "" }]);
  const remove = (id: string) => onChange(rows.filter((row) => row.id !== id));

  const copy = stdio
    ? {
        label: "Environment variables",
        add: "Add variable",
        keyPlaceholder: "Variable name",
        valuePlaceholder: "Variable value",
        remove: "Remove variable",
      }
    : {
        label: "Custom headers",
        add: "Add header",
        keyPlaceholder: "Header name",
        valuePlaceholder: "Header value",
        remove: "Remove header",
      };

  return (
    <>
      <div className="flex items-center justify-between">
        <Label className="text-sm">{copy.label}</Label>
        <Button
          type="button"
          variant="ghost"
          size="sm"
          onClick={add}
          disabled={disabled}
        >
          <HugeiconsIcon icon={PlusSignIcon} className="size-3.5" />
          {copy.add}
        </Button>
      </div>
      {rows.length === 0 ? (
        <div className="text-xs text-muted-foreground">
          {stdio ? (
            "Optional. Environment variables passed to the server process."
          ) : (
            <>
              Optional. Add an <code>Authorization</code> header here for
              servers that require auth.
            </>
          )}
        </div>
      ) : (
        <div className="flex flex-col gap-2">
          {rows.map((row, index) => (
            <div key={row.id} className="flex items-center gap-2">
              <Input
                value={row.key}
                disabled={disabled}
                placeholder={copy.keyPlaceholder}
                aria-label={`${copy.keyPlaceholder} ${index + 1}`}
                onChange={(e) => update(row.id, { key: e.target.value })}
              />
              <Input
                data-reload-snapshot-sensitive={true}
                value={row.value}
                disabled={disabled}
                placeholder={copy.valuePlaceholder}
                aria-label={`${copy.valuePlaceholder} ${index + 1}`}
                onChange={(e) => update(row.id, { value: e.target.value })}
              />
              <Button
                type="button"
                variant="ghost"
                size="icon"
                onClick={() => remove(row.id)}
                disabled={disabled}
                aria-label={copy.remove}
              >
                <HugeiconsIcon icon={Delete02Icon} className="size-3.5" />
              </Button>
            </div>
          ))}
        </div>
      )}
    </>
  );
}

export interface ChatMcpServersDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

type View =
  | { kind: "list" }
  | { kind: "create" }
  | { kind: "edit"; id: string };

export function ChatMcpServersDialog({
  open,
  onOpenChange,
}: ChatMcpServersDialogProps) {
  const [servers, setServers] = useState<McpServerConfig[]>([]);
  const [loading, setLoading] = useState(false);
  const [blenderBusy, setBlenderBusy] = useState(false);
  const [view, setView] = useState<View>({ kind: "list" });
  const [form, setForm] = useState<FormState>(EMPTY_FORM);
  const [saving, setSaving] = useState(false);
  const [testing, setTesting] = useState(false);
  const [codecPending, setCodecPending] = useState(false);
  const [decodingCommand, setDecodingCommand] = useState(false);
  const [codecError, setCodecError] = useState<string | null>(null);
  const [importing, setImporting] = useState(false);
  const [capabilities, setCapabilities] = useState<McpCapabilities | null>(
    null,
  );
  const [testResult, setTestResult] = useState<{
    ok: boolean;
    text: string;
  } | null>(null);
  const [importReport, setImportReport] = useState<{
    added: string[];
    skipped: string[];
    errors: string[];
  } | null>(null);
  const [refreshingIds, setRefreshingIds] = useState<ReadonlySet<string>>(
    () => new Set(),
  );
  const [togglingIds, setTogglingIds] = useState<ReadonlySet<string>>(
    () => new Set(),
  );
  const [busyIds, setBusyIds] = useState<ReadonlySet<string>>(() => new Set());
  const [confirmingDelete, setConfirmingDelete] =
    useState<McpServerConfig | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const formGenerationRef = useRef(0);
  const actionGenerationRef = useRef(0);
  const importGenerationRef = useRef(0);
  const activeEditIdRef = useRef<string | null>(null);
  const listRefreshGenerationRef = useRef(0);
  const openRef = useRef(open);
  const refreshingIdsRef = useRef(new Set<string>());
  const togglingIdsRef = useRef(new Set<string>());
  const busyIdsRef = useRef(new Set<string>());
  const importingRef = useRef(false);
  const latestFormRef = useRef(form);
  openRef.current = open;

  useEffect(() => {
    latestFormRef.current = form;
  }, [form]);

  useEffect(() => {
    return () => {
      formGenerationRef.current += 1;
      actionGenerationRef.current += 1;
      activeEditIdRef.current = null;
      openRef.current = false;
    };
  }, []);

  const refresh = useCallback(
    async (waitForPendingMutations = true, minimumMutationEpoch = 0) => {
      const generation = listRefreshGenerationRef.current + 1;
      listRefreshGenerationRef.current = generation;
      setLoading(true);
      try {
        const rows = await listMcpServers({
          waitForPendingMutations,
          minimumMutationEpoch,
        });
        if (listRefreshGenerationRef.current !== generation || !openRef.current)
          return;
        setServers((current) =>
          rows.map((row) => {
            if (!togglingIdsRef.current.has(row.id)) return row;
            const optimistic = current.find(
              (candidate) => candidate.id === row.id,
            );
            return optimistic
              ? { ...row, is_enabled: optimistic.is_enabled }
              : row;
          }),
        );
      } catch (err) {
        if (listRefreshGenerationRef.current !== generation || !openRef.current)
          return;
        toast.error("Failed to load MCP servers", {
          description: err instanceof Error ? err.message : String(err),
        });
      } finally {
        if (listRefreshGenerationRef.current === generation && openRef.current)
          setLoading(false);
      }
    },
    [],
  );

  useEffect(() => {
    if (!open) {
      listRefreshGenerationRef.current += 1;
      return;
    }
    const unsubscribe = subscribeToMcpServerMutationSettlements((epoch) => {
      void refresh(false, epoch);
    });
    return () => {
      unsubscribe();
      listRefreshGenerationRef.current += 1;
    };
  }, [open, refresh]);

  useEffect(() => {
    formGenerationRef.current += 1;
    actionGenerationRef.current += 1;
    activeEditIdRef.current = null;
    let cancelled = false;
    queueMicrotask(() => {
      if (cancelled) return;
      setSaving(false);
      setTesting(false);
      setCodecPending(false);
      setDecodingCommand(false);
      setCodecError(null);
      setTestResult(null);
      setImportReport(null);
      setImporting(importingRef.current);
      setConfirmingDelete(null);
      setRefreshingIds(new Set(refreshingIdsRef.current));
      setTogglingIds(new Set(togglingIdsRef.current));
      setBusyIds(new Set(busyIdsRef.current));
      if (!open) return;
      void refresh();
      // Reset to the list on each open, else a stale create/edit view persists.
      setView({ kind: "list" });
      setForm(EMPTY_FORM);
    });
    return () => {
      cancelled = true;
    };
  }, [open, refresh]);

  useEffect(() => {
    if (!open) return;
    let cancelled = false;
    getMcpCapabilities().then(
      (next) => {
        if (!cancelled) setCapabilities(next);
      },
      // An older backend has no /capabilities: keep the form usable and let Save report the gate.
      () => {
        if (!cancelled) setCapabilities(null);
      },
    );
    return () => {
      cancelled = true;
    };
  }, [open]);

  function startCreate() {
    formGenerationRef.current += 1;
    activeEditIdRef.current = null;
    setSaving(false);
    setTesting(false);
    setCodecPending(false);
    setDecodingCommand(false);
    setCodecError(null);
    setTestResult(null);
    setImportReport(null);
    setView({ kind: "create" });
    setForm(EMPTY_FORM);
  }

  async function startEdit(server: McpServerConfig) {
    if (blenderBusy) return;
    const generation = formGenerationRef.current + 1;
    formGenerationRef.current = generation;
    activeEditIdRef.current = server.id;
    setSaving(false);
    setTesting(false);
    setCodecError(null);
    setTestResult(null);
    setView({ kind: "edit", id: server.id });
    const baseForm: FormState = {
      displayName: server.display_name,
      url: server.url,
      transport: isHttpAddress(server.url) ? "http" : "stdio",
      arguments: [],
      stdioSnapshot: null,
      headers: headersFromObject(server.headers ?? {}),
      credentialTransport: isHttpAddress(server.url) ? "http" : "stdio",
      useOauth: server.use_oauth ?? false,
      cwd: server.cwd ?? "",
    };

    if (isHttpAddress(server.url)) {
      setCodecPending(false);
      setDecodingCommand(false);
      setForm(baseForm);
      return;
    }

    setCodecPending(true);
    setDecodingCommand(true);
    setForm(baseForm);
    try {
      const decoded = await decodeMcpStdioCommand(server.url);
      if (
        formGenerationRef.current !== generation ||
        activeEditIdRef.current !== server.id
      ) {
        return;
      }
      setForm({
        ...baseForm,
        url: decoded.command,
        arguments: argumentsFromStrings(decoded.arguments ?? []),
        stdioSnapshot: createMcpStdioSnapshot(
          server.url,
          decoded.command,
          decoded.arguments ?? [],
        ),
        useOauth: false,
      });
    } catch (err) {
      if (
        formGenerationRef.current !== generation ||
        activeEditIdRef.current !== server.id
      ) {
        return;
      }
      const message = err instanceof Error ? err.message : String(err);
      setCodecError(message);
      toast.error("Failed to read local command", { description: message });
    } finally {
      if (
        formGenerationRef.current === generation &&
        activeEditIdRef.current === server.id
      ) {
        setCodecPending(false);
        setDecodingCommand(false);
      }
    }
  }

  function cancelForm() {
    formGenerationRef.current += 1;
    activeEditIdRef.current = null;
    setSaving(false);
    setTesting(false);
    setCodecPending(false);
    setDecodingCommand(false);
    setCodecError(null);
    setTestResult(null);
    setView({ kind: "list" });
    setForm(EMPTY_FORM);
  }

  function handleOpenChange(next: boolean) {
    // once crud starts, dismissal must wait for the authoritative refresh
    if (!next && (blenderBusy || (saving && !codecPending) || busyIdsRef.current.size > 0))
      return;
    if (!next) {
      formGenerationRef.current += 1;
      actionGenerationRef.current += 1;
      activeEditIdRef.current = null;
      setSaving(false);
      setTesting(false);
      setCodecPending(false);
      setDecodingCommand(false);
      setCodecError(null);
      setConfirmingDelete(null);
    }
    onOpenChange(next);
  }

  async function encodeStdioForGeneration(
    generation: number,
    command: string,
    arguments_: string[],
  ): Promise<string | null> {
    setCodecPending(true);
    try {
      const encoded = await encodeMcpStdioCommand({
        command,
        arguments: arguments_,
      });
      if (formGenerationRef.current !== generation) return null;
      return encoded.url;
    } finally {
      if (formGenerationRef.current === generation) setCodecPending(false);
    }
  }

  async function testConnection() {
    if (!form.url.trim() || form.transport === "unknown") {
      toast.error("Enter an http(s):// URL or a local command first");
      return;
    }
    const stdio = form.transport === "stdio";
    if (!stdio && !isValidAddress(form.url)) {
      toast.error("Enter an http(s):// URL or a local command first");
      return;
    }
    const generation = formGenerationRef.current;
    setTesting(true);
    setTestResult(null);
    try {
      const url = stdio
        ? await encodeStdioForGeneration(
            generation,
            form.url,
            argumentsToStrings(form.arguments),
          )
        : form.url.trim();
      if (url === null || formGenerationRef.current !== generation) return;
      const result = await testMcpServer({
        url,
        headers: headersToObject(form.headers),
        useOauth: stdio ? false : form.useOauth,
        cwd: cwdForTransport(form),
      });
      if (formGenerationRef.current !== generation) return;
      // Inline rather than a 5 s toast: the result is what the user acts on next (fix the path, add an
      // argument), and a server listing no tools almost always means it started with the wrong args.
      if (result.ok && result.tool_count > 0) {
        setTestResult({
          ok: true,
          text: `Connected: ${result.tool_count} tool${result.tool_count === 1 ? "" : "s"} available.`,
        });
      } else if (result.ok) {
        setTestResult({
          ok: false,
          text: "Connected, but the server lists no tools. Check its arguments and environment variables.",
        });
      } else {
        setTestResult({
          ok: false,
          text: `Connection failed: ${result.error ?? "Unknown error"}`,
        });
      }
    } catch (err) {
      if (formGenerationRef.current !== generation) return;
      setTestResult({
        ok: false,
        text: `Connection test failed: ${err instanceof Error ? err.message : String(err)}`,
      });
    } finally {
      if (formGenerationRef.current === generation) setTesting(false);
    }
  }

  async function submitForm() {
    const trimmedName = form.displayName.trim();
    if (!trimmedName) {
      toast.error("Display name is required");
      return;
    }
    if (!form.url.trim() || form.transport === "unknown") {
      toast.error("URL or command is required");
      return;
    }
    const stdio = form.transport === "stdio";
    if (!stdio && !isValidAddress(form.url)) {
      toast.error("Enter an http(s):// URL or a local command");
      return;
    }
    const generation = formGenerationRef.current;
    setSaving(true);
    try {
      const headers = headersToObject(form.headers);
      let url: string | undefined;
      if (stdio) {
        const decision = resolveMcpStdioUrl(
          form.url,
          argumentsToStrings(form.arguments),
          form.stdioSnapshot,
        );
        if (decision.kind === "reuse") {
          url = view.kind === "edit" ? undefined : decision.url;
        } else {
          const encodedUrl = await encodeStdioForGeneration(
            generation,
            decision.command,
            decision.arguments,
          );
          if (encodedUrl === null) return;
          url = encodedUrl;
        }
      } else {
        url = form.url.trim();
      }
      if (formGenerationRef.current !== generation) return;
      if (view.kind === "edit") {
        await updateMcpServer(view.id, {
          displayName: trimmedName,
          url,
          headers: headers ?? null,
          useOauth: stdio ? false : form.useOauth,
          // Omitted for http: switching a row to a URL clears its working directory on the backend.
          cwd: stdio ? cwdForTransport(form) : undefined,
        });
        if (formGenerationRef.current !== generation) return;
        toast.success("MCP server updated");
      } else {
        if (url === undefined) return;
        await createMcpServer({
          displayName: trimmedName,
          url,
          headers: headers,
          useOauth: stdio ? false : form.useOauth,
          cwd: cwdForTransport(form),
        });
        if (formGenerationRef.current !== generation) return;
        toast.success("MCP server added");
      }
      if (formGenerationRef.current !== generation) return;
      cancelForm();
    } catch (err) {
      if (formGenerationRef.current !== generation) return;
      toast.error("Save failed", {
        description: err instanceof Error ? err.message : String(err),
      });
    } finally {
      if (formGenerationRef.current === generation) setSaving(false);
    }
  }

  // READMEs and Explorer's "Copy as path" hand out whole command lines ("npx -y @scope/server",
  // "\"C:\\Program Files\\x\\server.exe\" --stdio"), but the executable field is argv[0] alone and the
  // whole line would be run as one program name. Split with the backend's own parser (the browser must
  // not duplicate platform quoting rules) and only when the first token is recognizably a program.
  async function splitPastedCommand() {
    const value = form.url;
    if (
      form.transport !== "stdio" ||
      form.arguments.length > 0 ||
      !/\s|^\s*["']/.test(value)
    ) {
      return;
    }
    const generation = formGenerationRef.current;
    let decoded;
    try {
      decoded = await decodeMcpStdioCommand(value);
    } catch {
      return; // Unbalanced quotes etc. are reported by Test/Save with the backend's message.
    }
    // The user may have kept typing while the decode was in flight.
    const latest = latestFormRef.current;
    if (
      formGenerationRef.current !== generation ||
      latest.url !== value ||
      latest.arguments.length > 0
    ) {
      return;
    }
    const { command, arguments: rest } = decoded;
    if (rest.length > 0 && !looksLikeProgram(command)) return;
    setForm((prev) =>
      prev.url !== value || prev.arguments.length > 0
        ? prev
        : { ...prev, url: command, arguments: argumentsFromStrings(rest) },
    );
    if (rest.length > 0) {
      toast.info(
        `Split into the program and ${rest.length} argument${rest.length === 1 ? "" : "s"}`,
      );
    }
  }

  async function onImportFile(e: ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    e.target.value = ""; // let the user re-pick the same file later
    if (!file || importingRef.current) return;
    const generation = actionGenerationRef.current;
    const importGeneration = importGenerationRef.current + 1;
    importGenerationRef.current = importGeneration;
    importingRef.current = true;
    setImporting(true);
    try {
      let config: unknown;
      try {
        config = parseMcpConfigFile(await file.text());
      } catch (err) {
        if (actionGenerationRef.current === generation && openRef.current)
          toast.error("Couldn't read that file as JSON", {
            description: err instanceof Error ? err.message : String(err),
          });
        return;
      }
      if (actionGenerationRef.current !== generation || !openRef.current)
        return;
      const result = await importMcpServers(config);
      if (actionGenerationRef.current !== generation || !openRef.current)
        return;
      // Every name and every error, kept on screen: a 5 s toast showing five of twelve errors left
      // users guessing which servers made it in.
      cancelForm();
      setImportReport({
        added: result.created.map((server) => server.display_name),
        skipped: result.skipped,
        errors: result.errors,
      });
    } catch (err) {
      if (actionGenerationRef.current !== generation || !openRef.current)
        return;
      toast.error("Import failed", {
        description: err instanceof Error ? err.message : String(err),
      });
    } finally {
      if (importGenerationRef.current === importGeneration) {
        importingRef.current = false;
        if (openRef.current) setImporting(false);
      }
    }
  }

  async function removeServer(server: McpServerConfig) {
    if (busyIdsRef.current.has(server.id)) return;
    const generation = actionGenerationRef.current;
    busyIdsRef.current.add(server.id);
    setBusyIds(new Set(busyIdsRef.current));
    try {
      await deleteMcpServer(server.id);
      if (actionGenerationRef.current !== generation || !openRef.current)
        return;
      setServers((rows) => rows.filter((row) => row.id !== server.id));
    } catch (err) {
      if (actionGenerationRef.current !== generation || !openRef.current)
        return;
      toast.error("Delete failed", {
        description: err instanceof Error ? err.message : String(err),
      });
    } finally {
      busyIdsRef.current.delete(server.id);
      if (openRef.current) setBusyIds(new Set(busyIdsRef.current));
    }
  }

  async function toggleEnabled(server: McpServerConfig, next: boolean) {
    if (busyIdsRef.current.has(server.id)) return;
    const generation = actionGenerationRef.current;
    busyIdsRef.current.add(server.id);
    togglingIdsRef.current.add(server.id);
    setBusyIds(new Set(busyIdsRef.current));
    setTogglingIds(new Set(togglingIdsRef.current));
    // Optimistic update so the switch doesn't snap back during the round-trip.
    setServers((rows) =>
      rows.map((row) =>
        row.id === server.id ? { ...row, is_enabled: next } : row,
      ),
    );
    try {
      await updateMcpServer(server.id, { isEnabled: next });
    } catch (err) {
      if (actionGenerationRef.current !== generation || !openRef.current)
        return;
      setServers((rows) =>
        rows.map((row) =>
          row.id === server.id ? { ...row, is_enabled: !next } : row,
        ),
      );
      toast.error("Update failed", {
        description: err instanceof Error ? err.message : String(err),
      });
    } finally {
      busyIdsRef.current.delete(server.id);
      togglingIdsRef.current.delete(server.id);
      if (openRef.current) {
        setBusyIds(new Set(busyIdsRef.current));
        setTogglingIds(new Set(togglingIdsRef.current));
      }
    }
  }

  async function refreshTools(server: McpServerConfig) {
    if (busyIdsRef.current.has(server.id)) return;
    const generation = actionGenerationRef.current;
    busyIdsRef.current.add(server.id);
    refreshingIdsRef.current.add(server.id);
    setBusyIds(new Set(busyIdsRef.current));
    setRefreshingIds(new Set(refreshingIdsRef.current));
    try {
      const result = await refreshMcpServerTools(server.id);
      if (actionGenerationRef.current !== generation || !openRef.current)
        return;
      if (result.ok) {
        toast.success(
          `Refreshed "${server.display_name}" (${result.tool_count} tool${result.tool_count === 1 ? "" : "s"})`,
        );
      } else {
        toast.error(`Refresh failed for "${server.display_name}"`, {
          description: result.error ?? "Unknown error",
        });
      }
    } catch (err) {
      if (actionGenerationRef.current !== generation || !openRef.current)
        return;
      toast.error("Refresh failed", {
        description: err instanceof Error ? err.message : String(err),
      });
    } finally {
      busyIdsRef.current.delete(server.id);
      refreshingIdsRef.current.delete(server.id);
      if (openRef.current) {
        setBusyIds(new Set(busyIdsRef.current));
        setRefreshingIds(new Set(refreshingIdsRef.current));
      }
    }
  }

  const showForm = view.kind !== "list";
  const formPending = importing || codecPending || testing || saving;
  // A local stdio command uses env vars, not headers or OAuth.
  const addressIsCommand = form.transport === "stdio";
  // The backend refuses to test or add a local program while its gate is closed (e.g. a -H 0.0.0.0
  // bind); say why up front rather than after Save. Editing the name/env of a saved one still works.
  const stdioBlocked = capabilities !== null && !capabilities.stdio_enabled;
  const stdioBlockedForCommand = stdioBlocked && addressIsCommand;

  return (
    <Dialog open={open} onOpenChange={handleOpenChange}>
      <DialogContent
        className="max-w-2xl max-h-[85dvh] overflow-y-auto"
        showCloseButton={!blenderBusy && !(saving && !codecPending) && busyIds.size === 0}
        aria-busy={decodingCommand}
      >
        <DialogHeader>
          <DialogTitle>MCP Servers</DialogTitle>
          <DialogDescription>
            Connect remote servers by URL, or local programs (an .exe, npx,
            uvx…) that speak MCP over stdio.
          </DialogDescription>
        </DialogHeader>
        <input
          ref={fileInputRef}
          type="file"
          accept="application/json,.json"
          className="hidden"
          onChange={onImportFile}
          disabled={importing || formPending}
        />

        {showForm ? (
          <div className="flex flex-col gap-4">
            {view.kind === "create" && (
              <div className="flex items-center justify-between gap-3 rounded-md border border-dashed px-3 py-2">
                <span className="text-xs text-muted-foreground">
                  Import servers from a config file.
                </span>
                <Button
                  type="button"
                  size="sm"
                  variant="outline"
                  className="shrink-0"
                  onClick={() => fileInputRef.current?.click()}
                  disabled={importing || formPending}
                  title="Import servers from a mcpServers JSON config (Claude Desktop, Cursor, VS Code…)"
                >
                  {importing ? <Spinner /> : <UploadIcon className="size-3.5" />}
                  Import config
                </Button>
              </div>
            )}
            <div className="grid gap-2">
              <Label htmlFor="mcp-display-name">Display name</Label>
              <Input
                id="mcp-display-name"
                autoFocus
                value={form.displayName}
                disabled={formPending}
                onChange={(e) =>
                  setForm((prev) => ({ ...prev, displayName: e.target.value }))
                }
                placeholder="e.g. GitHub MCP"
              />
            </div>
            <div className="grid gap-2">
              <Label htmlFor="mcp-url">
                {addressIsCommand
                  ? "Executable"
                  : form.transport === "http"
                    ? "URL"
                    : "URL or executable"}
              </Label>
              <Input
                id="mcp-url"
                value={form.url}
                disabled={formPending}
                onChange={(e) => {
                  const url = e.target.value;
                  setCodecError(null);
                  setTestResult(null);
                  setForm((prev) => formWithAddress(prev, url, true));
                }}
                onBlur={() => {
                  setForm((prev) =>
                    prev.transport === "unknown" && prev.url.trim()
                      ? formWithAddress(prev, prev.url, false)
                      : prev,
                  );
                  void splitPastedCommand();
                }}
                placeholder={
                  addressIsCommand
                    ? "e.g. npx or C:\\path\\to\\server.exe"
                    : form.transport === "http"
                      ? "https://example.com/mcp"
                      : "https://example.com/mcp or npx, uvx, C:\\path\\to\\server.exe"
                }
              />
              <span className="text-xs text-muted-foreground">
                {addressIsCommand
                  ? "The program for a local stdio server (an .exe, npx, uvx…). Add each argument in an Arguments row below, or paste a whole command line and it will be split for you."
                  : form.transport === "http"
                    ? "An http(s) URL for a remote server."
                    : "An http(s) URL for a remote server, or a local program (an .exe, npx, uvx…) for stdio. Add local arguments in the Arguments rows."}
              </span>
              {stdioBlocked && form.transport !== "http" && (
                <Alert
                  variant={addressIsCommand ? "destructive" : "default"}
                  role={addressIsCommand ? "alert" : "note"}
                >
                  <AlertTitle>Local programs are turned off</AlertTitle>
                  <AlertDescription>
                    {capabilities?.stdio_disabled_reason ??
                      "Only http(s) MCP servers can be added on this server."}
                  </AlertDescription>
                </Alert>
              )}
              {decodingCommand && (
                <span
                  role="status"
                  aria-live="polite"
                  className="flex items-center gap-2 text-xs text-muted-foreground"
                >
                  <Spinner />
                  Reading local command…
                </span>
              )}
            </div>

            {addressIsCommand && (
              <ArgumentsEditor
                rows={form.arguments}
                disabled={formPending}
                onChange={(arguments_) =>
                  setForm((prev) => ({ ...prev, arguments: arguments_ }))
                }
              />
            )}

            {addressIsCommand && (
              <div className="grid gap-2">
                <Label htmlFor="mcp-cwd">Working directory</Label>
                <Input
                  id="mcp-cwd"
                  data-reload-snapshot-sensitive={true}
                  value={form.cwd}
                  disabled={formPending}
                  onChange={(e) => {
                    const cwd = e.target.value;
                    setTestResult(null);
                    setForm((prev) => ({ ...prev, cwd }));
                  }}
                  placeholder="e.g. C:\path\to\server-folder"
                />
                <span className="text-xs text-muted-foreground">
                  Optional. The folder the program starts in, for servers that
                  read config or data files next to themselves. Must be the full
                  path to an existing folder.
                </span>
              </div>
            )}

            {codecError && (
              <div className="flex items-center justify-between gap-3">
                <div
                  role="alert"
                  aria-live="assertive"
                  className="text-sm text-destructive"
                >
                  {codecError}
                </div>
                {view.kind === "edit" && (
                  <Button
                    type="button"
                    size="sm"
                    variant="outline"
                    disabled={formPending}
                    onClick={() => {
                      const server = servers.find(
                        (candidate) => candidate.id === view.id,
                      );
                      if (server) void startEdit(server);
                    }}
                  >
                    Retry
                  </Button>
                )}
              </div>
            )}

            {form.transport === "http" && (
              <div className="flex items-start justify-between gap-3">
                <div className="flex flex-col gap-0.5">
                  <Label className="text-sm" htmlFor="mcp-oauth">
                    Use OAuth sign-in
                  </Label>
                  <span className="text-xs text-muted-foreground">
                    For servers that require browser-based authentication
                    (GitHub, Linear, etc.). A browser window will open on first
                    connect.
                  </span>
                </div>
                <Switch
                  id="mcp-oauth"
                  checked={form.useOauth}
                  disabled={formPending}
                  onCheckedChange={(useOauth) =>
                    setForm((prev) => ({ ...prev, useOauth }))
                  }
                />
              </div>
            )}

            {form.transport !== "unknown" && (
              <HeadersEditor
                rows={form.headers}
                onChange={(headers) =>
                  setForm((prev) => ({ ...prev, headers }))
                }
                stdio={addressIsCommand}
                disabled={formPending}
              />
            )}

            {testResult && (
              <p
                role="status"
                aria-live="polite"
                className={
                  testResult.ok
                    ? "text-xs text-emerald-600 dark:text-emerald-400"
                    : "whitespace-pre-wrap break-words text-xs text-destructive"
                }
              >
                {testResult.text}
              </p>
            )}

            <div className="flex items-center justify-between gap-2 pt-2">
              <Button
                type="button"
                variant="outline"
                size="sm"
                onClick={testConnection}
                disabled={
                  formPending ||
                  codecError !== null ||
                  form.transport === "unknown" ||
                  !form.url.trim() ||
                  stdioBlockedForCommand
                }
              >
                {testing ? <Spinner /> : null}
                Test connection
              </Button>
              <div className="flex gap-2">
                <Button
                  variant="ghost"
                  onClick={cancelForm}
                  disabled={saving && !codecPending}
                >
                  Cancel
                </Button>
                <Button
                  onClick={submitForm}
                  disabled={
                    formPending ||
                    codecError !== null ||
                    form.transport === "unknown" ||
                    (stdioBlockedForCommand && view.kind === "create")
                  }
                >
                  {saving ? <Spinner /> : null}
                  {view.kind === "edit" ? "Save changes" : "Add server"}
                </Button>
              </div>
            </div>
          </div>
        ) : (
          <div className="flex min-w-0 flex-col gap-3">
            {open && <BlenderMcpSetup servers={servers} disabled={importing} onBusyChange={setBlenderBusy} />}
            <div className="flex justify-end gap-2">
              <Button
                size="sm"
                variant="outline"
                onClick={() => fileInputRef.current?.click()}
                disabled={importing || blenderBusy}
                title="Import servers from a mcpServers JSON config (Claude Desktop, Cursor, VS Code…)"
              >
                {importing ? <Spinner /> : <UploadIcon className="size-3.5" />}
                Import config
              </Button>
              <Button size="sm" onClick={startCreate} disabled={importing || blenderBusy}>
                <HugeiconsIcon icon={PlusSignIcon} className="size-3.5" />
                Add server
              </Button>
            </div>
            {importReport && (
              <Alert
                variant={importReport.errors.length ? "destructive" : "default"}
                role="status"
              >
                <AlertTitle>
                  Imported {importReport.added.length} server
                  {importReport.added.length === 1 ? "" : "s"}
                  {importReport.skipped.length
                    ? `, ${importReport.skipped.length} already added`
                    : ""}
                  {importReport.errors.length
                    ? `, ${importReport.errors.length} not imported`
                    : ""}
                </AlertTitle>
                <AlertDescription>
                  {importReport.added.length > 0 && (
                    <p>Added: {importReport.added.join(", ")}</p>
                  )}
                  {importReport.skipped.length > 0 && (
                    <p>Already added: {importReport.skipped.join(", ")}</p>
                  )}
                  {importReport.errors.length > 0 && (
                    <ul className="list-disc pl-4">
                      {importReport.errors.map((error, index) => (
                        <li key={index}>{error}</li>
                      ))}
                    </ul>
                  )}
                  <Button
                    type="button"
                    variant="ghost"
                    size="sm"
                    className="mt-1 h-auto px-0"
                    onClick={() => setImportReport(null)}
                  >
                    Dismiss
                  </Button>
                </AlertDescription>
              </Alert>
            )}
            {stdioBlocked &&
              servers.some(
                (server) => !server.builtin_id && !isHttpAddress(server.url),
              ) && (
                <Alert role="note">
                  <AlertTitle>Local programs are paused</AlertTitle>
                  <AlertDescription>
                    {capabilities?.stdio_disabled_reason ??
                      "Local-program servers aren't available on this server."}
                  </AlertDescription>
                </Alert>
              )}
            {/* Spinner only for the first load: a background refresh after every toggle or delete used
                to swap the whole list for a spinner, flashing it and dropping keyboard focus. */}
            {loading && servers.length === 0 ? (
              <div className="flex justify-center py-6">
                <Spinner />
              </div>
            ) : servers.filter((server) => !server.builtin_id).length === 0 ? (
              <div className="rounded-md border border-dashed px-4 py-6 text-center text-sm text-muted-foreground">
                No MCP servers yet. Use <strong>Add server</strong> for a URL or
                a local program, or <strong>Import config</strong> to bring
                over servers from Claude Desktop, Cursor or VS Code.
              </div>
            ) : (
              <ul className="flex flex-col divide-y rounded-md border">
                {servers.filter((server) => !server.builtin_id).map((server) => (
                  <li
                    key={server.id}
                    className="flex items-center justify-between gap-3 px-3 py-2"
                  >
                    <div className="min-w-0 flex-1">
                      <div className="flex min-w-0 items-center gap-2">
                        <span className="truncate font-medium">
                          {server.display_name}
                        </span>
                        {stdioBlocked && !isHttpAddress(server.url) && (
                          <span
                            className="shrink-0 rounded-sm bg-muted px-1.5 py-0.5 text-[10px] font-medium text-muted-foreground"
                            title={capabilities?.stdio_disabled_reason ?? undefined}
                          >
                            Paused
                          </span>
                        )}
                      </div>
                      <div className="truncate text-xs text-muted-foreground">
                        {displayAddress(server.url)}
                      </div>
                    </div>
                    <div className="flex items-center gap-1">
                      <Switch
                        checked={server.is_enabled}
                        onCheckedChange={(next) => toggleEnabled(server, next)}
                        aria-label={`Enable ${server.display_name}`}
                        disabled={importing || busyIds.has(server.id)}
                      />
                      <Button
                        type="button"
                        variant="ghost"
                        size="icon"
                        onClick={() => refreshTools(server)}
                        aria-label={`Refresh tools for ${server.display_name}`}
                        title={
                          stdioBlocked && !isHttpAddress(server.url)
                            ? (capabilities?.stdio_disabled_reason ?? undefined)
                            : "Refresh tools from this server"
                        }
                        disabled={
                          importing ||
                          busyIds.has(server.id) ||
                          (stdioBlocked && !isHttpAddress(server.url))
                        }
                      >
                        {refreshingIds.has(server.id) ? (
                          <Spinner />
                        ) : (
                          <RefreshCwIcon className="size-3.5" />
                        )}
                      </Button>
                      <Button
                        type="button"
                        variant="ghost"
                        size="icon"
                        onClick={() => void startEdit(server)}
                        aria-label={`Edit ${server.display_name}`}
                        disabled={importing || blenderBusy || busyIds.has(server.id)}
                      >
                        <HugeiconsIcon icon={Edit03Icon} className="size-3.5" />
                      </Button>
                      <Button
                        type="button"
                        variant="ghost"
                        size="icon"
                        onClick={() => setConfirmingDelete(server)}
                        aria-label={`Delete ${server.display_name}`}
                        disabled={importing || busyIds.has(server.id)}
                      >
                        <HugeiconsIcon icon={Delete02Icon} className="size-3.5" />
                      </Button>
                    </div>
                  </li>
                ))}
              </ul>
            )}
          </div>
        )}
      </DialogContent>
      <AlertDialog
        open={open && confirmingDelete !== null}
        onOpenChange={(next) => {
          if (!next) setConfirmingDelete(null);
        }}
      >
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Delete MCP server</AlertDialogTitle>
            <AlertDialogDescription>
              Delete{" "}
              <span className="font-medium text-foreground">
                &quot;{confirmingDelete?.display_name}&quot;
              </span>
              ? Its tools stop being available to chats. This cannot be undone.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Cancel</AlertDialogCancel>
            <AlertDialogAction
              variant="destructive"
              onClick={() => {
                const server = confirmingDelete;
                setConfirmingDelete(null);
                if (server) void removeServer(server);
              }}
            >
              Delete
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </Dialog>
  );
}

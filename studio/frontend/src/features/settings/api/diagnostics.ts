// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";
import { readFastApiError } from "@/lib/format-fastapi-error";
import { downloadBlobStreaming, isDownloadCancelled } from "@/lib/native-files";
import {
  DIAGNOSTICS_BUNDLE_ENDPOINT,
  DIAGNOSTICS_ENDPOINT,
  type DiagnosticsReport,
  DiagnosticsRequestError,
  diagnosticsArchiveFilename,
  diagnosticsFailureForStatus,
  parseDiagnosticsReport,
} from "../lib/diagnostics";

export async function loadDiagnostics(
  signal?: AbortSignal,
): Promise<DiagnosticsReport> {
  const response = await authFetch(DIAGNOSTICS_ENDPOINT, { signal });
  if (!response.ok) {
    throw new DiagnosticsRequestError(
      diagnosticsFailureForStatus(response.status),
      await readFastApiError(response, "Could not collect diagnostics."),
    );
  }
  return parseDiagnosticsReport(await response.json());
}

/**
 * The logs archive with the report added, saved where the user picks on desktop
 * and as a download in a browser. Resolves false when the user cancelled the
 * save dialog, true once the file is written (or the download handed over).
 *
 * Fetched here rather than by the desktop "Download all logs" command, which is
 * pinned to the bare export route and drops any query.
 */
export async function saveDiagnosticsBundle(): Promise<boolean> {
  const response = await authFetch(DIAGNOSTICS_BUNDLE_ENDPOINT);
  if (!response.ok) {
    throw new DiagnosticsRequestError(
      diagnosticsFailureForStatus(response.status),
      await readFastApiError(response, "Could not save the diagnostics bundle."),
    );
  }
  try {
    await downloadBlobStreaming(await response.blob(), diagnosticsArchiveFilename());
  } catch (error) {
    if (isDownloadCancelled(error)) return false;
    throw error;
  }
  return true;
}

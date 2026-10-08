// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useEffect } from "react";

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
import { useT } from "@/i18n";
import { loadVerdictOtherApps, loadVerdictSentence } from "@/lib/load-verdict";
import {
  answerMemoryOvercommitConsent,
  registerMemoryOvercommitHost,
  useMemoryOvercommitStore,
} from "../memory-overcommit-consent";

/**
 * "This model probably won't fit": the backend's memory guardrail refused a load, and this asks
 * whether to load it anyway. Mounted ONCE at the app root, since loadModel asks from every
 * surface (chat, Compare, Hub, Audio, recipes). With none mounted, loadModel treats the question
 * as declined rather than waiting on a dialog that does not exist.
 */
export function MemoryOvercommitDialog() {
  const t = useT();
  const request = useMemoryOvercommitStore((state) => state.pending[0] ?? null);

  useEffect(() => registerMemoryOvercommitHost(), []);

  if (!request) return null;
  const { verdict } = request;
  // Strict mode also asks about a load that works but streams from disk; it is slow, not doomed.
  const slow = verdict.level === "disk_streaming";
  const otherApps = loadVerdictOtherApps(verdict, t);

  return (
    <AlertDialog
      open
      onOpenChange={(open) => {
        // Escape / overlay click must resolve, or loadModel's await hangs.
        if (!open) answerMemoryOvercommitConsent(request.id, "cancel");
      }}
    >
      <AlertDialogContent>
        <AlertDialogHeader>
          <AlertDialogTitle>
            {slow ? t("loadVerdict.dialogTitleSlow") : t("loadVerdict.dialogTitle")}
          </AlertDialogTitle>
          <AlertDialogDescription className="space-y-2">
            <span className="block break-words font-medium text-foreground">
              {request.modelLabel}
            </span>
            <span className="block">
              {loadVerdictSentence(verdict, t)}
              {otherApps ? ` ${otherApps}` : ""}
            </span>
            <span className="block">
              {slow
                ? t("loadVerdict.dialogHintSlow")
                : otherApps
                  ? t("loadVerdict.dialogHint")
                  : // "Close the other app" with no other app named reads as a mistake.
                    t("loadVerdict.dialogHintNoOtherApps")}
            </span>
          </AlertDialogDescription>
        </AlertDialogHeader>
        <AlertDialogFooter>
          <AlertDialogCancel
            onClick={() => answerMemoryOvercommitConsent(request.id, "cancel")}
          >
            {t("loadVerdict.cancel")}
          </AlertDialogCancel>
          <AlertDialogAction
            onClick={() => answerMemoryOvercommitConsent(request.id, "load")}
          >
            {t("loadVerdict.loadAnyway")}
          </AlertDialogAction>
        </AlertDialogFooter>
      </AlertDialogContent>
    </AlertDialog>
  );
}

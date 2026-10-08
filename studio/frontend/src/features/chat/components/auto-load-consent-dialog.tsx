// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useEffect, useState } from "react";

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
import { Checkbox } from "@/components/ui/checkbox";
import {
  type AutoLoadConsentReason,
  answerAutoLoadConsent,
  formatAutoLoadSize,
  registerAutoLoadConsentHost,
  setAutoLoadsLastModelWithoutAsking,
  useAutoLoadConsentStore,
} from "../auto-load-consent";

function copyFor(reason: AutoLoadConsentReason, model: string, size: string | null) {
  const sized = size ? `${model} (about ${size})` : model;
  switch (reason) {
    case "last-used":
      return {
        title: "Load your last model?",
        body: `No model is loaded. Unsloth can load ${sized}, the model you used last, and then send your message.`,
        confirm: "Load and send",
      };
    case "smallest":
      return {
        title: "Load a model to send this?",
        body: `No model is loaded. The smallest model on this computer is ${sized}. Loading it uses GPU memory and can take a while. Load it, or choose a different model.`,
        confirm: "Load and send",
      };
    case "default-download":
      return {
        title: "Download a starter model?",
        body: `No model was found on this computer. Unsloth can download ${sized}, a small chat model, load it, and then send your message.`,
        confirm: "Download and send",
      };
  }
}

/**
 * Asks before a send loads a model the user did not choose. Mount once per page that can send; the
 * adapter waits on the answer, and with no dialog mounted a send declines instead of loading.
 */
export function AutoLoadConsentDialog({
  onChooseModel,
}: {
  /** Open the model picker; "Choose a model" ends the send so the pick is the user's own. */
  onChooseModel: () => void;
}) {
  const request = useAutoLoadConsentStore((state) => state.pending[0] ?? null);
  const [rememberLastModel, setRememberLastModel] = useState(false);

  useEffect(() => registerAutoLoadConsentHost(), []);

  // A fresh question starts unticked, even if the previous one was ticked and then declined.
  const requestId = request?.id ?? null;
  const [shownId, setShownId] = useState<number | null>(requestId);
  if (shownId !== requestId) {
    setShownId(requestId);
    setRememberLastModel(false);
  }

  if (!request) return null;
  const copy = copyFor(
    request.reason,
    request.modelLabel,
    formatAutoLoadSize(request.sizeBytes),
  );

  return (
    <AlertDialog
      open
      onOpenChange={(open) => {
        if (!open) answerAutoLoadConsent(request.id, "cancel");
      }}
    >
      <AlertDialogContent>
        <AlertDialogHeader>
          <AlertDialogTitle>{copy.title}</AlertDialogTitle>
          <AlertDialogDescription>{copy.body}</AlertDialogDescription>
        </AlertDialogHeader>
        {request.reason === "last-used" && (
          <label className="flex items-center gap-2 text-sm">
            <Checkbox
              checked={rememberLastModel}
              onCheckedChange={(checked) => setRememberLastModel(checked === true)}
            />
            Always load my last model when nothing is loaded
          </label>
        )}
        <AlertDialogFooter>
          <AlertDialogCancel
            onClick={() => answerAutoLoadConsent(request.id, "cancel")}
          >
            Cancel
          </AlertDialogCancel>
          <Button
            type="button"
            variant="outline"
            onClick={() => {
              answerAutoLoadConsent(request.id, "choose");
              onChooseModel();
            }}
          >
            Choose a model
          </Button>
          <AlertDialogAction
            onClick={() => {
              if (request.reason === "last-used" && rememberLastModel) {
                setAutoLoadsLastModelWithoutAsking(true);
              }
              answerAutoLoadConsent(request.id, "load");
            }}
          >
            {copy.confirm}
          </AlertDialogAction>
        </AlertDialogFooter>
      </AlertDialogContent>
    </AlertDialog>
  );
}

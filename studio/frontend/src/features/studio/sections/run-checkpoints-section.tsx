// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { SectionCard } from "@/components/section-card";
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
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { Spinner } from "@/components/ui/spinner";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { bumpInventoryVersion } from "@/features/hub";
import { formatSize, resetToNewChat, useRevealLabel } from "@/features/library";
import {
  clearModelConfigHandoff,
  createModelConfigHandoffRequestId,
  requestModelConfigHandoff,
} from "@/features/model-picker";
import {
  BEST_CHECKPOINT_LABEL_KEYS,
  type TrainingRunCheckpoint,
  type TrainingRunCheckpointsResponse,
  checkpointExportSearch,
  checkpointResumeAction,
  deleteTrainingRunCheckpoint,
  emitTrainingRunsChanged,
  forkTrainingRunCheckpoint,
  formatCheckpointEpoch,
  formatCheckpointLoss,
  formatCheckpointSavedAt,
  listTrainingRunCheckpoints,
  pickBestCheckpoint,
  revealTrainingRunCheckpoint,
  useTrainingActions,
} from "@/features/training";
import { useLocale, useT } from "@/i18n";
import { copyToClipboard } from "@/lib/copy-to-clipboard";
import { toast } from "@/lib/toast";
import { cn } from "@/lib/utils";
import {
  BubbleChatIcon,
  Copy01Icon,
  Delete02Icon,
  FolderExportIcon,
  FolderOpenIcon,
  Layers01Icon,
  MoreHorizontalIcon,
  PlayIcon,
  StarIcon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useNavigate } from "@tanstack/react-router";
import { type ReactElement, useCallback, useEffect, useState } from "react";

interface RunCheckpointsSectionProps {
  runId: string;
  /** Laya / Clef decision models export to GGUF but cannot be loaded in Chat. */
  isDecision?: boolean;
  onResumeStarted?: () => void;
}

type ListingState = {
  runId: string;
  listing: TrainingRunCheckpointsResponse | null;
  failed: boolean;
  error: string | null;
};

function errorText(error: unknown): string | undefined {
  return error instanceof Error ? error.message : undefined;
}

export function RunCheckpointsSection({
  runId,
  isDecision = false,
  onResumeStarted,
}: RunCheckpointsSectionProps): ReactElement {
  const t = useT();
  const locale = useLocale();
  const navigate = useNavigate();
  const revealLabel = useRevealLabel();
  const { resumeTrainingRunFromHistory, startBlocked } = useTrainingActions();
  const [state, setState] = useState<ListingState | null>(null);
  const [reloadToken, setReloadToken] = useState(0);
  const [busyId, setBusyId] = useState<string | null>(null);
  const [deleteTarget, setDeleteTarget] =
    useState<TrainingRunCheckpoint | null>(null);
  const [finalAcknowledged, setFinalAcknowledged] = useState(false);

  useEffect(() => {
    const controller = new AbortController();
    listTrainingRunCheckpoints(runId, controller.signal)
      .then((listing) =>
        setState({ runId, listing, failed: false, error: null }),
      )
      .catch((error: unknown) => {
        if (error instanceof DOMException && error.name === "AbortError") {
          return;
        }
        setState({
          runId,
          listing: null,
          failed: true,
          error: errorText(error) ?? null,
        });
      });
    return () => controller.abort();
  }, [runId, reloadToken]);

  const reload = useCallback(() => setReloadToken((value) => value + 1), []);
  const current = state?.runId === runId ? state : null;
  const listing = current?.listing ?? null;
  const checkpoints = listing?.checkpoints ?? [];
  const runName = listing?.run_name ?? null;
  const best = pickBestCheckpoint(checkpoints);
  const showEval = checkpoints.some((row) => row.eval_loss != null);
  const busy = busyId !== null;

  const handleChat = (checkpoint: TrainingRunCheckpoint) => {
    if (!runName) return;
    // The same hand-off the Library's "Chat with model" uses: Chat loads the adapter onto its base model.
    const requestId = createModelConfigHandoffRequestId();
    resetToNewChat();
    requestModelConfigHandoff({
      requestId,
      id: checkpoint.path,
      displayName: checkpoint.is_final ? runName : `${runName}/${checkpoint.id}`,
      meta: {
        source: "lora",
        isLora: checkpoint.is_adapter,
        isDownloaded: true,
        isGguf: false,
      },
    });
    void navigate({ to: "/chat", search: { new: requestId } }).catch(() =>
      clearModelConfigHandoff(requestId),
    );
  };

  const handleExport = (checkpoint: TrainingRunCheckpoint) => {
    if (!runName) return;
    void navigate({
      to: "/export",
      search: checkpointExportSearch(runName, checkpoint),
    });
  };

  const handleResume = async (checkpoint: TrainingRunCheckpoint) => {
    const action = checkpointResumeAction(checkpoint, startBlocked);
    if (action.kind === "disabled") return;
    setBusyId(checkpoint.id);
    try {
      if (action.kind === "in_place") {
        if (await resumeTrainingRunFromHistory(runId)) onResumeStarted?.();
        return;
      }
      // Never in this folder: an older checkpoint resumed here would save over the newer ones.
      const fork = await forkTrainingRunCheckpoint(runId, checkpoint.id);
      emitTrainingRunsChanged();
      if (await resumeTrainingRunFromHistory(fork.run_id)) {
        onResumeStarted?.();
      } else {
        toast(t("trainingRuns.checkpoints.forkSaved"), {
          description: t("trainingRuns.checkpoints.forkSavedDescription", {
            name: fork.output_dir_name,
          }),
        });
      }
    } catch (error) {
      toast.error(t("trainingRuns.checkpoints.forkFailed"), {
        description: errorText(error),
      });
    } finally {
      setBusyId(null);
    }
  };

  const handleReveal = (checkpoint: TrainingRunCheckpoint) => {
    revealTrainingRunCheckpoint(runId, checkpoint.id).catch((error: unknown) =>
      toast.error(t("trainingRuns.checkpoints.revealFailed"), {
        description: errorText(error),
      }),
    );
  };

  const handleCopyPath = async (checkpoint: TrainingRunCheckpoint) => {
    if (await copyToClipboard(checkpoint.path)) {
      toast.success(t("trainingRuns.checkpoints.pathCopied"));
    } else {
      toast.error(t("trainingRuns.checkpoints.copyFailed"));
    }
  };

  const closeDeleteDialog = () => {
    setDeleteTarget(null);
    setFinalAcknowledged(false);
  };

  const handleDelete = async () => {
    const target = deleteTarget;
    if (!target || (target.is_final && !finalAcknowledged)) return;
    closeDeleteDialog();
    setBusyId(target.id);
    try {
      await deleteTrainingRunCheckpoint(runId, target.id, {
        confirmFinal: target.is_final,
      });
      toast.success(t("trainingRuns.checkpoints.deleted"));
      bumpInventoryVersion();
      if (target.is_final) emitTrainingRunsChanged();
    } catch (error) {
      toast.error(t("trainingRuns.checkpoints.deleteFailed"), {
        description: errorText(error),
      });
    } finally {
      setBusyId(null);
      reload();
    }
  };

  const totalSize = formatSize(listing?.total_size_bytes ?? null, locale, t);
  const description = !listing
    ? t("trainingRuns.checkpoints.description")
    : checkpoints.length === 1
      ? t("trainingRuns.checkpoints.summaryOne", { size: totalSize ?? "--" })
      : t("trainingRuns.checkpoints.summary", {
          count: checkpoints.length,
          size: totalSize ?? "--",
        });

  return (
    <SectionCard
      icon={<HugeiconsIcon icon={Layers01Icon} className="size-5" />}
      title={t("trainingRuns.checkpoints.title")}
      description={description}
      accent="indigo"
      className="shadow-border border border-border/60 bg-card/90 ring-0 backdrop-blur-sm"
    >
      {current === null ? (
        <p className="flex items-center gap-2 text-xs text-muted-foreground">
          <Spinner className="size-3.5" />
          {t("trainingRuns.checkpoints.loading")}
        </p>
      ) : current.failed ? (
        <div className="flex items-center gap-3 text-xs text-destructive" role="alert">
          <span className="min-w-0 break-words">
            {current.error || t("trainingRuns.checkpoints.loadFailed")}
          </span>
          <Button size="xs" variant="outline" onClick={reload}>
            {t("trainingRuns.checkpoints.retry")}
          </Button>
        </div>
      ) : !runName ? (
        <p className="text-xs text-muted-foreground">
          {t("trainingRuns.checkpoints.missing")}
        </p>
      ) : checkpoints.length === 0 ? (
        <p className="text-xs text-muted-foreground">
          {t("trainingRuns.checkpoints.empty")}
        </p>
      ) : (
        <div className="-mx-1 overflow-x-auto">
          <Table className="text-xs">
            <TableHeader>
              <TableRow>
                <TableHead>{t("trainingRuns.checkpoints.colStep")}</TableHead>
                <TableHead className="text-right">
                  {t("trainingRuns.checkpoints.colEpoch")}
                </TableHead>
                <TableHead className="text-right">
                  {t("trainingRuns.checkpoints.colLoss")}
                </TableHead>
                {showEval && (
                  <TableHead className="text-right">
                    {t("trainingRuns.checkpoints.colEvalLoss")}
                  </TableHead>
                )}
                <TableHead>{t("trainingRuns.checkpoints.colSaved")}</TableHead>
                <TableHead className="text-right">
                  {t("trainingRuns.checkpoints.colSize")}
                </TableHead>
                <TableHead className="text-right">
                  <span className="sr-only">
                    {t("trainingRuns.checkpoints.colActions")}
                  </span>
                </TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {checkpoints.map((checkpoint) => {
                const isBest = best?.id === checkpoint.id;
                const resume = checkpointResumeAction(checkpoint, startBlocked);
                const rowBusy = busyId === checkpoint.id;
                const lossNote =
                  checkpoint.train_loss_step != null &&
                  checkpoint.train_loss_step !== checkpoint.step
                    ? t("trainingRuns.checkpoints.lossAtStep", {
                        step: checkpoint.train_loss_step,
                      })
                    : undefined;
                return (
                  <TableRow
                    key={checkpoint.id}
                    className={cn(isBest && "bg-control-accent/5")}
                  >
                    <TableCell>
                      <div className="flex flex-wrap items-center gap-1.5">
                        <span className="font-medium tabular-nums">
                          {checkpoint.step ?? "--"}
                        </span>
                        {checkpoint.is_final && (
                          <span className="rounded-full border border-border/60 px-2 py-0.5 text-ui-10 text-muted-foreground">
                            {t(
                              checkpoint.is_adapter
                                ? "trainingRuns.checkpoints.final"
                                : "trainingRuns.checkpoints.finalModel",
                            )}
                          </span>
                        )}
                        {isBest && best && (
                          <span className="inline-flex items-center gap-1 rounded-full bg-control-accent/15 px-2 py-0.5 text-ui-10 font-semibold text-control-accent">
                            <HugeiconsIcon icon={StarIcon} className="size-3" />
                            {t(BEST_CHECKPOINT_LABEL_KEYS[best.basis])}
                          </span>
                        )}
                      </div>
                    </TableCell>
                    <TableCell className="text-right tabular-nums">
                      {formatCheckpointEpoch(checkpoint.epoch, locale)}
                    </TableCell>
                    <TableCell
                      className="text-right tabular-nums"
                      title={lossNote}
                    >
                      {formatCheckpointLoss(checkpoint.train_loss, locale)}
                    </TableCell>
                    {showEval && (
                      <TableCell className="text-right tabular-nums">
                        {formatCheckpointLoss(checkpoint.eval_loss, locale)}
                      </TableCell>
                    )}
                    <TableCell className="whitespace-nowrap text-muted-foreground">
                      {formatCheckpointSavedAt(checkpoint.saved_at, locale)}
                    </TableCell>
                    <TableCell className="whitespace-nowrap text-right tabular-nums text-muted-foreground">
                      {formatSize(checkpoint.size_bytes, locale, t) ?? "--"}
                    </TableCell>
                    <TableCell>
                      <div className="flex items-center justify-end gap-1.5">
                        <Button
                          size="xs"
                          variant="outline"
                          className="gap-1"
                          disabled={isDecision || busy}
                          title={
                            isDecision
                              ? t("trainingRuns.checkpoints.chatUnavailableDecision")
                              : undefined
                          }
                          onClick={() => handleChat(checkpoint)}
                        >
                          <HugeiconsIcon icon={BubbleChatIcon} className="size-3" />
                          {t("trainingRuns.checkpoints.chat")}
                        </Button>
                        <Button
                          size="xs"
                          variant="outline"
                          className="gap-1"
                          disabled={busy}
                          onClick={() => handleExport(checkpoint)}
                        >
                          <HugeiconsIcon icon={FolderExportIcon} className="size-3" />
                          {t("trainingRuns.checkpoints.export")}
                        </Button>
                        <DropdownMenu>
                          <DropdownMenuTrigger asChild={true}>
                            <Button
                              size="icon-xs"
                              variant="ghost"
                              aria-label={t("trainingRuns.checkpoints.more")}
                              disabled={busy}
                            >
                              {rowBusy ? (
                                <Spinner className="size-3.5" />
                              ) : (
                                <HugeiconsIcon
                                  icon={MoreHorizontalIcon}
                                  className="size-3.5"
                                />
                              )}
                            </Button>
                          </DropdownMenuTrigger>
                          <DropdownMenuContent align="end" className="max-w-72">
                            <DropdownMenuItem
                              disabled={resume.kind === "disabled"}
                              onSelect={() => void handleResume(checkpoint)}
                            >
                              <HugeiconsIcon icon={PlayIcon} />
                              <span className="flex min-w-0 flex-col">
                                <span>{t("trainingRuns.checkpoints.resume")}</span>
                                <span className="text-ui-10 text-muted-foreground">
                                  {resume.kind === "disabled"
                                    ? (resume.reason ?? t(resume.reasonKey))
                                    : t(
                                        resume.kind === "fork"
                                          ? "trainingRuns.checkpoints.resumeForkHint"
                                          : "trainingRuns.checkpoints.resumeInPlaceHint",
                                      )}
                                </span>
                              </span>
                            </DropdownMenuItem>
                            <DropdownMenuSeparator />
                            {revealLabel && (
                              <DropdownMenuItem
                                onSelect={() => handleReveal(checkpoint)}
                              >
                                <HugeiconsIcon icon={FolderOpenIcon} />
                                {revealLabel}
                              </DropdownMenuItem>
                            )}
                            <DropdownMenuItem
                              onSelect={() => void handleCopyPath(checkpoint)}
                            >
                              <HugeiconsIcon icon={Copy01Icon} />
                              {t("trainingRuns.checkpoints.copyPath")}
                            </DropdownMenuItem>
                            <DropdownMenuSeparator />
                            <DropdownMenuItem
                              variant="destructive"
                              onSelect={() => setDeleteTarget(checkpoint)}
                            >
                              <HugeiconsIcon icon={Delete02Icon} />
                              {t(
                                checkpoint.is_final
                                  ? "trainingRuns.checkpoints.deleteFinal"
                                  : "trainingRuns.checkpoints.delete",
                              )}
                            </DropdownMenuItem>
                          </DropdownMenuContent>
                        </DropdownMenu>
                      </div>
                    </TableCell>
                  </TableRow>
                );
              })}
            </TableBody>
          </Table>
        </div>
      )}
      <AlertDialog
        open={deleteTarget !== null}
        onOpenChange={(open) => {
          if (!open) closeDeleteDialog();
        }}
      >
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>
              {deleteTarget?.is_final
                ? t("trainingRuns.checkpoints.deleteFinalTitle")
                : t("trainingRuns.checkpoints.deleteTitle", {
                    step: deleteTarget?.step ?? "--",
                  })}
            </AlertDialogTitle>
            <AlertDialogDescription>
              {deleteTarget?.is_final
                ? t("trainingRuns.checkpoints.deleteFinalDescription")
                : t("trainingRuns.checkpoints.deleteDescription")}
            </AlertDialogDescription>
          </AlertDialogHeader>
          {deleteTarget?.is_final && (
            <label
              htmlFor="delete-final-adapter"
              className="flex cursor-pointer items-start gap-2 text-sm"
            >
              <Checkbox
                id="delete-final-adapter"
                checked={finalAcknowledged}
                onCheckedChange={(value) => setFinalAcknowledged(value === true)}
                className="mt-0.5"
              />
              <span className="text-foreground">
                {t("trainingRuns.checkpoints.deleteFinalConfirm")}
              </span>
            </label>
          )}
          <AlertDialogFooter>
            <AlertDialogCancel>{t("common.cancel")}</AlertDialogCancel>
            <AlertDialogAction
              variant="destructive"
              disabled={deleteTarget?.is_final === true && !finalAcknowledged}
              onClick={() => void handleDelete()}
            >
              {t("common.delete")}
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </SectionCard>
  );
}

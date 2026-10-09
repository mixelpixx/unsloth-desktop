// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useT } from "@/i18n";
import { tensorSplitConcern } from "./hardware-check-model";
import { useHardwareCheck } from "./hardware-check-store";

/**
 * Under a Tensor Parallelism switch the user turned on: what the hardware check measured on the
 * GPUs it would span, when "Avoid tensor parallel on slow links" is on. A warning, never a block:
 * the switch is the user's. Nothing for a managed account (the check is the owner's) or when the
 * result does not describe the GPUs installed now.
 */
export function TensorParallelLinkWarning({
  gpuIds,
}: {
  /** The GPUs the load is pinned to, or null for every GPU. */
  gpuIds: readonly number[] | null;
}) {
  const t = useT();
  const status = useHardwareCheck();
  const concern = tensorSplitConcern(status, gpuIds);
  if (!concern) return null;
  return (
    <div
      role="note"
      data-hardware-check-warning=""
      className="flex flex-col gap-1 rounded-lg border border-amber-500/40 bg-amber-500/5 px-3 py-2 text-ui-11 leading-snug text-foreground/85"
    >
      <p className="font-medium text-amber-700 dark:text-amber-400">
        {t("hardwareCheck.tensorWarning.title")}
      </p>
      <p>{t("hardwareCheck.tensorWarning.body")}</p>
      <ul className="flex list-disc flex-col gap-0.5 pl-4">
        {concern.slow.map((slow) => (
          <li key={`slow-${slow.gpu}`}>
            {t("hardwareCheck.tensorWarning.slowLink", {
              gpu: slow.gpu,
              width: slow.width ?? "?",
              h2d: slow.h2d,
              best: slow.best,
              bestGpu: slow.bestGpu,
            })}
          </li>
        ))}
        {concern.noPeer.map((pair) => (
          <li key={`peer-${pair.a}-${pair.b}`}>
            {t("hardwareCheck.tensorWarning.noPeer", {
              a: pair.a,
              b: pair.b,
              copy: pair.copy,
            })}
          </li>
        ))}
      </ul>
    </div>
  );
}

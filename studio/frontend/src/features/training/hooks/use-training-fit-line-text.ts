// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useT } from "@/i18n";
import { useCallback } from "react";
import { type TrainingFitEstimate, trainingFitLine } from "../lib/training-fit";

/** The fit verdict as one translated sentence: the panel's status line, the bar's accessible
 *  name and the Start confirm's reason all say it the same way. */
export function useTrainingFitLineText(): (
  estimate: TrainingFitEstimate,
) => string {
  const t = useT();
  return useCallback(
    (estimate) => {
      const formatTarget = (gpuIds: number[]) =>
        gpuIds.length === 0
          ? t("trainingFit.targetAuto")
          : gpuIds
              .map((index) => t("trainingFit.gpu", { index: String(index) }))
              .join(" + ");
      const line = trainingFitLine(estimate, formatTarget);
      return t(line.key, line.params);
    },
    [t],
  );
}

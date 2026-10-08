// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Slider } from "@/components/ui/slider";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { InformationCircleIcon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { type ReactElement, type ReactNode, useState } from "react";

export function ParamsRow({
  label,
  tooltip,
  children,
}: {
  label: string;
  tooltip?: ReactNode;
  children: ReactNode;
}): ReactElement {
  return (
    // Match the tallest control so slider and select rows share one rhythm.
    <div className="flex min-h-9 items-center justify-between">
      <span className="flex items-center gap-1.5 text-xs font-medium text-muted-foreground">
        {label}
        {tooltip && (
          <Tooltip>
            <TooltipTrigger asChild={true}>
              <button
                type="button"
                className="text-foreground/70 hover:text-foreground"
              >
                <HugeiconsIcon
                  icon={InformationCircleIcon}
                  className="size-3"
                />
              </button>
            </TooltipTrigger>
            <TooltipContent>{tooltip}</TooltipContent>
          </Tooltip>
        )}
      </span>
      {children}
    </div>
  );
}

export function ParamsSliderRow({
  label,
  tooltip,
  value,
  onChange,
  min,
  max,
  step,
  format,
}: {
  label: string;
  tooltip?: ReactNode;
  value: number;
  onChange: (value: number) => void;
  min: number;
  max: number;
  step: number;
  format?: (value: number) => string;
}): ReactElement {
  // Typed text lives in a draft keyed to the value it was typed against, so clearing the
  // field does not send 0 and `format` cannot rewrite "0.0" to "0.00" mid-keystroke. An
  // outside change (the slider) discards the draft.
  const [draft, setDraft] = useState<{ value: number; text: string } | null>(
    null,
  );
  const inputValue =
    draft && draft.value === value
      ? draft.text
      : format
        ? format(value)
        : String(value);
  const commitDraft = () => {
    if (!draft || draft.value !== value) {
      setDraft(null);
      return;
    }
    const parsed = Number(draft.text);
    setDraft(null);
    if (draft.text.trim() === "" || !Number.isFinite(parsed)) {
      return;
    }
    const clamped = Math.min(max, Math.max(min, parsed));
    if (clamped !== value) {
      onChange(clamped);
    }
  };
  return (
    <ParamsRow label={label} tooltip={tooltip}>
      <div className="flex items-center gap-3">
        <Slider
          value={[value]}
          onValueChange={([nextValue]) => onChange(nextValue)}
          min={min}
          max={max}
          step={step}
          className="w-32"
        />
        <input
          type="number"
          value={inputValue}
          onChange={(event) => {
            const text = event.target.value;
            const parsed = Number(text);
            // Live-apply only complete in-range values; anything else waits for blur.
            if (
              text.trim() !== "" &&
              Number.isFinite(parsed) &&
              parsed >= min &&
              parsed <= max
            ) {
              onChange(parsed);
              setDraft({ value: parsed, text });
              return;
            }
            setDraft({ value, text });
          }}
          onBlur={commitDraft}
          onKeyDown={(event) => {
            if (event.key === "Enter") {
              commitDraft();
            }
          }}
          min={min}
          max={max}
          step={step}
          // w-12 fits 11px mono, not the 16px coarse-pointer focus-zoom floor.
          className="w-12 pointer-coarse:w-16 text-right font-mono text-xs font-medium bg-muted/50 border border-border rounded-lg px-1.5 py-0.5 focus:outline-none focus:ring-1 focus:ring-ring [&::-webkit-inner-spin-button]:appearance-none"
        />
      </div>
    </ParamsRow>
  );
}

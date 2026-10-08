// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { type TranslationKey, useT } from "@/i18n";
import { useEffect, useState } from "react";
import {
  LOAD_GUARDRAIL_MODES,
  type LoadGuardrailMode,
  type LoadGuardrailSettings,
  isLoadGuardrailMode,
  loadLoadGuardrailSettings,
  updateLoadGuardrailMode,
} from "../api/load-guardrails";
import { isSettingsRouteAbsent } from "../api/settings-route-absent";
import { SettingsRow } from "./settings-row";

const MODE_LABEL: Record<LoadGuardrailMode, TranslationKey> = {
  strict: "loadVerdict.settings.strict",
  balanced: "loadVerdict.settings.balanced",
  relaxed: "loadVerdict.settings.relaxed",
  off: "loadVerdict.settings.off",
};

const MODE_DESCRIPTION: Record<LoadGuardrailMode, TranslationKey> = {
  strict: "loadVerdict.settings.strictDescription",
  balanced: "loadVerdict.settings.balancedDescription",
  relaxed: "loadVerdict.settings.relaxedDescription",
  off: "loadVerdict.settings.offDescription",
};

/**
 * Load guardrails: which memory verdicts stop a load for "Load anyway". Beside the model memory
 * switches because it answers the same question from the other side: those decide where a
 * loaded model lives, this decides whether a load that will not fit gets started at all.
 */
export function LoadGuardrailsRow() {
  const t = useT();
  const [settings, setSettings] = useState<LoadGuardrailSettings | null>(null);
  const [absent, setAbsent] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    let cancelled = false;
    loadLoadGuardrailSettings()
      .then((loaded) => {
        if (!cancelled) setSettings(loaded);
      })
      .catch((loadError: unknown) => {
        if (cancelled) return;
        // A backend that predates the guardrail has nothing to configure: no row, no error.
        if (isSettingsRouteAbsent(loadError)) setAbsent(true);
        else setError(t("loadVerdict.settings.loadError"));
      });
    return () => {
      cancelled = true;
    };
  }, [t]);

  if (absent) return null;

  const persist = async (mode: LoadGuardrailMode) => {
    setSaving(true);
    setError(null);
    try {
      setSettings(await updateLoadGuardrailMode(mode));
    } catch {
      setError(t("loadVerdict.settings.saveError"));
    } finally {
      setSaving(false);
    }
  };

  const mode = settings?.mode ?? "balanced";

  return (
    <SettingsRow
      label={t("loadVerdict.settings.label")}
      description={t(MODE_DESCRIPTION[mode])}
      hint={t("loadVerdict.settings.hint")}
      below={
        error ? <span className="text-xs text-destructive">{error}</span> : undefined
      }
    >
      <Select
        value={mode}
        disabled={!settings || saving}
        onValueChange={(value) => {
          if (isLoadGuardrailMode(value) && value !== mode) void persist(value);
        }}
      >
        <SelectTrigger
          aria-label={t("loadVerdict.settings.label")}
          className="w-36"
          size="sm"
        >
          <SelectValue />
        </SelectTrigger>
        <SelectContent>
          {LOAD_GUARDRAIL_MODES.map((option) => (
            <SelectItem key={option} value={option}>
              {t(MODE_LABEL[option])}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
    </SettingsRow>
  );
}

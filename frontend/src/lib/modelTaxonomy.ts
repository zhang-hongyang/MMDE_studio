// Model taxonomy (GET /api/model-taxonomy): excluded fnmatch patterns +
// ordered categories. A model hidden by `excluded` never shows up; a visible
// model joins the FIRST category whose `match` patterns hit (fnmatch style,
// case-sensitive). Models not matching any category are dropped.
import { useEffect, useState } from "react";

export interface ModelCategory {
  id: string;
  label: string;
  /** fnmatch-style patterns, `*` matches any run of characters (incl. empty) */
  match: string[];
}

export interface ModelTaxonomy {
  excluded: string[];
  categories: ModelCategory[];
}

export interface ModelGroup {
  id: string;
  label: string;
  models: string[];
}

/** fnmatch-style match: only `*` is special (any chars, incl. empty); case-sensitive. */
export function fnmatch(name: string, pattern: string): boolean {
  const re = pattern
    .replace(/[.+^${}()|[\]\\]/g, "\\$&")
    .replace(/\*/g, ".*");
  return new RegExp(`^${re}$`).test(name);
}

/** models with excluded names removed, original order preserved */
export function visibleModels(models: string[], t: ModelTaxonomy): string[] {
  return models.filter((m) => !t.excluded.some((p) => fnmatch(m, p)));
}

/** group visible models by first matching category (declaration order); empty groups dropped */
export function classifyModels(models: string[], t: ModelTaxonomy): ModelGroup[] {
  const groups: ModelGroup[] = t.categories.map((c) => ({ id: c.id, label: c.label, models: [] }));
  for (const m of visibleModels(models, t)) {
    const i = t.categories.findIndex((c) => c.match.some((p) => fnmatch(m, p)));
    if (i >= 0) groups[i].models.push(m);
  }
  return groups.filter((g) => g.models.length > 0);
}

let cached: Promise<ModelTaxonomy | null> | null = null;

/** fetch taxonomy once per page load; null when the endpoint is unavailable */
export function loadModelTaxonomy(): Promise<ModelTaxonomy | null> {
  cached ??= fetch("/api/model-taxonomy")
    .then((r) => (r.ok ? (r.json() as Promise<ModelTaxonomy>) : null))
    .catch(() => null);
  return cached;
}

/**
 * Group `models` by the cached taxonomy. Returns null while loading or when
 * the taxonomy endpoint is unavailable — callers should then fall back to a
 * flat, unfiltered list.
 */
export function useModelGroups(models: string[]): ModelGroup[] | null {
  const [groups, setGroups] = useState<ModelGroup[] | null>(null);
  useEffect(() => {
    let live = true;
    void loadModelTaxonomy().then((t) => {
      if (live) setGroups(t ? classifyModels(models, t) : null);
    });
    return () => {
      live = false;
    };
  }, [models]);
  return groups;
}

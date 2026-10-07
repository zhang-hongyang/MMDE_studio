import { useMemo, useState } from "react";
import { ArrowDown, ArrowUp } from "lucide-react";
import type { ModelSplitMetrics } from "./api";
import { bestValues, fmt, isNum, metricKeys, metricMeta } from "./format";
import { cn } from "../../lib/utils";

interface Props {
  models: ModelSplitMetrics[];
}

type SortState = { key: string; dir: "asc" | "desc" } | null;

export function MetricsTable({ models }: Props) {
  const [sort, setSort] = useState<SortState>(null);
  const keys = useMemo(() => metricKeys(models), [models]);
  const best = useMemo(() => bestValues(models, keys), [models, keys]);

  const rows = useMemo(() => {
    const rs = [...models];
    if (sort) {
      const { key, dir } = sort;
      const mul = dir === "asc" ? 1 : -1;
      rs.sort((a, b) => {
        let cmp: number;
        if (key === "__model") {
          cmp = a.model.localeCompare(b.model);
        } else {
          const va = a.aggregate[key];
          const vb = b.aggregate[key];
          const na = isNum(va) ? va : null;
          const nb = isNum(vb) ? vb : null;
          // nulls always sink to the bottom regardless of direction
          if (na === null && nb === null) cmp = 0;
          else if (na === null) cmp = 1;
          else if (nb === null) cmp = -1;
          else cmp = na - nb;
        }
        return cmp * mul;
      });
    }
    return rs;
  }, [models, sort]);

  const onSort = (key: string) =>
    setSort((s) =>
      s?.key !== key ? { key, dir: "asc" } : s.dir === "asc" ? { key, dir: "desc" } : null,
    );

  if (models.length === 0) return null;

  return (
    <div className="card max-h-[480px] overflow-auto p-0">
      <table className="w-full border-collapse text-[12px]">
        <thead className="sticky top-0 z-10 bg-card">
          <tr className="h-9 text-left text-fg-muted">
            <th className="sticky left-0 z-10 bg-card px-3 font-medium">
              <button
                className="inline-flex items-center gap-1 hover:text-fg"
                onClick={() => onSort("__model")}
              >
                模型
                {sort?.key === "__model" &&
                  (sort.dir === "asc" ? (
                    <ArrowUp className="h-3 w-3" />
                  ) : (
                    <ArrowDown className="h-3 w-3" />
                  ))}
              </button>
            </th>
            {keys.map((k) => (
              <th key={k} className="px-2 text-right font-medium">
                <button
                  className={cn(
                    "inline-flex items-center gap-1 hover:text-fg",
                    sort?.key === k && "text-fg",
                  )}
                  onClick={() => onSort(k)}
                  title={`${metricMeta(k).label}（点击排序）`}
                >
                  {metricMeta(k).label}
                  {sort?.key === k &&
                    (sort.dir === "asc" ? (
                      <ArrowUp className="h-3 w-3" />
                    ) : (
                      <ArrowDown className="h-3 w-3" />
                    ))}
                </button>
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((m) => (
            <tr key={m.model} className="h-9 border-t border-border">
              <td className="sticky left-0 z-0 max-w-56 truncate bg-card px-3 font-mono">
                {m.model}
              </td>
              {keys.map((k) => {
                const v = m.aggregate[k];
                const isBest = isNum(v) && best.get(k) === v;
                return (
                  <td
                    key={k}
                    className={cn(
                      "num px-2 text-right",
                      isBest && "bg-success/15 font-medium text-success",
                    )}
                  >
                    {fmt(v)}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

// Model dropdown with two-level grouping (taxonomy category → models).
// Falls back to a flat, unfiltered list when /api/model-taxonomy is
// unavailable, so a backend without the endpoint never renders an empty menu.
import { Select } from "../components/ui/select";
import { useModelGroups } from "./modelTaxonomy";

export interface ModelSelectProps {
  models: string[];
  value: string;
  onValueChange: (v: string) => void;
  className?: string;
  ariaLabel: string;
}

export function ModelSelect({ models, value, onValueChange, className, ariaLabel }: ModelSelectProps) {
  const groups = useModelGroups(models);
  return (
    <Select.Root value={value} onValueChange={onValueChange}>
      <Select.Trigger className={className} aria-label={ariaLabel}>
        <Select.Value placeholder={ariaLabel} />
      </Select.Trigger>
      <Select.Content>
        {groups
          ? groups.map((g) => (
              <Select.Group key={g.id}>
                <Select.Label>{g.label}</Select.Label>
                {g.models.map((m) => (
                  <Select.Item key={m} value={m}>
                    {m}
                  </Select.Item>
                ))}
              </Select.Group>
            ))
          : models.map((m) => (
              <Select.Item key={m} value={m}>
                {m}
              </Select.Item>
            ))}
      </Select.Content>
    </Select.Root>
  );
}

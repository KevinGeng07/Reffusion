export interface ModelChoice {
  key: string;
  label: string;
}

// Mirrors api_data/image_model.py's MODEL_CHOICES (key -> label). Kept in
// sync manually since it's a small, rarely-changing reference list.
export const MODEL_CHOICES: ModelChoice[] = [
  { key: "sd-turbo", label: "SD-Turbo" },
  { key: "sdxl-turbo", label: "SDXL-Turbo" },
  { key: "flux2-klein", label: "FLUX.2 Klein 4B (fp8)" },
  { key: "realvisxl-v5", label: "RealVisXL V5.0" },
];

export function modelLabel(key: string): string {
  return MODEL_CHOICES.find((m) => m.key === key)?.label ?? key;
}

export type ModelDownloadStatus =
  | "not_downloaded"
  | "queued"
  | "downloading"
  | "ready"
  | "error";

export interface ModelStatus {
  key: string;
  label: string;
  status: ModelDownloadStatus;
  downloaded_bytes: number;
  total_bytes: number;
}

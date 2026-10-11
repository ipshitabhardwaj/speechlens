const BASE = "/api";

export interface RecordingResponse {
  id: string;
  duration: number;
  sample_rate: number;
  warnings: string[];
}

export interface FlawResponse {
  rule_id: string;
  category: string;
  severity: string;
  start_seconds: number;
  end_seconds: number;
  participant_measurement: Record<string, unknown>;
  reference_measurement: Record<string, unknown>;
  deviation: number;
  evidence: string;
  interpretation: string;
  recommendation: string;
  scope: string;
}

export interface WordTimingResponse {
  word: string;
  start: number;
  end: number;
  score: number;
}

export interface DimensionResponse {
  name: string;
  score: number;
  weight: number;
  available: boolean;
  unavailable_reason: string | null;
}

export interface RubricScoreResponse {
  overall_score: number;
  version: string;
  dimensions: DimensionResponse[];
}

export interface AnalysisResultResponse {
  flaws: FlawResponse[];
  unavailable: string[];
  rubric: RubricScoreResponse;
  alignment_method: string;
  participant_duration: number;
  reference_duration: number;
  participant_rate_wpm: number | null;
  reference_rate_wpm: number | null;
  participant_words: WordTimingResponse[] | null;
  reference_words: WordTimingResponse[] | null;
}

export interface AnalysisResponse {
  id: string;
  participant_id: string;
  reference_id: string;
  status: string;
  stage: string | null;
  error: string | null;
  result: AnalysisResultResponse | null;
  created_at: string;
}

export async function uploadRecording(file: File): Promise<RecordingResponse> {
  const form = new FormData();
  form.append("file", file);
  const resp = await fetch(`${BASE}/recordings`, { method: "POST", body: form });
  if (!resp.ok) {
    const msg = await resp.text();
    throw new Error(msg);
  }
  return resp.json();
}

export async function startAnalysis(
  participantId: string,
  referenceId: string,
  transcript: string,
): Promise<AnalysisResponse> {
  const resp = await fetch(`${BASE}/analyses`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      participant_id: participantId,
      reference_id: referenceId,
      transcript,
    }),
  });
  if (!resp.ok) {
    const msg = await resp.text();
    throw new Error(msg);
  }
  return resp.json();
}

export async function getAnalysis(id: string): Promise<AnalysisResponse> {
  const resp = await fetch(`${BASE}/analyses/${id}`);
  if (!resp.ok) throw new Error(`Analysis ${id} not found`);
  return resp.json();
}

export function audioUrl(recordingId: string): string {
  return `${BASE}/recordings/${recordingId}/audio`;
}

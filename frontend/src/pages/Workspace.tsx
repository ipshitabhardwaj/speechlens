import { useCallback, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import {
  uploadRecording,
  startAnalysis,
  getAnalysis,
  type RecordingResponse,
  type AnalysisResponse,
} from "../api";

type UploadState = { status: "idle" } | { status: "uploading" } | { status: "done"; rec: RecordingResponse } | { status: "error"; message: string };

const STAGE_LABELS: Record<string, string> = {
  queued: "Queued",
  decoding: "Decoding audio",
  extracting_features: "Extracting features",
  loading_alignment_model: "Loading alignment model",
  aligning: "Aligning",
  comparing: "Detecting flaws",
  scoring: "Scoring rubric",
};

export default function Workspace() {
  const navigate = useNavigate();
  const [refState, setRefState] = useState<UploadState>({ status: "idle" });
  const [partState, setPartState] = useState<UploadState>({ status: "idle" });
  const [transcript, setTranscript] = useState("");
  const [analysisStatus, setAnalysisStatus] = useState<AnalysisResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [running, setRunning] = useState(false);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const handleUpload = useCallback(async (file: File, setter: (s: UploadState) => void) => {
    setter({ status: "uploading" });
    try {
      const rec = await uploadRecording(file);
      setter({ status: "done", rec });
    } catch (e) {
      setter({ status: "error", message: e instanceof Error ? e.message : "Upload failed" });
    }
  }, []);

  const canRun =
    refState.status === "done" &&
    partState.status === "done" &&
    transcript.trim().length > 0 &&
    !running;

  const handleRun = useCallback(async () => {
    if (refState.status !== "done" || partState.status !== "done") return;
    setRunning(true);
    setError(null);
    setAnalysisStatus(null);

    try {
      const analysis = await startAnalysis(partState.rec.id, refState.rec.id, transcript.trim());
      setAnalysisStatus(analysis);

      pollRef.current = setInterval(async () => {
        try {
          const updated = await getAnalysis(analysis.id);
          setAnalysisStatus(updated);
          if (updated.status === "complete") {
            if (pollRef.current) clearInterval(pollRef.current);
            navigate(`/results/${updated.id}`);
          } else if (updated.status === "error") {
            if (pollRef.current) clearInterval(pollRef.current);
            setError(updated.error || "Analysis failed");
            setRunning(false);
          }
        } catch {
          if (pollRef.current) clearInterval(pollRef.current);
          setError("Lost connection to server");
          setRunning(false);
        }
      }, 500);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to start analysis");
      setRunning(false);
    }
  }, [refState, partState, transcript, navigate]);

  return (
    <div className="min-h-screen bg-[#FAF9F6] text-[#2D3748]">
      <header className="border-b border-[#E2DFD8] px-6 py-4">
        <h1 className="text-2xl font-semibold text-[#1A202C]">SpeechLens</h1>
        <p className="text-sm text-[#718096] mt-1">Contrastive Speech Analytics</p>
      </header>

      <main className="max-w-2xl mx-auto px-4 py-8 space-y-8">
        {/* Reference recording */}
        <section>
          <h2 className="text-lg font-medium mb-2">Reference recording</h2>
          <p className="text-sm text-[#718096] mb-3">The ideal delivery to compare against.</p>
          <UploadBox state={refState} onFile={(f) => handleUpload(f, setRefState)} label="reference" />
        </section>

        {/* Participant recording */}
        <section>
          <h2 className="text-lg font-medium mb-2">Participant recording</h2>
          <p className="text-sm text-[#718096] mb-3">The recording to analyse.</p>
          <UploadBox state={partState} onFile={(f) => handleUpload(f, setPartState)} label="participant" />
        </section>

        {/* Transcript */}
        <section>
          <h2 className="text-lg font-medium mb-2">Transcript</h2>
          <p className="text-sm text-[#718096] mb-3">
            The text both recordings read. Must match exactly.
          </p>
          <textarea
            className="w-full border border-[#CBD5E0] rounded-lg px-4 py-3 text-sm bg-white focus:outline-none focus:ring-2 focus:ring-[#4A6FA5] resize-y min-h-[120px]"
            placeholder="Paste the passage text here..."
            value={transcript}
            onChange={(e) => setTranscript(e.target.value)}
          />
          {transcript.trim() && (
            <p className="text-xs text-[#718096] mt-1">
              {transcript.trim().split(/\s+/).length} words
            </p>
          )}
        </section>

        {/* Run button */}
        <button
          onClick={handleRun}
          disabled={!canRun}
          className="w-full py-3 rounded-lg font-medium text-white transition-colors disabled:opacity-40 disabled:cursor-not-allowed bg-[#1A202C] hover:bg-[#2D3748]"
        >
          {running ? "Analysing..." : "Run analysis"}
        </button>

        {/* Progress */}
        {analysisStatus && running && (
          <div className="border border-[#E2DFD8] rounded-lg p-4 bg-white">
            <p className="text-sm font-medium mb-2">Analysis in progress</p>
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 rounded-full bg-[#4A6FA5] animate-pulse" />
              <p className="text-sm text-[#718096]">
                {STAGE_LABELS[analysisStatus.stage || "queued"] || analysisStatus.stage || "Starting"}
              </p>
            </div>
          </div>
        )}

        {/* Error */}
        {error && (
          <div className="border border-[#C9605A] rounded-lg p-4 bg-[#FFF5F5]">
            <p className="text-sm font-medium text-[#C9605A]">Error</p>
            <p className="text-sm text-[#C9605A] mt-1">{error}</p>
          </div>
        )}
      </main>
    </div>
  );
}


function UploadBox({
  state,
  onFile,
  label,
}: {
  state: UploadState;
  onFile: (f: File) => void;
  label: string;
}) {
  const inputRef = useRef<HTMLInputElement>(null);

  if (state.status === "done") {
    return (
      <div className="border border-[#C6E4C0] bg-[#F0FAF0] rounded-lg p-4">
        <p className="text-sm font-medium text-[#48886B]">Uploaded</p>
        <p className="text-xs text-[#718096] mt-1">
          {state.rec.duration.toFixed(1)}s &middot; {state.rec.sample_rate} Hz
          {state.rec.warnings.length > 0 && ` · ${state.rec.warnings.join(", ")}`}
        </p>
      </div>
    );
  }

  if (state.status === "uploading") {
    return (
      <div className="border border-[#E2DFD8] rounded-lg p-4 bg-white flex items-center gap-2">
        <div className="w-4 h-4 border-2 border-[#4A6FA5] border-t-transparent rounded-full animate-spin" />
        <p className="text-sm text-[#718096]">Uploading...</p>
      </div>
    );
  }

  return (
    <div>
      <button
        onClick={() => inputRef.current?.click()}
        className="w-full border-2 border-dashed border-[#CBD5E0] rounded-lg p-6 text-center text-sm text-[#718096] hover:border-[#4A6FA5] hover:text-[#4A6FA5] transition-colors"
      >
        Click to upload {label} audio (WAV, MP3, M4A)
      </button>
      <input
        ref={inputRef}
        type="file"
        accept="audio/*"
        className="hidden"
        onChange={(e) => {
          const f = e.target.files?.[0];
          if (f) onFile(f);
        }}
      />
      {state.status === "error" && (
        <p className="text-xs text-[#C9605A] mt-1">{state.message}</p>
      )}
    </div>
  );
}

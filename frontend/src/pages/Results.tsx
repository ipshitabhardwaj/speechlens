import { useCallback, useEffect, useRef, useState } from "react";
import { useParams, Link } from "react-router-dom";
import WaveSurfer from "wavesurfer.js";
import RegionsPlugin, { type Region } from "wavesurfer.js/dist/plugins/regions.js";
import {
  getAnalysis,
  audioUrl,
  type AnalysisResponse,
  type AnalysisResultResponse,
  type FlawResponse,
} from "../api";

const STAGE_LABELS: Record<string, string> = {
  queued: "Queued",
  decoding: "Decoding audio",
  extracting_features: "Extracting features",
  aligning: "Aligning",
  comparing: "Detecting flaws",
  scoring: "Scoring rubric",
};

const SEVERITY_COLORS: Record<string, string> = {
  mild: "rgba(234, 179, 8, 0.25)",
  moderate: "rgba(234, 134, 58, 0.30)",
  severe: "rgba(201, 96, 90, 0.35)",
};

const SEVERITY_TEXT: Record<string, string> = {
  mild: "text-[#B7791F]",
  moderate: "text-[#C05621]",
  severe: "text-[#C9605A]",
};

const SEVERITY_BG: Record<string, string> = {
  mild: "bg-[#FEFCBF]",
  moderate: "bg-[#FEEBC8]",
  severe: "bg-[#FFF5F5]",
};

export default function Results() {
  const { id } = useParams<{ id: string }>();
  const [analysis, setAnalysis] = useState<AnalysisResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  useEffect(() => {
    if (!id) return;
    let cancelled = false;

    async function poll() {
      try {
        const data = await getAnalysis(id!);
        if (cancelled) return;
        setAnalysis(data);
        setLoading(false);

        if (data.status === "complete" || data.status === "error") {
          if (pollRef.current) clearInterval(pollRef.current);
        }
      } catch (e) {
        if (cancelled) return;
        setError(e instanceof Error ? e.message : "Failed to load analysis");
        setLoading(false);
      }
    }

    poll();
    pollRef.current = setInterval(poll, 500);

    return () => {
      cancelled = true;
      if (pollRef.current) clearInterval(pollRef.current);
    };
  }, [id]);

  if (loading) {
    return <PageShell><LoadingState /></PageShell>;
  }

  if (error) {
    return <PageShell><ErrorState message={error} /></PageShell>;
  }

  if (!analysis) {
    return <PageShell><EmptyState /></PageShell>;
  }

  if (analysis.status === "error") {
    return <PageShell><ErrorState message={analysis.error || "Analysis failed"} /></PageShell>;
  }

  if (analysis.status !== "complete" || !analysis.result) {
    return (
      <PageShell>
        <div className="border border-[#E2DFD8] rounded-lg p-6 bg-white text-center">
          <div className="w-6 h-6 border-2 border-[#4A6FA5] border-t-transparent rounded-full animate-spin mx-auto mb-3" />
          <p className="text-sm text-[#718096]">
            {STAGE_LABELS[analysis.stage || "queued"] || "Processing"}...
          </p>
        </div>
      </PageShell>
    );
  }

  return (
    <PageShell>
      <ResultsView
        participantId={analysis.participant_id}
        referenceId={analysis.reference_id}
        result={analysis.result}
      />
    </PageShell>
  );
}


function PageShell({ children }: { children: React.ReactNode }) {
  return (
    <div className="min-h-screen bg-[#FAF9F6] text-[#2D3748]">
      <header className="border-b border-[#E2DFD8] px-6 py-4 flex items-center gap-4">
        <Link to="/" className="text-2xl font-semibold text-[#1A202C] hover:text-[#4A6FA5] transition-colors">
          SpeechLens
        </Link>
        <span className="text-sm text-[#718096]">Results</span>
      </header>
      <main className="max-w-4xl mx-auto px-4 py-8 space-y-8">{children}</main>
    </div>
  );
}


function LoadingState() {
  return (
    <div className="text-center py-12">
      <div className="w-6 h-6 border-2 border-[#4A6FA5] border-t-transparent rounded-full animate-spin mx-auto mb-3" />
      <p className="text-sm text-[#718096]">Loading analysis...</p>
    </div>
  );
}

function EmptyState() {
  return (
    <div className="text-center py-12">
      <p className="text-[#718096]">No analysis found.</p>
      <Link to="/" className="text-sm text-[#4A6FA5] underline mt-2 inline-block">
        Start a new analysis
      </Link>
    </div>
  );
}

function ErrorState({ message }: { message: string }) {
  return (
    <div className="border border-[#C9605A] rounded-lg p-6 bg-[#FFF5F5]">
      <p className="font-medium text-[#C9605A]">Error</p>
      <p className="text-sm text-[#C9605A] mt-1">{message}</p>
      <Link to="/" className="text-sm text-[#4A6FA5] underline mt-3 inline-block">
        Start a new analysis
      </Link>
    </div>
  );
}


function ResultsView({
  participantId,
  referenceId,
  result,
}: {
  participantId: string;
  referenceId: string;
  result: AnalysisResultResponse;
}) {
  const partWsRef = useRef<WaveSurfer | null>(null);
  const [selectedFlaw, setSelectedFlaw] = useState<number | null>(null);

  const seekToTime = useCallback((time: number) => {
    const ws = partWsRef.current;
    if (!ws) return;
    const dur = ws.getDuration();
    if (dur > 0) ws.seekTo(Math.min(time / dur, 1));
  }, []);

  const handleFlawClick = useCallback((index: number) => {
    setSelectedFlaw(index);
    seekToTime(result.flaws[index].start_seconds);
  }, [result.flaws, seekToTime]);

  return (
    <div className="space-y-8">
      {/* Overall score */}
      <section className="border border-[#E2DFD8] rounded-lg p-6 bg-white">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-lg font-medium">Overall score</h2>
            <p className="text-xs text-[#718096] mt-1">
              Rubric v{result.rubric.version} &middot; alignment: {result.alignment_method.replace("_", " ")}
            </p>
          </div>
          <div className="text-4xl font-bold" style={{ color: scoreColor(result.rubric.overall_score) }}>
            {result.rubric.overall_score.toFixed(0)}
          </div>
        </div>
      </section>

      {/* Participant waveform + flaw regions */}
      <section>
        <h2 className="text-lg font-medium mb-2">Participant recording</h2>
        <p className="text-xs text-[#718096] mb-2">
          {result.participant_duration.toFixed(1)}s
          {result.participant_rate_wpm != null && ` · ${result.participant_rate_wpm.toFixed(0)} WPM`}
        </p>
        <WaveformPlayer
          key={`part-${participantId}`}
          recordingUrl={audioUrl(participantId)}
          flaws={result.flaws}
          onReady={(ws) => { partWsRef.current = ws; }}
          onRegionClick={(flawIdx) => handleFlawClick(flawIdx)}
          isParticipant
        />
      </section>

      {/* Reference waveform */}
      <section>
        <h2 className="text-lg font-medium mb-2">Reference recording</h2>
        <p className="text-xs text-[#718096] mb-2">
          {result.reference_duration.toFixed(1)}s
          {result.reference_rate_wpm != null && ` · ${result.reference_rate_wpm.toFixed(0)} WPM`}
        </p>
        <WaveformPlayer
          key={`ref-${referenceId}`}
          recordingUrl={audioUrl(referenceId)}
          flaws={[]}
          isParticipant={false}
        />
      </section>

      {/* Rubric breakdown */}
      <section className="border border-[#E2DFD8] rounded-lg p-6 bg-white">
        <h2 className="text-lg font-medium mb-4">Rubric breakdown</h2>
        <div className="space-y-3">
          {result.rubric.dimensions.map((dim) => (
            <div key={dim.name}>
              <div className="flex items-center justify-between mb-1">
                <span className="text-sm font-medium capitalize">{dim.name.replace("_", " ")}</span>
                <span className="text-sm">
                  {dim.available ? (
                    <span style={{ color: scoreColor(dim.score) }}>{dim.score.toFixed(0)}</span>
                  ) : (
                    <span className="text-[#A0AEC0] italic">unavailable</span>
                  )}
                  <span className="text-[#A0AEC0] ml-1">({(dim.weight * 100).toFixed(0)}%)</span>
                </span>
              </div>
              {dim.available && (
                <div className="w-full h-2 bg-[#EDF2F7] rounded-full overflow-hidden">
                  <div
                    className="h-full rounded-full transition-all"
                    style={{
                      width: `${Math.max(dim.score, 0)}%`,
                      backgroundColor: scoreColor(dim.score),
                    }}
                  />
                </div>
              )}
              {!dim.available && dim.unavailable_reason && (
                <p className="text-xs text-[#A0AEC0] italic">{dim.unavailable_reason}</p>
              )}
            </div>
          ))}
        </div>
      </section>

      {/* Flaws list */}
      <section>
        <h2 className="text-lg font-medium mb-4">
          Detected flaws
          {result.flaws.length === 0 && (
            <span className="text-sm font-normal text-[#48886B] ml-2">None detected</span>
          )}
        </h2>
        <div className="space-y-3">
          {result.flaws.map((flaw, i) => (
            <FlawCard
              key={i}
              flaw={flaw}
              index={i}
              selected={selectedFlaw === i}
              onClick={() => handleFlawClick(i)}
            />
          ))}
        </div>
      </section>

      {/* Unavailable panel */}
      {result.unavailable.length > 0 && (
        <section className="border border-[#E2DFD8] rounded-lg p-6 bg-white">
          <h2 className="text-lg font-medium mb-3">Unavailable</h2>
          <p className="text-xs text-[#718096] mb-2">
            These features could not be computed for this analysis.
          </p>
          <ul className="space-y-1">
            {result.unavailable.map((reason, i) => (
              <li key={i} className="text-sm text-[#A0AEC0] flex items-start gap-2">
                <span className="mt-0.5">&#9679;</span>
                <span>{reason}</span>
              </li>
            ))}
          </ul>
        </section>
      )}

      {/* Back link */}
      <div className="text-center pb-8">
        <Link to="/" className="text-sm text-[#4A6FA5] underline">
          New analysis
        </Link>
      </div>
    </div>
  );
}


function WaveformPlayer({
  recordingUrl,
  flaws,
  onReady,
  onRegionClick,
  isParticipant,
}: {
  recordingUrl: string;
  flaws: FlawResponse[];
  onReady?: (ws: WaveSurfer) => void;
  onRegionClick?: (flawIdx: number) => void;
  isParticipant: boolean;
}) {
  const containerRef = useRef<HTMLDivElement>(null);
  const wsRef = useRef<WaveSurfer | null>(null);
  const [playing, setPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);

  useEffect(() => {
    if (!containerRef.current) return;

    const regions = RegionsPlugin.create();
    const ws = WaveSurfer.create({
      container: containerRef.current,
      url: recordingUrl,
      waveColor: isParticipant ? "#4A6FA5" : "#718096",
      progressColor: isParticipant ? "#1A202C" : "#2D3748",
      cursorColor: "#C9605A",
      height: 80,
      barWidth: 2,
      barGap: 1,
      barRadius: 2,
      plugins: [regions],
    });

    ws.on("ready", () => {
      setDuration(ws.getDuration());
      onReady?.(ws);

      if (isParticipant && flaws.length > 0) {
        flaws.forEach((flaw, i) => {
          const region = regions.addRegion({
            start: flaw.start_seconds,
            end: flaw.end_seconds,
            color: SEVERITY_COLORS[flaw.severity] || "rgba(200, 200, 200, 0.3)",
            drag: false,
            resize: false,
          });
          region.on("click", () => onRegionClick?.(i));
        });
      }
    });

    ws.on("timeupdate", (t: number) => setCurrentTime(t));
    ws.on("play", () => setPlaying(true));
    ws.on("pause", () => setPlaying(false));

    wsRef.current = ws;

    return () => {
      ws.destroy();
      wsRef.current = null;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [recordingUrl, isParticipant]);

  return (
    <div className="border border-[#E2DFD8] rounded-lg p-4 bg-white">
      <div ref={containerRef} />
      <div className="flex items-center gap-3 mt-3">
        <button
          onClick={() => wsRef.current?.playPause()}
          className="px-4 py-1.5 rounded text-sm font-medium bg-[#1A202C] text-white hover:bg-[#2D3748] transition-colors"
        >
          {playing ? "Pause" : "Play"}
        </button>
        <span className="text-xs text-[#718096] tabular-nums">
          {formatTime(currentTime)} / {formatTime(duration)}
        </span>
      </div>
    </div>
  );
}


function FlawCard({
  flaw,
  index,
  selected,
  onClick,
}: {
  flaw: FlawResponse;
  index: number;
  selected: boolean;
  onClick: () => void;
}) {
  const bg = selected ? SEVERITY_BG[flaw.severity] || "bg-[#F7FAFC]" : "bg-white";
  const border = selected ? "border-[#4A6FA5]" : "border-[#E2DFD8]";

  return (
    <div
      onClick={onClick}
      className={`border ${border} rounded-lg p-4 ${bg} cursor-pointer hover:border-[#4A6FA5] transition-colors`}
    >
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-2">
          <span className={`text-xs font-medium px-2 py-0.5 rounded ${SEVERITY_BG[flaw.severity]} ${SEVERITY_TEXT[flaw.severity]}`}>
            {flaw.severity}
          </span>
          <span className="text-sm font-medium capitalize">{flaw.rule_id.replace("_", " ")}</span>
        </div>
        <span className="text-xs text-[#718096] tabular-nums">
          {flaw.start_seconds.toFixed(1)}s &ndash; {flaw.end_seconds.toFixed(1)}s
        </span>
      </div>

      {/* Evidence (observed) */}
      <div className="mb-2">
        <p className="text-xs font-medium text-[#718096] mb-0.5">Evidence</p>
        <p className="text-sm">{flaw.evidence}</p>
      </div>

      {/* Interpretation (inferred) */}
      <div className="mb-2">
        <p className="text-xs font-medium text-[#718096] mb-0.5">Interpretation</p>
        <p className="text-sm italic text-[#4A5568]">{flaw.interpretation}</p>
      </div>

      {/* Measurements */}
      <div className="flex gap-6 text-xs text-[#718096]">
        <div>
          <span className="font-medium">Participant: </span>
          {Object.entries(flaw.participant_measurement).map(([k, v]) => (
            <span key={k}>{k.replace(/_/g, " ")} = {typeof v === "number" ? v.toFixed(2) : v}</span>
          ))}
        </div>
        <div>
          <span className="font-medium">Deviation: </span>
          {flaw.deviation.toFixed(2)}
        </div>
      </div>

      {/* Recommendation */}
      <p className="text-xs text-[#4A6FA5] mt-2">{flaw.recommendation}</p>
    </div>
  );
}


function scoreColor(score: number): string {
  if (score >= 80) return "#48886B";
  if (score >= 60) return "#B7791F";
  return "#C9605A";
}

function formatTime(seconds: number): string {
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m}:${s.toString().padStart(2, "0")}`;
}

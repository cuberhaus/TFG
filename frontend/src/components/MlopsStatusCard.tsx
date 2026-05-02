import { useEffect, useState } from 'react';
import {
  Activity,
  ChevronDown,
  ChevronUp,
  ExternalLink,
  Database,
  BarChart3,
  Search,
  Sliders,
  Play,
  ArrowRight,
  Copy,
  Check,
  Info,
} from 'lucide-react';
import { api, getOrCreateSessionId } from '../api';

type DriftStatus = 'ok' | 'warning' | 'critical' | 'unknown';

interface MlopsStats {
  enabled: boolean;
  predictions_today?: number;
  predictions_with_boxes_today?: number;
  distinct_sessions_today?: number;
  avg_latency_ms_today?: number;
  predictions_last_hour?: number;
  last_prediction_at?: string | null;
  drift_status?: DriftStatus;
  mlflow_url?: string;
  evidently_url?: string;
  error?: string;
}

const DRIFT_BADGE: Record<DriftStatus, { label: string; bg: string; text: string; dot: string }> = {
  ok:       { label: 'No drift',        bg: 'bg-emerald-500/15 border-emerald-500/40', text: 'text-emerald-300', dot: 'bg-emerald-400' },
  warning:  { label: 'Minor drift',     bg: 'bg-amber-500/15 border-amber-500/40',     text: 'text-amber-300',   dot: 'bg-amber-400'   },
  critical: { label: 'Drift detected',  bg: 'bg-rose-500/15 border-rose-500/40',       text: 'text-rose-300',    dot: 'bg-rose-400'    },
  unknown:  { label: 'Drift unknown',   bg: 'bg-gray-700/40 border-gray-600/50',       text: 'text-gray-300',    dot: 'bg-gray-500'    },
};

const POLL_INTERVAL_MS = 30_000;

export default function MlopsStatusCard() {
  const [stats, setStats] = useState<MlopsStats | null>(null);
  // Default to collapsed: the header alone (drift badge + dot) is enough
  // to monitor at a glance, and the full card was overlapping content
  // (especially in the Dataset Explorer's bottom-row thumbnails).
  const [collapsed, setCollapsed] = useState(true);
  const [showHelp, setShowHelp] = useState(false);
  const [copied, setCopied] = useState(false);
  const [sessionId] = useState(() => getOrCreateSessionId());

  useEffect(() => {
    let cancelled = false;

    const tick = async () => {
      try {
        const { data } = await api.get<MlopsStats>('/api/mlops/stats');
        if (!cancelled) setStats(data);
      } catch (err) {
        if (!cancelled) setStats({ enabled: false, error: (err as Error).message });
      }
    };

    tick();
    const id = window.setInterval(tick, POLL_INTERVAL_MS);
    return () => {
      cancelled = true;
      window.clearInterval(id);
    };
  }, []);

  const copySessionId = async () => {
    try {
      await navigator.clipboard.writeText(sessionId);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1500);
    } catch {
      // Clipboard API may be blocked (e.g. inside an iframe without
      // permission). Fall back to a no-op — the id is already on screen
      // and selectable manually.
    }
  };

  if (!stats) {
    return null;
  }

  if (!stats.enabled) {
    return (
      <div className="fixed bottom-4 right-4 z-30 max-w-xs">
        <div className="bg-gray-800/95 border border-gray-700 rounded-lg shadow-xl p-3 text-xs">
          <div className="flex items-center gap-2 text-gray-400">
            <Activity className="w-4 h-4" />
            <span className="font-medium">MLOps observability offline</span>
          </div>
          <div className="text-gray-500 mt-1.5 leading-relaxed">
            Run{' '}
            <code className="bg-gray-900 px-1.5 py-0.5 rounded text-gray-300">make mlops-up</code>{' '}
            to start logging every inference, tracking every training run, and
            watching for input drift.
          </div>
        </div>
      </div>
    );
  }

  const drift = DRIFT_BADGE[stats.drift_status ?? 'unknown'];

  // Collapsed view — a small inline pill with just an activity icon and
  // a coloured drift dot. ~110-130px wide instead of the previous full
  // 320px card, so it stops overlapping image thumbnails and other
  // content. Click anywhere on it to expand back to the full card.
  if (collapsed) {
    return (
      <div className="fixed bottom-4 right-4 z-30">
        <button
          onClick={() => setCollapsed(false)}
          title={`MLOps observability · ${drift.label}`}
          className="group flex items-center gap-2 bg-gray-800/95 backdrop-blur-sm border border-gray-700 hover:border-gray-600 rounded-full pl-2.5 pr-3 py-1.5 shadow-xl transition-colors"
        >
          <Activity className="w-3.5 h-3.5 text-blue-400 flex-shrink-0" />
          <span className="text-[11px] font-medium text-gray-200">MLOps</span>
          <span
            className={`w-2 h-2 rounded-full ${drift.dot} flex-shrink-0`}
            aria-label={drift.label}
          />
          <ChevronUp className="w-3 h-3 text-gray-500 group-hover:text-gray-300 transition-colors" />
        </button>
      </div>
    );
  }

  // Expanded view (full card). Only rendered when the user clicked the
  // collapsed pill; closing returns to the small pill — the X button
  // hides the whole component for the rest of the session.
  return (
    <div className="fixed bottom-4 right-4 z-30 w-80 max-w-[calc(100vw-2rem)]">
      <div className="bg-gray-800/95 backdrop-blur-sm border border-gray-700 rounded-lg shadow-2xl overflow-hidden">
        {/* Header — click anywhere to collapse, dedicated X to hide. */}
        <div className="w-full px-3 py-2 flex items-center justify-between bg-gray-800 border-b border-gray-700">
          <button
            onClick={() => setCollapsed(true)}
            className="flex items-center gap-2 min-w-0 flex-1 hover:opacity-80 transition-opacity"
          >
            <Activity className="w-4 h-4 text-blue-400 flex-shrink-0" />
            <span className="text-sm font-semibold text-gray-200">MLOps observability</span>
            <span
              className={`text-[10px] uppercase tracking-wider px-2 py-0.5 rounded-full border ${drift.bg} ${drift.text} flex-shrink-0`}
            >
              {drift.label}
            </span>
          </button>
          <button
            onClick={() => setCollapsed(true)}
            className="text-gray-500 hover:text-gray-300 transition-colors p-0.5 flex-shrink-0 ml-2"
            title="Minimise"
            aria-label="Minimise"
          >
            <ChevronDown className="w-4 h-4" />
          </button>
        </div>

        <div className="p-3 space-y-3 text-xs text-gray-300">
          {/* What writes the data — answers "how do I make stuff
              show up here?" without leaving the page. */}
          <div>
              <button
                onClick={() => setShowHelp((s) => !s)}
                className="w-full flex items-center justify-between text-[10px] uppercase tracking-wider text-gray-500 hover:text-gray-400 transition-colors"
              >
                <span className="flex items-center gap-1.5">
                  <Info className="w-3 h-3" />
                  What gets logged?
                </span>
                {showHelp ? (
                  <ChevronUp className="w-3 h-3" />
                ) : (
                  <ChevronDown className="w-3 h-3" />
                )}
              </button>

              {showHelp && (
                <div className="mt-2 space-y-1.5 rounded-md bg-gray-900/70 border border-gray-700 p-2.5">
                  <FlowRow
                    Icon={Search}
                    iconClass="text-cyan-400"
                    action="Inference tab"
                    arrow="→ 1 row in prediction-log per upload"
                    sink="Evidently"
                    sinkClass="text-purple-400"
                  />
                  <FlowRow
                    Icon={Sliders}
                    iconClass="text-fuchsia-400"
                    action="Hyperparameter Tuning"
                    arrow="→ 1 nested MLflow run per Optuna trial"
                    sink="MLflow"
                    sinkClass="text-blue-400"
                  />
                  <FlowRow
                    Icon={Play}
                    iconClass="text-teal-400"
                    action="Detection Training"
                    arrow="→ 1 MLflow run per script invocation"
                    sink="MLflow"
                    sinkClass="text-blue-400"
                  />
                  <p className="pt-1 text-[10px] text-gray-500 leading-relaxed">
                    Every backend call from this tab carries the session id
                    below — paste it into Sentry to pull the matching traces.
                  </p>
                </div>
              )}
            </div>

            <div className="grid grid-cols-2 gap-2">
              <Stat label="Predictions today" value={stats.predictions_today ?? 0} />
              <Stat label="With boxes" value={stats.predictions_with_boxes_today ?? 0} />
              <Stat label="Last hour" value={stats.predictions_last_hour ?? 0} />
              <Stat label="Distinct sessions" value={stats.distinct_sessions_today ?? 0} />
            </div>

            {typeof stats.avg_latency_ms_today === 'number' && (
              <div className="text-[11px] text-gray-500">
                Avg latency today: {stats.avg_latency_ms_today} ms
              </div>
            )}

            {/* Where to find the data — clearly captioned. */}
            <div className="pt-2 border-t border-gray-700 space-y-2">
              <div className="text-[10px] uppercase tracking-wider text-gray-500">
                Where to look
              </div>
              <DeepLink
                href={stats.mlflow_url ?? 'http://localhost:15000'}
                Icon={BarChart3}
                iconClass="text-blue-400"
                hoverClass="hover:text-blue-300"
                title="MLflow — training runs"
                caption="Compare experiments, promote a model"
              />
              <DeepLink
                href={stats.evidently_url ?? 'http://localhost:15001'}
                Icon={Database}
                iconClass="text-purple-400"
                hoverClass="hover:text-purple-300"
                title="Evidently — input drift"
                caption="Are uploads shifting away from training data?"
              />
            </div>

            <div className="pt-2 border-t border-gray-700">
              <div className="flex items-center justify-between text-[10px] uppercase tracking-wider text-gray-500 mb-1">
                <span>Tab session id</span>
                <button
                  onClick={copySessionId}
                  className="flex items-center gap-1 normal-case tracking-normal text-[10px] text-gray-400 hover:text-gray-200 transition-colors"
                  title="Copy to clipboard, then paste into Sentry's search"
                >
                  {copied ? (
                    <>
                      <Check className="w-3 h-3 text-emerald-400" />
                      Copied
                    </>
                  ) : (
                    <>
                      <Copy className="w-3 h-3" />
                      Copy
                    </>
                  )}
                </button>
              </div>
              <code className="block text-[10px] text-gray-400 font-mono break-all leading-snug">
                {sessionId}
              </code>
            <p className="mt-1 text-[10px] text-gray-500 leading-relaxed">
              Tags every backend call + prediction-log row. Paste into
              Sentry's search to find this tab's traces.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}

function Stat({ label, value }: { label: string; value: number }) {
  return (
    <div className="bg-gray-900/60 border border-gray-700 rounded px-2.5 py-1.5">
      <div className="text-[10px] uppercase tracking-wider text-gray-500">{label}</div>
      <div className="text-sm font-semibold text-gray-100 tabular-nums">{value}</div>
    </div>
  );
}

interface FlowRowProps {
  Icon: typeof Search;
  iconClass: string;
  action: string;
  arrow: string;
  sink: string;
  sinkClass: string;
}

function FlowRow({ Icon, iconClass, action, arrow, sink, sinkClass }: FlowRowProps) {
  return (
    <div className="flex items-start gap-1.5 text-[11px] leading-snug">
      <Icon className={`w-3.5 h-3.5 mt-0.5 flex-shrink-0 ${iconClass}`} />
      <div className="text-gray-300">
        <span className="font-medium text-gray-200">{action}</span>{' '}
        <span className="text-gray-500">{arrow}</span>{' '}
        <ArrowRight className="inline w-3 h-3 text-gray-600 mx-0.5 -mt-0.5" />{' '}
        <span className={`font-medium ${sinkClass}`}>{sink}</span>
      </div>
    </div>
  );
}

interface DeepLinkProps {
  href: string;
  Icon: typeof BarChart3;
  iconClass: string;
  hoverClass: string;
  title: string;
  caption: string;
}

function DeepLink({ href, Icon, iconClass, hoverClass, title, caption }: DeepLinkProps) {
  return (
    <a
      href={href}
      target="_blank"
      rel="noreferrer"
      className={`group flex items-start gap-2 rounded-md border border-gray-700 bg-gray-900/40 hover:bg-gray-900/70 hover:border-gray-600 px-2.5 py-1.5 transition-colors`}
    >
      <Icon className={`w-3.5 h-3.5 mt-0.5 flex-shrink-0 ${iconClass}`} />
      <div className="flex-1 min-w-0">
        <div className={`text-xs font-medium text-gray-200 ${hoverClass} transition-colors flex items-center gap-1`}>
          {title}
          <ExternalLink className="w-3 h-3 opacity-0 group-hover:opacity-100 transition-opacity" />
        </div>
        <div className="text-[10px] text-gray-500 leading-snug">{caption}</div>
      </div>
    </a>
  );
}

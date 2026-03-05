import { useEffect, useMemo, useState } from "react";
import {
  Bar,
  BarChart,
  Cell,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis
} from "recharts";

const API_BASE_URL = "http://localhost:8000";
const POLL_INTERVAL_MS = 5000;
const CATEGORY_OPTIONS = ["all", "question", "complaint", "sales", "spam", "other"] as const;
const CLASSIFIER_OPTIONS = ["all", "lmstudio", "rules"] as const;
const STATUS_OPTIONS = ["all", "ok", "error"] as const;
const PIE_COLORS = ["#0ea5e9", "#06b6d4", "#14b8a6", "#22c55e", "#f97316", "#a855f7"];

type Category = "question" | "complaint" | "sales" | "spam" | "other";
type ClassifierUsed = "lmstudio" | "rules";

type ClassifyResult = {
  category: Category;
  confidence: number;
  suggested_reply: string;
  classifier_used?: ClassifierUsed;
  latency_ms?: number;
  request_id?: string;
};

type InfoResponse = {
  active_classifier: string;
  model: string | null;
  version: string;
};

type StatsResponse = {
  total_requests: number;
  ok_rate: number;
  avg_latency_ms: number;
  avg_confidence: number;
  category_counts: Record<string, number>;
  classifier_counts: Record<string, number>;
  errors_last_10: Array<{
    request_id: string;
    classifier_name: string;
    error_message: string;
    created_at: string;
  }>;
  last_updated_iso: string;
};

type RecentRow = {
  request_id: string;
  text: string;
  category: string | null;
  confidence: number | null;
  suggested_reply: string | null;
  classifier_name: string;
  latency_ms: number;
  ok: boolean;
  error_message: string | null;
  created_at: string;
};

type BatchItem = {
  text: string;
  result: ClassifyResult;
};

async function apiGet<T>(path: string): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`);
  if (!response.ok) {
    throw new Error(`${response.status} ${response.statusText}`);
  }
  return (await response.json()) as T;
}

async function apiPost<T>(path: string, payload: unknown): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload)
  });
  if (!response.ok) {
    throw new Error(`${response.status} ${response.statusText}`);
  }
  return (await response.json()) as T;
}

function truncate(value: string, max = 90): string {
  return value.length > max ? `${value.slice(0, max)}...` : value;
}

function formatTime(iso: string): string {
  return new Date(iso).toLocaleTimeString();
}

function useDarkMode(): [boolean, () => void] {
  const [isDark, setIsDark] = useState<boolean>(() => {
    const fromStorage = localStorage.getItem("inboxpilot-dark");
    if (fromStorage) {
      return fromStorage === "true";
    }
    return window.matchMedia("(prefers-color-scheme: dark)").matches;
  });

  useEffect(() => {
    document.documentElement.classList.toggle("dark", isDark);
    localStorage.setItem("inboxpilot-dark", String(isDark));
  }, [isDark]);

  return [isDark, () => setIsDark((prev) => !prev)];
}

function confidenceToPercent(value: number): number {
  return Math.max(0, Math.min(100, Math.round(value * 100)));
}

function App() {
  const [isDark, toggleDarkMode] = useDarkMode();
  const [backendOnline, setBackendOnline] = useState(false);
  const [info, setInfo] = useState<InfoResponse | null>(null);
  const [stats, setStats] = useState<StatsResponse | null>(null);
  const [recentRows, setRecentRows] = useState<RecentRow[]>([]);
  const [isLivePaused, setIsLivePaused] = useState(false);
  const [toast, setToast] = useState("");

  const [singleMessage, setSingleMessage] = useState("");
  const [singleLoading, setSingleLoading] = useState(false);
  const [singleResult, setSingleResult] = useState<ClassifyResult | null>(null);

  const [activeClassifyTab, setActiveClassifyTab] = useState<"single" | "batch">("single");
  const [batchInput, setBatchInput] = useState("");
  const [batchLoading, setBatchLoading] = useState(false);
  const [batchResults, setBatchResults] = useState<BatchItem[]>([]);

  const [categoryFilter, setCategoryFilter] =
    useState<(typeof CATEGORY_OPTIONS)[number]>("all");
  const [classifierFilter, setClassifierFilter] =
    useState<(typeof CLASSIFIER_OPTIONS)[number]>("all");
  const [statusFilter, setStatusFilter] = useState<(typeof STATUS_OPTIONS)[number]>("all");
  const [searchQuery, setSearchQuery] = useState("");
  const [selectedRow, setSelectedRow] = useState<RecentRow | null>(null);

  const categoryChartData = useMemo(
    () =>
      Object.entries(stats?.category_counts ?? {}).map(([name, value]) => ({
        name,
        value
      })),
    [stats]
  );

  const classifierChartData = useMemo(
    () =>
      Object.entries(stats?.classifier_counts ?? {}).map(([name, value]) => ({
        name,
        value
      })),
    [stats]
  );

  const recentQueryPath = useMemo(() => {
    const params = new URLSearchParams({ limit: "20" });
    if (categoryFilter !== "all") params.set("category", categoryFilter);
    if (classifierFilter !== "all") params.set("classifier", classifierFilter);
    if (statusFilter !== "all") params.set("status", statusFilter);
    if (searchQuery.trim()) params.set("q", searchQuery.trim());
    return `/recent?${params.toString()}`;
  }, [categoryFilter, classifierFilter, statusFilter, searchQuery]);

  const showToast = (message: string): void => {
    setToast(message);
    window.setTimeout(() => setToast(""), 2800);
  };

  const refreshHeader = async () => {
    try {
      await apiGet<{ status: string }>("/health");
      setBackendOnline(true);
      const infoData = await apiGet<InfoResponse>("/info");
      setInfo(infoData);
    } catch {
      setBackendOnline(false);
    }
  };

  const refreshLiveData = async () => {
    try {
      const [statsData, recentData] = await Promise.all([
        apiGet<StatsResponse>("/stats?window_minutes=60"),
        apiGet<RecentRow[]>(recentQueryPath)
      ]);
      setStats(statsData);
      setRecentRows(recentData);
    } catch (error) {
      showToast(`Failed to refresh live data: ${(error as Error).message}`);
    }
  };

  useEffect(() => {
    void refreshHeader();
    void refreshLiveData();
  }, [recentQueryPath]);

  useEffect(() => {
    if (isLivePaused) {
      return;
    }
    const timer = window.setInterval(() => {
      void refreshHeader();
      void refreshLiveData();
    }, POLL_INTERVAL_MS);
    return () => window.clearInterval(timer);
  }, [isLivePaused, recentQueryPath]);

  const classifySingle = async () => {
    if (!singleMessage.trim()) {
      showToast("Enter a message to classify.");
      return;
    }
    setSingleLoading(true);
    try {
      const response = await apiPost<ClassifyResult>("/classify", { text: singleMessage });
      setSingleResult(response);
      await refreshLiveData();
    } catch (error) {
      showToast(`Classify failed: ${(error as Error).message}`);
    } finally {
      setSingleLoading(false);
    }
  };

  const classifyBatch = async () => {
    const messages = batchInput
      .split("\n")
      .map((line) => line.trim())
      .filter(Boolean);
    if (messages.length === 0) {
      showToast("Enter one message per line.");
      return;
    }

    setBatchLoading(true);
    try {
      try {
        const batchResponse = await apiPost<{ results: Array<Omit<ClassifyResult, "classifier_used">> }>(
          "/classify/batch",
          { texts: messages }
        );
        const classifier: ClassifierUsed =
          info?.active_classifier === "rules" ? "rules" : "lmstudio";
        const items: BatchItem[] = batchResponse.results.map((result, index) => ({
          text: messages[index] ?? "",
          result: {
            ...result,
            classifier_used: classifier
          }
        }));
        setBatchResults(items);
      } catch {
        const items = await classifySequentialFallback(messages, 3);
        setBatchResults(items);
      }
      await refreshLiveData();
    } catch (error) {
      showToast(`Batch classify failed: ${(error as Error).message}`);
    } finally {
      setBatchLoading(false);
    }
  };

  const classifySequentialFallback = async (
    messages: string[],
    concurrency: number
  ): Promise<BatchItem[]> => {
    const results: BatchItem[] = new Array(messages.length);
    let nextIndex = 0;

    const worker = async () => {
      while (nextIndex < messages.length) {
        const current = nextIndex;
        nextIndex += 1;
        const text = messages[current];
        try {
          const result = await apiPost<ClassifyResult>("/classify", { text });
          results[current] = { text, result };
        } catch (error) {
          results[current] = {
            text,
            result: {
              category: "other",
              confidence: 0,
              suggested_reply: `Failed: ${(error as Error).message}`,
              classifier_used: "rules"
            }
          };
        }
      }
    };

    const workers = Array.from({ length: Math.min(concurrency, messages.length) }, () => worker());
    await Promise.all(workers);
    return results;
  };

  const copyReply = async (value: string) => {
    try {
      await navigator.clipboard.writeText(value);
      showToast("Suggested reply copied.");
    } catch {
      showToast("Could not copy to clipboard.");
    }
  };

  return (
    <div className="min-h-screen bg-slate-50 text-slate-800 transition-colors dark:bg-slate-950 dark:text-slate-100">
      <header className="sticky top-0 z-20 border-b border-slate-200 bg-white/95 px-4 py-4 backdrop-blur dark:border-slate-800 dark:bg-slate-900/95">
        <div className="mx-auto flex w-full max-w-7xl flex-col gap-3 md:flex-row md:items-center md:justify-between">
          <div>
            <h1 className="text-xl font-semibold">InboxPilot Dashboard</h1>
            <p className="text-sm text-slate-500 dark:text-slate-400">
              Internal admin view for real-time classification and monitoring
            </p>
          </div>
          <div className="flex flex-wrap items-center gap-3">
            <span
              className={`rounded-full px-3 py-1 text-xs font-semibold ${
                backendOnline
                  ? "bg-emerald-100 text-emerald-700 dark:bg-emerald-900/40 dark:text-emerald-300"
                  : "bg-rose-100 text-rose-700 dark:bg-rose-900/40 dark:text-rose-300"
              }`}
            >
              Backend {backendOnline ? "online" : "offline"}
            </span>
            <span className="rounded-full bg-slate-100 px-3 py-1 text-xs font-semibold text-slate-700 dark:bg-slate-800 dark:text-slate-200">
              {info?.active_classifier ?? "unknown"}
              {info?.model ? ` (${info.model})` : ""}
            </span>
            <button
              type="button"
              onClick={toggleDarkMode}
              className="rounded-lg border border-slate-300 px-3 py-1.5 text-sm hover:bg-slate-100 dark:border-slate-700 dark:hover:bg-slate-800"
            >
              {isDark ? "Light mode" : "Dark mode"}
            </button>
          </div>
        </div>
      </header>

      <main className="mx-auto grid w-full max-w-7xl gap-4 px-4 py-4 lg:grid-cols-2">
        <section className="rounded-xl border border-slate-200 bg-white p-4 shadow-soft dark:border-slate-800 dark:bg-slate-900">
          <div className="mb-4 flex items-center justify-between">
            <h2 className="text-lg font-semibold">Classify</h2>
            <div className="flex gap-2 rounded-lg bg-slate-100 p-1 dark:bg-slate-800">
              <button
                type="button"
                onClick={() => setActiveClassifyTab("single")}
                className={`rounded-md px-3 py-1 text-sm ${
                  activeClassifyTab === "single"
                    ? "bg-white text-slate-900 dark:bg-slate-700 dark:text-white"
                    : "text-slate-600 dark:text-slate-300"
                }`}
              >
                Single
              </button>
              <button
                type="button"
                onClick={() => setActiveClassifyTab("batch")}
                className={`rounded-md px-3 py-1 text-sm ${
                  activeClassifyTab === "batch"
                    ? "bg-white text-slate-900 dark:bg-slate-700 dark:text-white"
                    : "text-slate-600 dark:text-slate-300"
                }`}
              >
                Batch
              </button>
            </div>
          </div>

          {activeClassifyTab === "single" ? (
            <div className="space-y-3">
              <textarea
                value={singleMessage}
                onChange={(event) => setSingleMessage(event.target.value)}
                rows={6}
                placeholder="Paste a customer message..."
                className="w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm outline-none ring-sky-200 focus:ring dark:border-slate-700 dark:bg-slate-950"
              />
              <button
                type="button"
                onClick={classifySingle}
                disabled={singleLoading}
                className="rounded-lg bg-sky-600 px-4 py-2 text-sm font-semibold text-white hover:bg-sky-500 disabled:cursor-not-allowed disabled:bg-sky-300"
              >
                {singleLoading ? "Classifying..." : "Classify"}
              </button>

              {singleResult && (
                <div className="space-y-3 rounded-lg border border-slate-200 bg-slate-50 p-3 dark:border-slate-700 dark:bg-slate-950">
                  <div className="flex items-center justify-between">
                    <span className="rounded-full bg-sky-100 px-2 py-1 text-xs font-semibold uppercase text-sky-700 dark:bg-sky-900/30 dark:text-sky-300">
                      {singleResult.category}
                    </span>
                    <span className="text-xs text-slate-500 dark:text-slate-400">
                      {confidenceToPercent(singleResult.confidence)}%
                    </span>
                  </div>
                  <div className="h-2 rounded-full bg-slate-200 dark:bg-slate-700">
                    <div
                      className="h-2 rounded-full bg-sky-500"
                      style={{ width: `${confidenceToPercent(singleResult.confidence)}%` }}
                    />
                  </div>
                  <div className="rounded-md border border-slate-200 bg-white p-3 text-sm dark:border-slate-700 dark:bg-slate-900">
                    {singleResult.suggested_reply}
                  </div>
                  <button
                    type="button"
                    onClick={() => copyReply(singleResult.suggested_reply)}
                    className="text-xs font-semibold text-sky-700 hover:underline dark:text-sky-300"
                  >
                    Copy suggested reply
                  </button>
                  <dl className="grid grid-cols-1 gap-2 text-xs text-slate-500 dark:text-slate-400 md:grid-cols-3">
                    <div>
                      <dt className="font-semibold">Classifier</dt>
                      <dd>{singleResult.classifier_used ?? "-"}</dd>
                    </div>
                    <div>
                      <dt className="font-semibold">Latency</dt>
                      <dd>{singleResult.latency_ms ?? "-"} ms</dd>
                    </div>
                    <div>
                      <dt className="font-semibold">Request ID</dt>
                      <dd className="truncate">{singleResult.request_id ?? "-"}</dd>
                    </div>
                  </dl>
                </div>
              )}
            </div>
          ) : (
            <div className="space-y-3">
              <textarea
                value={batchInput}
                onChange={(event) => setBatchInput(event.target.value)}
                rows={7}
                placeholder="One message per line..."
                className="w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm outline-none ring-sky-200 focus:ring dark:border-slate-700 dark:bg-slate-950"
              />
              <button
                type="button"
                onClick={classifyBatch}
                disabled={batchLoading}
                className="rounded-lg bg-teal-600 px-4 py-2 text-sm font-semibold text-white hover:bg-teal-500 disabled:cursor-not-allowed disabled:bg-teal-300"
              >
                {batchLoading ? "Running batch..." : "Batch classify"}
              </button>
              <div className="space-y-2">
                {batchResults.map((item, index) => (
                  <div
                    key={`${index}-${item.text}`}
                    className="rounded-lg border border-slate-200 p-3 text-sm dark:border-slate-700"
                  >
                    <p className="mb-1 truncate font-medium">{item.text}</p>
                    <p className="text-slate-600 dark:text-slate-300">
                      <span className="font-semibold uppercase">{item.result.category}</span> ·{" "}
                      {confidenceToPercent(item.result.confidence)}% ·{" "}
                      {item.result.classifier_used ?? "-"}
                    </p>
                    <p className="mt-1 text-slate-500 dark:text-slate-400">
                      {item.result.suggested_reply}
                    </p>
                    <p className="mt-1 text-xs text-slate-500 dark:text-slate-400">
                      latency: {item.result.latency_ms ?? "-"} ms · request:{" "}
                      {item.result.request_id ?? "-"}
                    </p>
                  </div>
                ))}
              </div>
            </div>
          )}
        </section>

        <section className="rounded-xl border border-slate-200 bg-white p-4 shadow-soft dark:border-slate-800 dark:bg-slate-900">
          <div className="mb-4 flex items-center justify-between">
            <h2 className="text-lg font-semibold">Live Metrics</h2>
            <button
              type="button"
              onClick={() => setIsLivePaused((prev) => !prev)}
              className="rounded-lg border border-slate-300 px-3 py-1.5 text-sm hover:bg-slate-100 dark:border-slate-700 dark:hover:bg-slate-800"
            >
              {isLivePaused ? "Resume live updates" : "Pause live updates"}
            </button>
          </div>

          {stats ? (
            <div className="space-y-4">
              <div className="grid grid-cols-2 gap-3">
                <StatCard label="Total requests" value={String(stats.total_requests)} />
                <StatCard label="OK rate" value={`${Math.round(stats.ok_rate * 100)}%`} />
                <StatCard label="Avg latency" value={`${stats.avg_latency_ms} ms`} />
                <StatCard
                  label="Avg confidence"
                  value={`${Math.round(stats.avg_confidence * 100)}%`}
                />
              </div>
              <div className="h-56 rounded-lg border border-slate-200 p-2 dark:border-slate-700">
                <h3 className="mb-2 text-sm font-semibold text-slate-600 dark:text-slate-300">
                  Category distribution
                </h3>
                <ResponsiveContainer width="100%" height="85%">
                  <PieChart>
                    <Pie data={categoryChartData} dataKey="value" nameKey="name" outerRadius={70}>
                      {categoryChartData.map((entry, index) => (
                        <Cell key={entry.name} fill={PIE_COLORS[index % PIE_COLORS.length]} />
                      ))}
                    </Pie>
                    <Tooltip />
                  </PieChart>
                </ResponsiveContainer>
              </div>
              <div className="h-56 rounded-lg border border-slate-200 p-2 dark:border-slate-700">
                <h3 className="mb-2 text-sm font-semibold text-slate-600 dark:text-slate-300">
                  Classifier usage
                </h3>
                <ResponsiveContainer width="100%" height="85%">
                  <BarChart data={classifierChartData}>
                    <XAxis dataKey="name" />
                    <YAxis />
                    <Tooltip />
                    <Bar dataKey="value" fill="#0ea5e9" radius={[6, 6, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>
          ) : (
            <p className="text-sm text-slate-500 dark:text-slate-400">Loading metrics...</p>
          )}
        </section>
      </main>

      <section className="mx-auto mb-8 w-full max-w-7xl px-4">
        <div className="rounded-xl border border-slate-200 bg-white p-4 shadow-soft dark:border-slate-800 dark:bg-slate-900">
          <div className="mb-3 flex flex-col gap-3 md:flex-row md:items-end md:justify-between">
            <div>
              <h2 className="text-lg font-semibold">Recent Activity</h2>
              <p className="text-sm text-slate-500 dark:text-slate-400">
                Auto-refresh every 5 seconds unless paused.
              </p>
            </div>
            <div className="flex flex-wrap gap-2">
              <select
                value={categoryFilter}
                onChange={(event) =>
                  setCategoryFilter(event.target.value as (typeof CATEGORY_OPTIONS)[number])
                }
                className="rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm dark:border-slate-700 dark:bg-slate-950"
              >
                {CATEGORY_OPTIONS.map((option) => (
                  <option key={option} value={option}>
                    Category: {option}
                  </option>
                ))}
              </select>
              <select
                value={classifierFilter}
                onChange={(event) =>
                  setClassifierFilter(event.target.value as (typeof CLASSIFIER_OPTIONS)[number])
                }
                className="rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm dark:border-slate-700 dark:bg-slate-950"
              >
                {CLASSIFIER_OPTIONS.map((option) => (
                  <option key={option} value={option}>
                    Classifier: {option}
                  </option>
                ))}
              </select>
              <select
                value={statusFilter}
                onChange={(event) => setStatusFilter(event.target.value as (typeof STATUS_OPTIONS)[number])}
                className="rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm dark:border-slate-700 dark:bg-slate-950"
              >
                {STATUS_OPTIONS.map((option) => (
                  <option key={option} value={option}>
                    Status: {option}
                  </option>
                ))}
              </select>
              <input
                value={searchQuery}
                onChange={(event) => setSearchQuery(event.target.value)}
                placeholder="Search text..."
                className="rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm dark:border-slate-700 dark:bg-slate-950"
              />
              <button
                type="button"
                onClick={() => void refreshLiveData()}
                className="rounded-lg bg-slate-800 px-3 py-2 text-sm font-semibold text-white hover:bg-slate-700 dark:bg-slate-200 dark:text-slate-900 dark:hover:bg-slate-300"
              >
                Refresh
              </button>
            </div>
          </div>

          <div className="overflow-x-auto">
            <table className="min-w-full text-left text-sm">
              <thead className="border-b border-slate-200 text-slate-500 dark:border-slate-700 dark:text-slate-400">
                <tr>
                  <th className="px-2 py-2">Time</th>
                  <th className="px-2 py-2">Category</th>
                  <th className="px-2 py-2">Confidence</th>
                  <th className="px-2 py-2">Latency</th>
                  <th className="px-2 py-2">Classifier</th>
                  <th className="px-2 py-2">Status</th>
                  <th className="px-2 py-2">Text</th>
                </tr>
              </thead>
              <tbody>
                {recentRows.map((row) => (
                  <tr
                    key={row.request_id}
                    onClick={() => setSelectedRow(row)}
                    className="cursor-pointer border-b border-slate-100 hover:bg-slate-50 dark:border-slate-800 dark:hover:bg-slate-800/40"
                  >
                    <td className="px-2 py-2">{formatTime(row.created_at)}</td>
                    <td className="px-2 py-2 uppercase">{row.category ?? "-"}</td>
                    <td className="px-2 py-2">
                      {typeof row.confidence === "number"
                        ? `${confidenceToPercent(row.confidence)}%`
                        : "-"}
                    </td>
                    <td className="px-2 py-2">{row.latency_ms} ms</td>
                    <td className="px-2 py-2">{row.classifier_name}</td>
                    <td className="px-2 py-2">
                      <span
                        className={`rounded-full px-2 py-1 text-xs font-semibold ${
                          row.ok
                            ? "bg-emerald-100 text-emerald-700 dark:bg-emerald-900/30 dark:text-emerald-300"
                            : "bg-rose-100 text-rose-700 dark:bg-rose-900/30 dark:text-rose-300"
                        }`}
                      >
                        {row.ok ? "ok" : "error"}
                      </span>
                    </td>
                    <td className="px-2 py-2">{truncate(row.text)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </section>

      {selectedRow && (
        <div className="fixed inset-0 z-30 flex">
          <button
            type="button"
            className="h-full w-full bg-black/40"
            onClick={() => setSelectedRow(null)}
            aria-label="Close details"
          />
          <aside className="h-full w-full max-w-md overflow-y-auto bg-white p-4 shadow-2xl dark:bg-slate-900">
            <div className="mb-3 flex items-center justify-between">
              <h3 className="text-lg font-semibold">Request Details</h3>
              <button
                type="button"
                onClick={() => setSelectedRow(null)}
                className="rounded-md border border-slate-300 px-2 py-1 text-sm dark:border-slate-700"
              >
                Close
              </button>
            </div>
            <dl className="space-y-3 text-sm">
              <Detail label="Request ID" value={selectedRow.request_id} />
              <Detail label="Created At" value={selectedRow.created_at} />
              <Detail label="Classifier" value={selectedRow.classifier_name} />
              <Detail label="Status" value={selectedRow.ok ? "ok" : "error"} />
              <Detail
                label="Confidence"
                value={
                  typeof selectedRow.confidence === "number"
                    ? `${confidenceToPercent(selectedRow.confidence)}%`
                    : "-"
                }
              />
              <Detail label="Latency" value={`${selectedRow.latency_ms} ms`} />
              <Detail label="Category" value={selectedRow.category ?? "-"} />
              <Detail label="Text" value={selectedRow.text} pre />
              <Detail label="Suggested Reply" value={selectedRow.suggested_reply ?? "-"} pre />
              {selectedRow.error_message && (
                <Detail label="Error" value={selectedRow.error_message} pre />
              )}
            </dl>
          </aside>
        </div>
      )}

      {toast && (
        <div className="fixed bottom-5 right-5 z-40 rounded-lg bg-slate-900 px-4 py-2 text-sm text-white shadow-lg dark:bg-slate-200 dark:text-slate-900">
          {toast}
        </div>
      )}
    </div>
  );
}

function StatCard({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-lg border border-slate-200 bg-slate-50 p-3 dark:border-slate-700 dark:bg-slate-950">
      <p className="text-xs font-semibold uppercase text-slate-500 dark:text-slate-400">{label}</p>
      <p className="mt-1 text-xl font-semibold">{value}</p>
    </div>
  );
}

function Detail({ label, value, pre = false }: { label: string; value: string; pre?: boolean }) {
  return (
    <div>
      <dt className="mb-1 text-xs font-semibold uppercase text-slate-500 dark:text-slate-400">
        {label}
      </dt>
      <dd
        className={`rounded-md border border-slate-200 bg-slate-50 px-2 py-2 dark:border-slate-700 dark:bg-slate-950 ${
          pre ? "whitespace-pre-wrap break-words" : ""
        }`}
      >
        {value}
      </dd>
    </div>
  );
}

export default App;

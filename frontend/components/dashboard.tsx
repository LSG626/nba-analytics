"use client";

import { ChangeEvent, useRef, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import {
  Activity,
  BarChart3,
  ChevronDown,
  CircleDot,
  Database,
  FileUp,
  Gauge,
  Menu,
  ShieldCheck,
  Users,
  X,
} from "lucide-react";
import { API_BASE, apiGet } from "@/lib/api";
import type { Overview, PredictionTeam } from "@/lib/api";
import { Button } from "@/components/ui/button";
import { AnalyticsPage } from "@/components/pages/analytics-page";
import { BacktestingPage } from "@/components/pages/backtesting-page";
import { GamePredictionsPage, PlayerPredictionsPage } from "@/components/pages/predictions-page";
import { ModelPerformancePage } from "@/components/pages/model-page";
import { OverviewPage } from "@/components/pages/overview-page";
import { TeamPage } from "@/components/pages/team-page";

type View =
  | "overview"
  | "analytics"
  | "games"
  | "players"
  | "teams"
  | "models"
  | "backtesting";

const navigation: Array<{
  id: View;
  label: string;
  icon: typeof Activity;
  group: string;
}> = [
  { id: "overview", label: "Overview", icon: Activity, group: "WORKSPACE" },
  { id: "analytics", label: "EDA & analytics", icon: BarChart3, group: "WORKSPACE" },
  { id: "games", label: "Game predictions", icon: CircleDot, group: "PREDICTION LAB" },
  { id: "players", label: "Player predictions", icon: Users, group: "PREDICTION LAB" },
  { id: "teams", label: "Team analytics", icon: ShieldCheck, group: "INSIGHT" },
  { id: "models", label: "Model performance", icon: Gauge, group: "INSIGHT" },
  { id: "backtesting", label: "Historical backtesting", icon: Database, group: "INSIGHT" },
];

export function Dashboard() {
  const [view, setView] = useState<View>("overview");
  const [mobileOpen, setMobileOpen] = useState(false);
  const [uploadMessage, setUploadMessage] = useState("");
  const fileInput = useRef<HTMLInputElement>(null);
  const health = useQuery({
    queryKey: ["api", "health"],
    queryFn: ({ signal }) => apiGet<{ status: string }>("/api/health", signal),
    refetchInterval: 30_000,
  });
  const overview = useQuery({
    queryKey: ["eda", "overview"],
    queryFn: ({ signal }) => apiGet<Overview>("/api/eda/overview", signal),
  });
  const teamList = useQuery({
    queryKey: ["prediction-team-list"],
    queryFn: ({ signal }) =>
      apiGet<{ available: boolean; teams: PredictionTeam[] }>(
        "/api/predictions/teams/list",
        signal,
      ),
  });
  const seasons = overview.data?.counts.season_labels ?? [];
  const teams =
    teamList.data?.teams.map((team) => ({
      value: team.team_id,
      label: team.team,
    })) ?? [];

  const active = navigation.find((item) => item.id === view);
  const activeGroup = active?.group;

  async function uploadDataset(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    if (!file) return;
    setUploadMessage("");
    const form = new FormData();
    form.append("file", file);
    try {
      const response = await fetch(`${API_BASE}/api/datasets/upload`, {
        method: "POST",
        body: form,
      });
      const payload: unknown = await response.json().catch(() => null);
      if (!response.ok) {
        const detail =
          payload && typeof payload === "object" && "detail" in payload
            ? String(payload.detail)
            : `Upload failed (${response.status}).`;
        throw new Error(detail);
      }
      setUploadMessage(`${file.name} uploaded to raw data. Run ingestion to process it.`);
      await overview.refetch();
    } catch (error) {
      setUploadMessage(error instanceof Error ? error.message : "Upload failed.");
    } finally {
      event.target.value = "";
    }
  }

  function navigate(next: View) {
    setView(next);
    setMobileOpen(false);
  }

  return (
    <div className="app-shell">
      {mobileOpen && <button className="sidebar-backdrop" onClick={() => setMobileOpen(false)} aria-label="Close navigation" />}
      <aside className={`sidebar${mobileOpen ? " sidebar-open" : ""}`}>
        <div className="brand">
          <div className="brand-mark"><span /></div>
          <div><strong>COURTSIDE<span>AI</span></strong><small>NBA INTELLIGENCE</small></div>
          <button className="mobile-close" onClick={() => setMobileOpen(false)} aria-label="Close menu"><X size={18} /></button>
        </div>
        <div className="workspace-switch">
          <span className="workspace-icon"><Activity size={16} /></span>
          <span><strong>Local workspace</strong><small>Supplied datasets</small></span>
          <ChevronDown size={14} />
        </div>
        <nav className="side-nav" aria-label="Primary navigation">
          {(["WORKSPACE", "PREDICTION LAB", "INSIGHT"] as const).map((group) => (
            <div className="nav-group" key={group}>
              <p>{group}</p>
              {navigation.filter((item) => item.group === group).map((item) => (
                <button
                  type="button"
                  key={item.id}
                  className={`nav-item${view === item.id ? " nav-active" : ""}`}
                  onClick={() => navigate(item.id)}
                  aria-current={view === item.id ? "page" : undefined}
                >
                  <item.icon size={17} strokeWidth={1.8} />
                  <span>{item.label}</span>
                  {view === item.id && <i />}
                </button>
              ))}
            </div>
          ))}
        </nav>
        <div className="sidebar-bottom">
          <div className="sidebar-status">
            <span className={`status-pip${health.isError ? " status-offline" : ""}`} />
            <span>{health.isError ? "API unavailable" : "API connected"}</span>
            <small>{health.isPending ? "…" : health.isError ? "offline" : "local"}</small>
          </div>
          <p>COURTSIDE AI <span>·</span> LOCAL ANALYTICS</p>
        </div>
      </aside>

      <div className="main-shell">
        <header className="topbar">
          <div className="topbar-left">
            <button className="mobile-menu" onClick={() => setMobileOpen(true)} aria-label="Open navigation"><Menu size={19} /></button>
            <div className="breadcrumbs"><span>COURTSIDE</span><i>/</i><strong>{active?.label}</strong></div>
          </div>
          <div className="topbar-right">
            <div className="api-pill"><span className={`status-pip${health.isError ? " status-offline" : ""}`} />{health.isError ? "Backend offline" : "Live API"}</div>
            <Button variant="secondary" className="upload-button" onClick={() => fileInput.current?.click()} type="button">
              <FileUp size={15} /> <span>Upload data</span>
            </Button>
            <input ref={fileInput} className="sr-only" type="file" accept=".csv,text/csv" onChange={uploadDataset} />
          </div>
        </header>
        {uploadMessage && (
          <div className={`upload-message${uploadMessage.toLowerCase().includes("failed") ? " upload-error" : ""}`} role="status">
            {uploadMessage}<button type="button" onClick={() => setUploadMessage("")} aria-label="Dismiss">×</button>
          </div>
        )}
        {health.isError && view !== "overview" && (
          <div className="offline-banner" role="status">
            Backend unavailable. Start FastAPI to load current analyses and predictions.
          </div>
        )}
        <main className="content-area">
          {view === "overview" && <OverviewPage />}
          {view === "analytics" && <AnalyticsPage seasons={seasons} teams={teams} />}
          {view === "games" && <GamePredictionsPage seasons={seasons} teams={teams} />}
          {view === "players" && <PlayerPredictionsPage teams={teams} />}
          {view === "teams" && <TeamPage seasons={seasons} teams={teams} />}
          {view === "models" && <ModelPerformancePage />}
          {view === "backtesting" && <BacktestingPage seasons={seasons} teams={teams} />}
        </main>
        <footer className="page-footer">
          <span>COURTSIDE AI</span><i /> Dashboard values are returned by the connected local API.
        </footer>
      </div>
      <span className="sr-only">Active section: {activeGroup}</span>
    </div>
  );
}

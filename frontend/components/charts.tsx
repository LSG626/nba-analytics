"use client";

import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { CorrelationTable } from "@/lib/api";
import { EmptyState, Panel, PanelHeading } from "@/components/ui";

export interface ChartPoint {
  [key: string]: string | number | null;
}

const chartColors = ["#b84c29", "#173e35", "#6f8f83", "#d4a84e"];

const tooltipStyle = {
  border: "1px solid #e6e9e6",
  borderRadius: 10,
  background: "#ffffff",
  boxShadow: "0 12px 30px rgba(27, 38, 34, .08)",
  fontSize: 12,
};

function ChartFrame({
  title,
  detail,
  children,
  className,
}: {
  title: string;
  detail?: string;
  children: React.ReactNode;
  className?: string;
}) {
  return (
    <Panel className={className}>
      <PanelHeading title={title} detail={detail} />
      {children}
    </Panel>
  );
}

export function TrendChart({
  title,
  detail,
  data,
  xKey,
  yKey,
  yLabel,
  color = chartColors[0],
}: {
  title: string;
  detail?: string;
  data: ChartPoint[];
  xKey: string;
  yKey: string;
  yLabel?: string;
  color?: string;
}) {
  if (!data.length) {
    return (
      <ChartFrame title={title} detail={detail}>
        <EmptyState title="No chart data available" />
      </ChartFrame>
    );
  }
  return (
    <ChartFrame title={title} detail={detail}>
      <div className="chart-area">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data} margin={{ top: 14, right: 12, bottom: 4, left: 0 }}>
            <CartesianGrid stroke="#edf0ed" vertical={false} />
            <XAxis
              dataKey={xKey}
              tickLine={false}
              axisLine={false}
              minTickGap={24}
              tick={{ fill: "#858d87", fontSize: 11 }}
            />
            <YAxis
              tickLine={false}
              axisLine={false}
              width={48}
              tick={{ fill: "#858d87", fontSize: 11 }}
              label={
                yLabel
                  ? { value: yLabel, angle: -90, position: "insideLeft", fill: "#858d87", fontSize: 10 }
                  : undefined
              }
            />
            <Tooltip contentStyle={tooltipStyle} />
            <Line
              type="monotone"
              dataKey={yKey}
              stroke={color}
              strokeWidth={2.2}
              dot={false}
              activeDot={{ r: 4, fill: color, stroke: "#fff", strokeWidth: 2 }}
              connectNulls
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </ChartFrame>
  );
}

export function MultiTrendChart({
  title,
  detail,
  data,
  xKey,
  series,
  yLabel,
}: {
  title: string;
  detail?: string;
  data: ChartPoint[];
  xKey: string;
  series: Array<{ key: string; label: string; color: string }>;
  yLabel?: string;
}) {
  if (!data.length || !series.length) {
    return (
      <ChartFrame title={title} detail={detail}>
        <EmptyState title="No chart data available" />
      </ChartFrame>
    );
  }
  return (
    <ChartFrame title={title} detail={detail}>
      <div className="chart-area">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data} margin={{ top: 14, right: 12, bottom: 4, left: 0 }}>
            <CartesianGrid stroke="#edf0ed" vertical={false} />
            <XAxis dataKey={xKey} tickLine={false} axisLine={false} minTickGap={24} tick={{ fill: "#858d87", fontSize: 11 }} />
            <YAxis
              tickLine={false}
              axisLine={false}
              width={48}
              tick={{ fill: "#858d87", fontSize: 11 }}
              label={yLabel ? { value: yLabel, angle: -90, position: "insideLeft", fill: "#858d87", fontSize: 10 } : undefined}
            />
            <Tooltip contentStyle={tooltipStyle} />
            {series.map((item) => (
              <Line
                key={item.key}
                type="monotone"
                dataKey={item.key}
                name={item.label}
                stroke={item.color}
                strokeWidth={2}
                dot={false}
                activeDot={{ r: 4, stroke: "#fff", strokeWidth: 2 }}
                connectNulls
              />
            ))}
          </LineChart>
        </ResponsiveContainer>
      </div>
      <div className="chart-legend">
        {series.map((item) => <span key={item.key}><i style={{ backgroundColor: item.color }} />{item.label}</span>)}
      </div>
    </ChartFrame>
  );
}

export function BarChartPanel({
  title,
  detail,
  data,
  xKey,
  yKey,
  color = chartColors[0],
  horizontal = false,
}: {
  title: string;
  detail?: string;
  data: ChartPoint[];
  xKey: string;
  yKey: string;
  color?: string;
  horizontal?: boolean;
}) {
  if (!data.length) {
    return (
      <ChartFrame title={title} detail={detail}>
        <EmptyState title="No chart data available" />
      </ChartFrame>
    );
  }
  return (
    <ChartFrame title={title} detail={detail}>
      <div className={`chart-area${horizontal ? " chart-area-tall" : ""}`}>
        <ResponsiveContainer width="100%" height="100%">
          <BarChart
            data={data}
            layout={horizontal ? "vertical" : "horizontal"}
            margin={{ top: 12, right: 12, bottom: 8, left: horizontal ? 30 : 0 }}
          >
            <CartesianGrid stroke="#edf0ed" vertical={!horizontal} horizontal={horizontal} />
            {horizontal ? (
              <>
                <XAxis type="number" tickLine={false} axisLine={false} tick={{ fill: "#858d87", fontSize: 11 }} />
                <YAxis dataKey={xKey} type="category" width={112} tickLine={false} axisLine={false} tick={{ fill: "#68716b", fontSize: 11 }} />
              </>
            ) : (
              <>
                <XAxis dataKey={xKey} tickLine={false} axisLine={false} minTickGap={18} tick={{ fill: "#858d87", fontSize: 11 }} />
                <YAxis tickLine={false} axisLine={false} tick={{ fill: "#858d87", fontSize: 11 }} />
              </>
            )}
            <Tooltip contentStyle={tooltipStyle} />
            <Bar dataKey={yKey} fill={color} radius={horizontal ? [0, 4, 4, 0] : [4, 4, 0, 0]} maxBarSize={28}>
              {data.map((point, index) => (
                <Cell key={String(point[xKey]) + index} fill={chartColors[index % chartColors.length]} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
    </ChartFrame>
  );
}

export function MultiBarChart({
  title,
  detail,
  data,
  xKey,
  series,
}: {
  title: string;
  detail?: string;
  data: ChartPoint[];
  xKey: string;
  series: Array<{ key: string; label: string; color: string }>;
}) {
  if (!data.length || !series.length) {
    return (
      <ChartFrame title={title} detail={detail}>
        <EmptyState title="No chart data available" />
      </ChartFrame>
    );
  }
  return (
    <ChartFrame title={title} detail={detail}>
      <div className="chart-area">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={data} margin={{ top: 12, right: 12, bottom: 8, left: 0 }}>
            <CartesianGrid stroke="#edf0ed" vertical={false} />
            <XAxis dataKey={xKey} tickLine={false} axisLine={false} tick={{ fill: "#858d87", fontSize: 11 }} />
            <YAxis tickLine={false} axisLine={false} tick={{ fill: "#858d87", fontSize: 11 }} />
            <Tooltip contentStyle={tooltipStyle} />
            {series.map((item) => (
              <Bar key={item.key} dataKey={item.key} name={item.label} fill={item.color} radius={[4, 4, 0, 0]} maxBarSize={24} />
            ))}
          </BarChart>
        </ResponsiveContainer>
      </div>
      <div className="chart-legend">
        {series.map((item) => <span key={item.key}><i style={{ backgroundColor: item.color }} />{item.label}</span>)}
      </div>
    </ChartFrame>
  );
}

export function ScatterPanel({
  title,
  detail,
  data,
  xLabel,
  yLabel,
}: {
  title: string;
  detail?: string;
  data: Array<{ x: number; y: number }>;
  xLabel: string;
  yLabel: string;
}) {
  if (!data.length) {
    return (
      <ChartFrame title={title} detail={detail}>
        <EmptyState title="Relationship not available" />
      </ChartFrame>
    );
  }
  return (
    <ChartFrame title={title} detail={detail}>
      <div className="chart-area">
        <ResponsiveContainer width="100%" height="100%">
          <ScatterChart margin={{ top: 12, right: 18, bottom: 12, left: 0 }}>
            <CartesianGrid stroke="#edf0ed" />
            <XAxis
              type="number"
              dataKey="x"
              name={xLabel}
              tickLine={false}
              axisLine={false}
              tick={{ fill: "#858d87", fontSize: 11 }}
              label={{ value: xLabel, position: "insideBottom", offset: -8, fill: "#858d87", fontSize: 10 }}
            />
            <YAxis
              type="number"
              dataKey="y"
              name={yLabel}
              tickLine={false}
              axisLine={false}
              tick={{ fill: "#858d87", fontSize: 11 }}
              label={{ value: yLabel, angle: -90, position: "insideLeft", fill: "#858d87", fontSize: 10 }}
            />
            <Tooltip cursor={{ strokeDasharray: "3 3" }} contentStyle={tooltipStyle} />
            <Scatter data={data} fill={chartColors[0]} fillOpacity={0.45} />
          </ScatterChart>
        </ResponsiveContainer>
      </div>
    </ChartFrame>
  );
}

export function CorrelationHeatmap({
  title,
  table,
}: {
  title: string;
  table?: CorrelationTable;
}) {
  if (!table || !table.columns.length) {
    return (
      <Panel>
        <PanelHeading title={title} />
        <EmptyState title="Correlation matrix unavailable" />
      </Panel>
    );
  }
  const colorFor = (value: number | null) => {
    if (value === null || !Number.isFinite(value)) return "#f0f1ef";
    if (value < 0) {
      const strength = Math.min(1, Math.abs(value));
      return `rgba(184, 76, 41, ${0.08 + strength * 0.72})`;
    }
    return `rgba(23, 62, 53, ${0.08 + Math.min(1, value) * 0.72})`;
  };
  return (
    <Panel>
      <PanelHeading
        title={title}
        detail={`${table.source} · ${table.observations.toLocaleString()} observations`}
      />
      <div className="heatmap-scroll">
        <div className="heatmap" style={{ gridTemplateColumns: `128px repeat(${table.columns.length}, minmax(42px, 1fr))` }}>
          <span />
          {table.columns.map((column) => (
            <span className="heatmap-label heatmap-column" key={`col-${column}`} title={column}>
              {column.replaceAll("_", " ")}
            </span>
          ))}
          {table.columns.map((row, rowIndex) => (
            <div className="heatmap-row" key={`row-${row}`}>
              <span className="heatmap-label" title={row}>{row.replaceAll("_", " ")}</span>
              {table.columns.map((column, columnIndex) => {
                const value = table.matrix[rowIndex]?.[columnIndex] ?? null;
                return (
                  <span
                    className="heat-cell"
                    key={`${row}-${column}`}
                    style={{ backgroundColor: colorFor(value) }}
                    title={`${row} × ${column}: ${value === null ? "unavailable" : value.toFixed(3)}`}
                  >
                    {value === null ? "—" : value.toFixed(1)}
                  </span>
                );
              })}
            </div>
          ))}
        </div>
      </div>
      <div className="heatmap-legend"><span>−1 negative</span><span>0 neutral</span><span>+1 positive</span></div>
    </Panel>
  );
}

export function MetricBars({
  rows,
  labelKey,
  valueKey,
  unit = "",
}: {
  rows: ChartPoint[];
  labelKey: string;
  valueKey: string;
  unit?: string;
}) {
  if (!rows.length) return <EmptyState title="No measurable data" />;
  return (
    <div className="metric-bars">
      {rows.map((row) => {
        const value = Number(row[valueKey]);
        const maximum = Math.max(...rows.map((entry) => Number(entry[valueKey]) || 0), 1);
        return (
          <div className="metric-bar-row" key={String(row[labelKey])}>
            <span className="metric-bar-name">{String(row[labelKey])}</span>
            <span className="metric-bar-track">
              <span style={{ width: `${Math.max(3, (value / maximum) * 100)}%` }} />
            </span>
            <strong>{Number.isFinite(value) ? `${value.toFixed(1)}${unit}` : "—"}</strong>
          </div>
        );
      })}
    </div>
  );
}

export function ZeroReference() {
  return <ReferenceLine y={0} stroke="#bfc6c0" />;
}

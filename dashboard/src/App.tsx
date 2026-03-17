import { useState, useEffect, useRef, useMemo } from 'react';
import { 
  Activity, ShieldAlert, AlertTriangle, ShieldCheck, 
  BarChart3, PieChart, ActivitySquare, ServerCrash, LayoutDashboard, Target
} from 'lucide-react';
import { 
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip as RechartsTooltip, 
  ResponsiveContainer, PieChart as RechartsPie, Pie, Cell, Legend
} from 'recharts';
import ThreatIntelligence from './components/ThreatIntelligence';

// --- Types ---
interface Transaction {
  transaction_id: string;
  user_id: string;
  amount: number;
  merchant_category: string;
  location_city: string;
  risk_score: number;
  should_display_alert: boolean;
  should_block: boolean;
  fraud_type_predicted: string;
  timestamp: string;
  shap_top3?: any[];
  shap?: any[];
  explanation?: any;
}

interface Metrics {
  total_scored: number;
  tp: number; fp: number; tn: number; fn: number;
  accuracy: number; precision: number; recall: number; f1: number; fpr: number;
}

// --- Constants ---
const API_BASE = 'http://localhost:8000';
const WS_URL = 'ws://localhost:8000/stream?tps=20&rows=10000';

const METRIC_COLORS = {
  legit: '#238636',   // green
  fraud: '#da3633',   // red
  warning: '#d29922', // yellow/orange
  border: '#30363D',  // gray
  card: '#161B22'     // dark gray
};

const PIE_COLORS = ['#da3633', '#d29922', '#8957e5', '#2f81f7', '#db6d28', '#f85149'];

// --- Helper Functions ---
const formatCurrency = (val: number) => new Intl.NumberFormat('en-IN', { style: 'currency', currency: 'INR', maximumFractionDigits: 0 }).format(val);

export default function App() {
  // --- State ---
  const [isConnected, setIsConnected] = useState(false);
  const [transactions, setTransactions] = useState<Transaction[]>([]);
  const [alerts, setAlerts] = useState<Transaction[]>([]);
  const [metrics, setMetrics] = useState<Metrics | null>(null);
  const [timeSeriesData, setTimeSeriesData] = useState<any[]>([]);
  const [isHoveringFeed, setIsHoveringFeed] = useState(false);
  const [activeTab, setActiveTab] = useState<'dashboard' | 'threat_intel'>('dashboard');

  // --- Refs ---
  const wsRef = useRef<WebSocket | null>(null);
  const txBuffer = useRef<Transaction[]>([]);
  const seriesBuffer = useRef<any[]>([]);
  const isHoveringRef = useRef(false);

  // --- WebSocket Connection (Panel 1 & 4) ---
  useEffect(() => {
    const connectWs = () => {
      const ws = new WebSocket(WS_URL);
      
      ws.onopen = () => setIsConnected(true);
      ws.onclose = () => {
        setIsConnected(false);
        setTimeout(connectWs, 3000); // Retry 3s
      };
      ws.onerror = () => setIsConnected(false);
      
      ws.onmessage = (event) => {
        try {
          const data: Transaction = JSON.parse(event.data);
          
          // Update transaction feed
          txBuffer.current = [data, ...txBuffer.current.filter(t => t.transaction_id !== data.transaction_id)].slice(0, 500);
          if (!isHoveringRef.current) {
            setTransactions([...txBuffer.current]);
          }

          // Update timeseries buffer (group by second for simple chart)
          const now = new Date();
          const timeLabel = now.toLocaleTimeString([], { hour12: false, minute: '2-digit', second: '2-digit' });
          
          const lastPoint = seriesBuffer.current[seriesBuffer.current.length - 1];
          if (lastPoint && lastPoint.time === timeLabel) {
            seriesBuffer.current[seriesBuffer.current.length - 1] = {
              ...lastPoint,
              total: lastPoint.total + 1,
              fraud: lastPoint.fraud + (data.should_display_alert ? 1 : 0),
              legit: lastPoint.legit + (!data.should_display_alert ? 1 : 0),
            };
          } else {
            seriesBuffer.current.push({
              time: timeLabel,
              total: 1,
              fraud: data.should_display_alert ? 1 : 0,
              legit: !data.should_display_alert ? 1 : 0,
            });
            if (seriesBuffer.current.length > 60) seriesBuffer.current.shift(); // Keep 60s
          }
          setTimeSeriesData([...seriesBuffer.current]);

        } catch (e) {
          console.error("WS Parse Error", e);
        }
      };
      wsRef.current = ws;
    };

    connectWs();
    return () => wsRef.current?.close();
  }, []);

  // --- REST Polling (Panels 2, 3, 5, 6) ---
  useEffect(() => {
    const fetchApiData = async () => {
      try {
        // Fetch Metrics
        const mRes = await fetch(`${API_BASE}/metrics`);
        if (mRes.ok) setMetrics(await mRes.json());

        // Fetch Alerts
        const aRes = await fetch(`${API_BASE}/alerts`);
        if (aRes.ok) setAlerts(await aRes.json());
      } catch (e) {
        // Graceful fallback: connection banner handles offline state
      }
    };

    fetchApiData();
    const interval = setInterval(fetchApiData, 2000);
    return () => clearInterval(interval);
  }, []);

  // --- Derived Data ---
  const fraudTypeStats = useMemo(() => {
    const counts: Record<string, number> = {};
    alerts.forEach(a => {
      const t = a.fraud_type_predicted || 'anomaly';
      counts[t] = (counts[t] || 0) + 1;
    });
    return Object.entries(counts)
      .map(([name, value]) => ({ name, value }))
      .sort((a,b) => b.value - a.value);
  }, [alerts]);


  // --- Render Helpers ---
  const getRiskBadge = (score: number) => {
    if (score >= 0.90) return <span className="px-2 py-0.5 rounded text-xs font-bold bg-[#da3633]/20 text-[#da3633] border border-[#da3633]/30">CRITICAL {(score*100).toFixed(1)}%</span>;
    if (score >= 0.40) return <span className="px-2 py-0.5 rounded text-xs font-bold bg-[#d29922]/20 text-[#d29922] border border-[#d29922]/30">HIGH {(score*100).toFixed(1)}%</span>;
    return <span className="px-2 py-0.5 rounded text-xs font-medium bg-[#238636]/10 text-gray-400">LOW</span>;
  };

  return (
    <div className="flex h-screen bg-[#0D1117] text-gray-300 font-sans overflow-hidden">
      
      {/* --- Sidebar Navigation --- */}
      <div className="w-16 md:w-64 bg-[#161B22] border-r border-[#30363D] flex flex-col shrink-0 transition-all z-20">
        <div className="h-16 flex items-center justify-center md:justify-start md:px-6 border-b border-[#30363D] shrink-0">
          <ShieldCheck className="text-indigo-400" size={24} />
          <span className="ml-3 font-bold text-white hidden md:block">FraudShield</span>
        </div>
        <div className="flex-1 py-4 flex flex-col gap-2 px-3">
          <button 
            onClick={() => setActiveTab('dashboard')}
            className={`flex items-center gap-3 px-3 py-2.5 rounded-lg transition-colors ${activeTab === 'dashboard' ? 'bg-indigo-500/20 text-indigo-400 font-semibold' : 'text-gray-400 hover:bg-[#30363D]/50 hover:text-gray-200'}`}
          >
            <LayoutDashboard size={20} className="shrink-0" />
            <span className="hidden md:block">Dashboard</span>
          </button>
          <button 
            onClick={() => setActiveTab('threat_intel')}
            className={`flex items-center gap-3 px-3 py-2.5 rounded-lg transition-colors ${activeTab === 'threat_intel' ? 'bg-emerald-500/20 text-emerald-400 font-semibold' : 'text-gray-400 hover:bg-[#30363D]/50 hover:text-gray-200'}`}
          >
            <Target size={20} className="shrink-0" />
            <span className="hidden md:block text-left">Threat Intelligence</span>
          </button>
        </div>
      </div>

      <div className="flex-1 flex flex-col min-w-0 relative pt-6">
        {/* --- Connection Banner --- */}
        {!isConnected && (
          <div className="fixed top-0 left-0 right-0 bg-[#da3633] text-white text-center py-2 z-50 flex items-center justify-center gap-2 text-sm font-medium shadow-lg animate-pulse">
            <ServerCrash size={16} /> Connecting to AI Backend ({WS_URL})...
          </div>
        )}

        {activeTab === 'dashboard' ? (
          <div className="flex-1 overflow-hidden flex flex-col px-6 pb-6 pt-0">
            {/* --- Header & KPIs (Panel 2) --- */}
            <div className={`shrink-0 mb-4 transition-opacity duration-500 ${!isConnected ? 'mt-8 opacity-50' : 'opacity-100'}`}>
        <div className="flex items-center gap-3 mb-4">
          <div className="w-10 h-10 rounded-lg bg-indigo-500/20 flex items-center justify-center border border-indigo-500/30 shadow-[0_0_15px_rgba(99,102,241,0.2)]">
            <ShieldCheck className="text-indigo-400" size={24} />
          </div>
          <div>
            <h1 className="text-2xl font-bold text-white tracking-tight">UPI FraudShield</h1>
            <p className="text-sm text-gray-400 flex items-center gap-2">
              <span className="w-2 h-2 rounded-full bg-green-500 animate-soft-pulse"></span>
              Live Transaction Streaming (M6)
            </p>
          </div>
        </div>

        <div className="grid grid-cols-5 gap-4 h-24 shrink-0">
          <div className="bg-[#161B22] border border-[#30363D] rounded-xl p-4 shadow-sm flex flex-col justify-center">
            <div className="text-gray-400 text-xs font-medium uppercase tracking-wider mb-1">Total Scoped</div>
            <div className="text-3xl font-light text-white">{metrics?.total_scored?.toLocaleString() || 0}</div>
          </div>
          <div className="bg-[#161B22] border border-[#30363D] rounded-xl p-4 shadow-sm relative overflow-hidden flex flex-col justify-center">
            <div className="text-gray-400 text-xs font-medium uppercase tracking-wider mb-1">Alerts Generated</div>
            <div className="text-3xl font-light text-[#d29922]">{alerts.length}</div>
          </div>
          <div className="bg-[#161B22] border border-[#30363D] rounded-xl p-4 shadow-sm relative overflow-hidden flex flex-col justify-center">
             <div className="absolute top-0 right-0 w-16 h-16 bg-[#da3633]/10 rounded-bl-[100px] pointer-events-none"></div>
            <div className="text-gray-400 text-xs font-medium uppercase tracking-wider mb-1">Tx Blocked (&gt;90%)</div>
            <div className="text-3xl font-bold text-[#da3633]">{alerts.filter(a => a.should_block).length}</div>
          </div>
          <div className="bg-[#161B22] border border-[#30363D] rounded-xl p-4 shadow-sm relative overflow-hidden flex flex-col justify-center">
            <div className="text-gray-400 text-xs font-medium uppercase tracking-wider mb-1 flex items-center justify-between">
              False Pos Rate
              {(metrics?.fpr || 0) > 0.02 ? (
                <AlertTriangle size={14} className="text-[#da3633]" />
              ) : (
                <ShieldCheck size={14} className="text-[#238636]" />
              )}
            </div>
            <div className={`text-3xl font-light ${!metrics ? 'text-white' : metrics.fpr > 0.02 ? 'text-[#da3633]' : 'text-[#238636]'}`}>
              {metrics ? (metrics.fpr * 100).toFixed(2) : 0}%
            </div>
          </div>
          <div className="bg-[#161B22] border border-[#30363D] rounded-xl p-4 shadow-sm flex flex-col justify-center">
            <div className="text-gray-400 text-xs font-medium uppercase tracking-wider mb-1">Model F1-Score</div>
            <div className="text-3xl font-light text-indigo-400">{metrics?.f1.toFixed(4) || "0.0000"}</div>
          </div>
        </div>
      </div>

      {/* --- Main 2-Column Grid --- */}
      <div className={`flex-1 min-h-0 grid grid-cols-12 gap-6 ${!isConnected ? 'opacity-50 pointer-events-none' : ''}`}>
        
        {/* LEFT COLUMN: Feed & Alerts */}
        <div className="col-span-7 flex flex-col gap-2 h-full min-h-0">
          
          {/* Panel 1: Live Transaction Feed */}
          <div className="bg-[#161B22] border border-[#30363D] rounded-xl shadow-lg flex flex-col flex-1 min-h-0">
            <div className="p-4 border-b border-[#30363D] flex justify-between items-center bg-[#161B22] rounded-t-xl z-10 shrink-0">
              <h2 className="text-lg font-semibold text-white flex items-center gap-2">
                <Activity size={18} className="text-blue-400" />
                Live Network Feed
              </h2>
              <span className="text-xs text-gray-400 flex items-center gap-1">
                {isHoveringFeed ? '⏸ Paused' : <><span className="w-1.5 h-1.5 rounded-full bg-green-500 animate-pulse"></span> Auto-scrolling</>}
              </span>
            </div>
            <div 
              className="overflow-y-auto flex-1 min-h-0 p-2"
              onMouseEnter={() => { setIsHoveringFeed(true); isHoveringRef.current = true; }}
              onMouseLeave={() => { setIsHoveringFeed(false); isHoveringRef.current = false; setTransactions([...txBuffer.current]); }}
            >
              <table className="w-full text-sm text-left">
                <thead className="text-xs text-gray-500 uppercase sticky top-0 bg-[#161B22]">
                  <tr>
                    <th className="px-3 py-2">ID</th>
                    <th className="px-3 py-2">Amount</th>
                    <th className="px-3 py-2">Category</th>
                    <th className="px-3 py-2">City</th>
                    <th className="px-3 py-2 text-right">Risk Score</th>
                  </tr>
                </thead>
                <tbody>
                  {transactions.slice(0, 100).map((t, i) => (
                    <tr key={String(t.transaction_id) + i} className="border-b border-[#30363D]/50 hover:bg-[#30363D]/30 transition-colors animate-slide-in">
                      <td className="px-3 py-2.5 font-mono text-xs text-gray-400 truncate max-w-[80px]">{String(t.transaction_id || 'N/A').substring(0,8)}</td>
                      <td className="px-3 py-2.5 font-medium text-gray-200">{formatCurrency(t.amount)}</td>
                      <td className="px-3 py-2.5 text-gray-400 capitalize">{t.merchant_category.replace('_', ' ')}</td>
                      <td className="px-3 py-2.5 text-gray-400">{t.location_city}</td>
                      <td className="px-3 py-2.5 text-right">{getRiskBadge(t.risk_score)}</td>
                    </tr>
                  ))}
                  {transactions.length === 0 && (
                    <tr><td colSpan={5} className="text-center py-10 text-gray-500">Waiting for transactions...</td></tr>
                  )}
                </tbody>
              </table>
            </div>
          </div>

          {/* Panel 3: Fraud Alert Cards */}
          <div className="bg-[#161B22] border border-[#30363D] rounded-xl shadow-lg flex flex-col flex-1 min-h-0">
            <div className="p-4 border-b border-[#30363D] flex justify-between items-center bg-[#161B22] rounded-t-xl z-10 shrink-0">
              <h2 className="text-lg font-semibold text-white flex items-center gap-2">
                <ShieldAlert size={18} className="text-[#d29922]" />
                Actionable Alerts Queue
              </h2>
              <span className="bg-[#d29922]/20 text-[#d29922] text-xs px-2 py-1 rounded-full font-bold border border-[#d29922]/30">
                {alerts.length} Flagged
              </span>
            </div>
            <div className="overflow-y-auto flex-1 min-h-0 p-4 flex flex-col gap-4">
              {alerts.length === 0 ? (
                <div className="text-center py-10 text-gray-500">No alerts generated yet.</div>
              ) : (
                alerts.map((a, i) => (
                  <div key={a.transaction_id + i} className="border border-[#30363D] rounded-lg p-4 bg-[#0D1117] relative overflow-hidden hover:border-gray-600 transition-colors shrink-0">
                    {a.should_block && <div className="absolute left-0 top-0 bottom-0 w-1 bg-[#da3633]"></div>}
                    {!a.should_block && <div className="absolute left-0 top-0 bottom-0 w-1 bg-[#d29922]"></div>}
                    
                    <div className="flex justify-between items-start mb-2 pl-2">
                      <div>
                        <div className="flex items-center gap-2 mb-1">
                          {getRiskBadge(a.risk_score)}
                          <span className="text-xs font-mono text-gray-500">{a.transaction_id}</span>
                        </div>
                        <h3 className="text-white font-medium text-lg">{formatCurrency(a.amount)} at <span className="capitalize">{a.merchant_category.replace('_', ' ')}</span></h3>
                      </div>
                      <div className="text-right">
                        <div className="text-xs text-gray-400 uppercase tracking-wide">Detected Pattern</div>
                        <div className="text-sm text-gray-200 font-medium capitalize mt-0.5 bg-gray-800 px-2 py-1 rounded inline-block">{a.fraud_type_predicted.replace('_', ' ')}</div>
                      </div>
                    </div>
                    
                    {a.explanation?.short_summary && (
                      <p className="text-sm text-gray-300 mt-3 pl-2 border-l-2 border-gray-700 ml-1 py-1 italic">
                        "{a.explanation.short_summary}"
                      </p>
                    )}

                    {/* SHAP Mini Chart inside card */}
                    {a.shap && a.shap.length > 0 && (
                      <div className="mt-4 bg-[#161B22] rounded border border-[#30363D] p-3 pl-4">
                        <div className="text-xs text-gray-500 uppercase font-semibold mb-2">AI Explainability (SHAP Top Factors)</div>
                        <div className="flex flex-col gap-2">
                          {a.shap.map((s: any, idx: number) => (
                            <div key={idx} className="flex items-center text-xs">
                              <div className="w-1/3 text-gray-400 truncate pr-2">{s.name}</div>
                              <div className="w-2/3 flex items-center gap-2">
                                <div className="bg-gray-800 h-2 rounded-full overflow-hidden flex-1 max-w-[150px]">
                                  <div 
                                    className={`h-full ${s.direction === 'increases_risk' ? 'bg-[#da3633]' : 'bg-[#2f81f7]'}`} 
                                    style={{ width: `${s.bar_width_pct}%` }}
                                  ></div>
                                </div>
                                <span className="font-mono text-gray-500">{s.value > 0 ? '+' : ''}{s.value.toFixed(2)}</span>
                              </div>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                ))
              )}
            </div>
          </div>
        </div>

        {/* RIGHT COLUMN: Charts & Metrics */}
        <div className="col-span-5 flex flex-col gap-2 h-full min-h-0">
          
          {/* Panel 4: Timeline */}
          <div className="bg-[#161B22] border border-[#30363D] rounded-xl p-4 shadow-lg flex-1 min-h-0 flex flex-col">
            <h2 className="text-base font-semibold text-white mb-4 flex items-center gap-2 shrink-0">
              <ActivitySquare size={16} className="text-gray-400" />
              Real-time Volume (60s)
            </h2>
            <div className="flex-1 min-h-0">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={timeSeriesData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#30363D" vertical={false} />
                  <XAxis dataKey="time" stroke="#8b949e" fontSize={11} tickMargin={10} minTickGap={20} />
                  <YAxis stroke="#8b949e" fontSize={11} />
                  <RechartsTooltip 
                    contentStyle={{ backgroundColor: '#0D1117', borderColor: '#30363D', borderRadius: '8px' }}
                    itemStyle={{ fontSize: '12px' }}
                    labelStyle={{ color: '#8b949e', fontSize: '11px', marginBottom: '4px' }}
                  />
                  <Line type="monotone" dataKey="legit" stroke={METRIC_COLORS.legit} strokeWidth={2} dot={false} isAnimationActive={false} />
                  <Line type="monotone" dataKey="fraud" stroke={METRIC_COLORS.fraud} strokeWidth={2} dot={false} isAnimationActive={false} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Panel 5: Fraud Type Breakdown */}
          <div className="bg-[#161B22] border border-[#30363D] rounded-xl p-4 shadow-lg flex-1 min-h-0 flex flex-col overflow-hidden">
            <h2 className="text-base font-semibold text-white mb-2 flex items-center gap-2 shrink-0">
              <PieChart size={16} className="text-gray-400" />
              Fraud Typography
            </h2>
            {alerts.length === 0 ? (
               <div className="flex-1 min-h-0 flex items-center justify-center text-gray-500 text-sm">Waiting for triggers...</div>
            ) : (
              <div className="flex-[2] min-h-0 relative flex items-center justify-center">
                <ResponsiveContainer width="100%" height="100%">
                  <RechartsPie>
                    <Pie
                      data={fraudTypeStats}
                      cx="45%" cy="50%"
                      innerRadius="50%" outerRadius="80%"
                      dataKey="value"
                      nameKey="name"
                      stroke={METRIC_COLORS.card}
                      strokeWidth={2}
                    >
                      {fraudTypeStats.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={PIE_COLORS[index % PIE_COLORS.length]} />
                      ))}
                    </Pie>
                    <RechartsTooltip 
                      contentStyle={{ backgroundColor: '#0D1117', borderColor: '#30363D', borderRadius: '8px' }}
                      itemStyle={{ color: '#fff', fontSize: '12px' }}
                    />
                    <Legend 
                      layout="vertical" verticalAlign="middle" align="right"
                      iconType="circle"
                      wrapperStyle={{ fontSize: '11px', color: '#c9d1d9', right: '-10px' }}
                    />
                  </RechartsPie>
                </ResponsiveContainer>
              </div>
            )}
          </div>

          {/* Panel 6: Model Performance Panel */}
          <div className="bg-[#161B22] border border-[#30363D] rounded-xl p-4 shadow-lg flex-1 min-h-0 flex flex-col">
            <h2 className="text-base font-semibold text-white flex items-center gap-2 mb-4 shrink-0">
              <BarChart3 size={16} className="text-indigo-400" />
              Active Model Evaluation (Layer 3)
            </h2>
            <div className="flex flex-col gap-4 flex-1 justify-center px-2">
              
              <div className="grid grid-cols-[100px_1fr_60px] items-center gap-3">
                <span className="text-sm text-gray-400 font-medium">Precision</span>
                <div className="bg-gray-800 h-2 rounded-full overflow-hidden"><div className="bg-blue-500 h-full transition-all duration-500" style={{ width: `${(metrics?.precision || 0)*100}%` }}></div></div>
                <span className="text-sm text-right font-mono text-gray-200">{(metrics?.precision || 0).toFixed(4)}</span>
              </div>
              
              <div className="grid grid-cols-[100px_1fr_60px] items-center gap-3">
                <span className="text-sm text-gray-400 font-medium">Recall</span>
                <div className="bg-gray-800 h-2 rounded-full overflow-hidden"><div className="bg-blue-500 h-full transition-all duration-500" style={{ width: `${(metrics?.recall || 0)*100}%` }}></div></div>
                <span className="text-sm text-right font-mono text-gray-200">{(metrics?.recall || 0).toFixed(4)}</span>
              </div>

              <div className="grid grid-cols-[100px_1fr_60px] items-center gap-3">
                <span className="text-sm text-gray-400 font-medium text-indigo-300">F1-Score</span>
                <div className="bg-gray-800 h-2 rounded-full overflow-hidden"><div className="bg-indigo-500 h-full transition-all duration-500" style={{ width: `${(metrics?.f1 || 0)*100}%` }}></div></div>
                <span className="text-sm text-right font-mono text-indigo-300 font-bold">{(metrics?.f1 || 0).toFixed(4)}</span>
              </div>

              <div className="grid grid-cols-[100px_1fr_60px] items-center gap-3 mt-2 border-t border-[#30363D] pt-4">
                <span className="text-sm text-gray-400 font-medium flex items-center gap-1">
                  FPR 
                  {(metrics?.fpr || 0) > 0.02 ? (
                    <AlertTriangle size={14} className="text-[#da3633] ml-1" />
                  ) : (
                    <ShieldCheck size={14} className="text-[#238636] ml-1" />
                  )}
                </span>
                <div className="bg-gray-800 h-2 rounded-full overflow-hidden">
                  <div 
                    className={`h-full transition-all duration-500 ${(metrics?.fpr || 0) > 0.02 ? 'bg-[#da3633]' : 'bg-[#238636]'}`} 
                    style={{ width: `${Math.min((metrics?.fpr || 0)*100, 100)}%` }}
                  ></div>
                </div>
                <span className={`text-sm text-right font-mono font-bold ${(metrics?.fpr || 0) > 0.02 ? 'text-[#da3633]' : 'text-[#238636]'}`}>
                  {(metrics?.fpr || 0).toFixed(4)}
                </span>
              </div>

            </div>
            </div>
          </div>
        </div>
        </div>
        ) : (
          <ThreatIntelligence transactions={transactions} alerts={alerts} />
        )}
      </div>
    </div>
  );
}

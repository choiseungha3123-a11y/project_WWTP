"use client";

import { useState, useRef } from "react";
import { 
  AlertTriangle, Activity, DatabaseZap, CheckCircle2, 
  PencilLine, Loader2, X, Paperclip, ImageIcon, AlertCircle 
} from "lucide-react";

// ----------------------------------------------------------------------
// 1. 인터페이스 정의 (any 제거)
// ----------------------------------------------------------------------

interface TmsRecord { 
  PH_VU: number; 
  TOC_VU: number; 
  TN_VU: number; 
  TP_VU: number; 
  SS_VU: number; 
}

interface WeatherData { 
  RN_15m: number; 
}

interface AlertProps { 
  latestValues: TmsRecord | null; 
  latestWeather: WeatherData | null; 
  isDarkMode?: boolean; 
}

// 상태 타입 정의
type StatusType = "danger" | "warning" | "normal";

interface AlertItem {
  id: number;
  title: string;
  limit: string;
  valueText?: string;
  status: StatusType;
  icon: React.ReactNode;
  actionContext: string;
  details?: { name: string; val: number; status: StatusType }[];
}

// ----------------------------------------------------------------------
// 2. 메인 컴포넌트
// ----------------------------------------------------------------------

export default function Row3Alerts({ latestValues, latestWeather, isDarkMode = true }: AlertProps) {
  const [isSubmitting, setIsSubmitting] = useState<boolean>(false);
  const [showQuickMemo, setShowQuickMemo] = useState<boolean>(false);
  const [memoContent, setMemoContent] = useState<string>("");
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // 실시간 데이터 추출
  const ph = latestValues?.PH_VU ?? 7.0;
  const toc = latestValues?.TOC_VU ?? 0;
  const tn = latestValues?.TN_VU ?? 0;
  const tp = latestValues?.TP_VU ?? 0;
  const ss = latestValues?.SS_VU ?? 0;
  const rain = latestWeather?.RN_15m ?? 0;

  // --- 상태 판별 헬퍼 함수 ---
  const checkStatus = (val: number, limit: number): StatusType => {
    if (val >= limit) return "danger";
    if (val >= limit * 0.9) return "warning"; // 90% 도달 시 주의
    return "normal";
  };

  const getOverallStatus = (statuses: StatusType[]): StatusType => {
    if (statuses.includes("danger")) return "danger";
    if (statuses.includes("warning")) return "warning";
    return "normal";
  };

  // 개별 수치 상태 판별
  const tocStatus = checkStatus(toc, 15);
  const tnStatus = checkStatus(tn, 20);
  const tpStatus = checkStatus(tp, 0.5);
  const ssStatus = checkStatus(ss, 10);
  const rainStatus = checkStatus(rain, 10);

  // pH는 범위형 (5.8~8.5) -> 주의 구간은 6.0 이하 또는 8.3 이상
  const phStatus: StatusType = (ph <= 5.8 || ph >= 8.5) ? "danger" : (ph <= 6.0 || ph >= 8.3) ? "warning" : "normal";

  // 수질 통합 상태
  const tmsOverallStatus = getOverallStatus([tocStatus, tnStatus, tpStatus, ssStatus]);

  // --- 이벤트 핸들러 ---
  const handlePrepareAction = (title: string, valueInfo: string) => {
    setMemoContent(`[${title}] 조치 (${valueInfo}) : `);
    setSelectedFile(null);
    setShowQuickMemo(true);
  };

  const handleSubmitMemo = async () => {
    if (!memoContent.trim()) return alert("내용을 입력해주세요.");
    setIsSubmitting(true);
    try {
      const formData = new FormData();
      formData.append("content", memoContent);
      if (selectedFile) formData.append("file", selectedFile);

      const res = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/api/board/memo/create`, {
        method: "PUT",
        headers: { "Authorization": `Bearer ${localStorage.getItem("accessToken")}` },
        body: formData
      });
      
      const result = await res.json();
      if (result.success) {
        alert("등록되었습니다.");
        setShowQuickMemo(false);
        setMemoContent("");
        setSelectedFile(null);
        window.dispatchEvent(new CustomEvent("refreshMemoList"));
      }
    } catch (e) {
      console.error(e);
      alert("등록 중 오류가 발생했습니다.");
    } finally {
      setIsSubmitting(false);
    }
  };

  // --- 테마 설정 ---
  const theme = {
    container: isDarkMode ? "bg-slate-800/40 border-white/10" : "bg-white border-blue-100",
    headerLabel: isDarkMode ? "text-slate-500" : "text-blue-500/70",
    itemNormal: isDarkMode ? "bg-slate-900/40 border-white/5" : "bg-slate-50 border-slate-100",
    itemWarning: isDarkMode ? "bg-amber-500/10 border-amber-500/30 shadow-[0_0_10px_rgba(245,158,11,0.05)]" : "bg-amber-50 border-amber-100",
    itemDanger: isDarkMode ? "bg-red-500/10 border-red-500/30 shadow-[0_0_10px_rgba(239,68,68,0.05)]" : "bg-red-50 border-red-100",
    title: isDarkMode ? "text-slate-200" : "text-slate-800",
    limit: isDarkMode ? "text-slate-500" : "text-slate-400",
    detailBox: isDarkMode ? "bg-black/20 border-white/5" : "bg-white border-slate-100 shadow-sm",
    memoOverlay: isDarkMode ? "bg-slate-950/95" : "bg-white/98",
    memoInput: isDarkMode ? "bg-white/5 border-white/10 text-slate-200" : "bg-slate-50 border-slate-200 text-slate-800"
  };

  const alerts: AlertItem[] = [
    { id: 1, title: "pH 이상관측", limit: "5.8~8.5", valueText: ph.toFixed(2), status: phStatus, icon: <Activity className="w-4 h-4" />, actionContext: `pH ${ph.toFixed(2)}` },
    { id: 2, title: "수질 기준 초과", limit: "TOC/TN/TP/SS", status: tmsOverallStatus, icon: <AlertTriangle className="w-4 h-4" />, actionContext: `수질주의/초과`, 
      details: [
        { name: "TOC", val: toc, status: tocStatus }, { name: "T-N", val: tn, status: tnStatus }, { name: "T-P", val: tp, status: tpStatus }, { name: "SS", val: ss, status: ssStatus }
      ] 
    },
    { id: 3, title: "강우 감지", limit: "기준 10mm", valueText: `${rain.toFixed(1)}mm`, status: rainStatus, icon: <DatabaseZap className="w-4 h-4" />, actionContext: `${rain.toFixed(1)}mm` },
  ];

  return (
    <div className={`p-4 rounded-3xl border h-full flex flex-col shadow-inner relative overflow-hidden transition-all duration-500 ${theme.container}`}>
      {/* 상단 헤더 */}
      <div className="flex justify-between items-center mb-3 shrink-0">
        <h3 className={`text-[10px] font-bold uppercase tracking-widest ${theme.headerLabel}`}>Event Detection</h3>
        <div className="flex items-center gap-2">
          <span className="w-1 h-1 rounded-full bg-red-500 animate-ping"></span>
          <span className="text-[9px] text-red-400 font-black">REAL-TIME</span>
        </div>
      </div>
      
      {/* 알림 리스트 */}
      <div className="flex flex-col gap-2 flex-1 overflow-y-auto custom-scrollbar">
        {alerts.map((alert) => (
          <div key={alert.id} className={`px-3 py-2 rounded-xl border transition-all ${
            alert.status === 'danger' ? theme.itemDanger : alert.status === 'warning' ? theme.itemWarning : theme.itemNormal
          }`}>
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <div className={
                  alert.status === 'danger' ? 'text-red-500' : alert.status === 'warning' ? 'text-amber-500' : 'text-emerald-500'
                }>
                  {alert.status === 'danger' ? alert.icon : alert.status === 'warning' ? <AlertCircle className="w-3.5 h-3.5" /> : <CheckCircle2 className="w-3.5 h-3.5" />}
                </div>
                <div className="flex flex-col">
                  <span className={`text-[12px] font-bold leading-tight ${theme.title}`}>{alert.title}</span>
                  <span className={`text-[8px] font-medium uppercase tracking-tighter ${theme.limit}`}>{alert.limit}</span>
                </div>
              </div>

              {/* 수치 및 조치 버튼 (주의/위험 시 노출) */}
              <div className="flex items-center gap-2">
                {!alert.details && (
                  <span className={`text-[11px] font-mono font-bold ${
                    alert.status === 'danger' ? 'text-red-400' : alert.status === 'warning' ? 'text-amber-500' : 'text-slate-400'
                  }`}>
                    {alert.valueText}
                  </span>
                )}
                {alert.status !== 'normal' && (
                  <button 
                    onClick={() => handlePrepareAction(alert.title, alert.actionContext)} 
                    className={`flex items-center gap-1 px-2 py-0.5 rounded-md text-white text-[9px] font-black shadow-sm transition-transform active:scale-95 ${
                      alert.status === 'danger' ? 'bg-red-500 hover:bg-red-600' : 'bg-amber-500 hover:bg-amber-600'
                    }`}
                  >
                    <PencilLine className="w-2.5 h-2.5" /> 조치
                  </button>
                )}
              </div>
            </div>

            {/* 세부 수질 항목 (TOC, TN, TP, SS) */}
            {alert.details && (
              <div className={`mt-1.5 grid grid-cols-4 gap-1 border-t pt-1.5 ${isDarkMode ? 'border-white/5' : 'border-slate-100'}`}>
                {alert.details.map((d) => (
                  <div key={d.name} className={`flex flex-col items-center justify-center rounded-md py-1 border transition-colors ${theme.detailBox}`}>
                    <span className="text-[7px] text-slate-500 font-bold mb-0.5">{d.name}</span>
                    <span className={`text-[9px] font-mono font-bold ${
                      d.status === 'danger' ? 'text-red-400' : d.status === 'warning' ? 'text-amber-500' : 'text-emerald-500/80'
                    }`}>
                      {d.val.toFixed(1)}
                    </span>
                  </div>
                ))}
              </div>
            )}
          </div>
        ))}
      </div>

      {/* 퀵 메모 오버레이 (조치 레이어) */}
      {showQuickMemo && (
        <div className={`absolute inset-0 z-20 p-5 flex flex-col justify-center rounded-3xl backdrop-blur-md transition-all ${theme.memoOverlay}`}>
          <div className="flex justify-between items-center mb-3">
            <h4 className={`text-xs font-bold flex items-center gap-2 ${theme.title}`}>
              <PencilLine className="w-3 h-3 text-blue-400" /> 조치 사항 보고
            </h4>
            <button onClick={() => setShowQuickMemo(false)} className="text-slate-400 hover:text-red-500 transition-colors">
              <X className="w-4 h-4"/>
            </button>
          </div>
          <textarea 
            className={`w-full h-24 rounded-xl p-3 text-sm outline-none resize-none transition-all focus:ring-1 focus:ring-blue-500 ${theme.memoInput}`}
            value={memoContent} 
            onChange={(e) => setMemoContent(e.target.value)} 
            autoFocus 
          />
          <div className="mt-3 flex items-center justify-between">
            <div className="flex items-center gap-2">
              <input type="file" ref={fileInputRef} className="hidden" accept="image/*" onChange={(e) => setSelectedFile(e.target.files?.[0] || null)} />
              <button 
                onClick={() => fileInputRef.current?.click()} 
                className={`flex items-center gap-1 px-2 py-1.5 rounded-lg border text-[10px] font-bold transition-all ${
                  selectedFile ? 'border-blue-500 text-blue-500 bg-blue-50' : 'border-slate-200 text-slate-400 hover:bg-slate-50'
                }`}
              >
                {selectedFile ? <ImageIcon className="w-3 h-3" /> : <Paperclip className="w-3 h-3" />} 
                {selectedFile ? '변경' : '첨부'}
              </button>
              {selectedFile && <span className="text-[9px] text-slate-500 truncate max-w-20">{selectedFile.name}</span>}
            </div>
            <button 
              onClick={handleSubmitMemo} 
              disabled={isSubmitting} 
              className="px-4 py-1.5 bg-blue-600 hover:bg-blue-500 text-white rounded-lg text-[11px] font-bold flex items-center gap-2 shadow-md transition-all active:scale-95 disabled:opacity-50"
            >
              {isSubmitting ? <Loader2 className="w-3 h-3 animate-spin" /> : "등록 완료"}
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
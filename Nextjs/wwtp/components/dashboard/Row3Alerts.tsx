"use client";

import { useState, useRef } from "react";
import { 
  AlertTriangle, Activity, DatabaseZap, CheckCircle2, 
  PencilLine, Loader2, X, Paperclip, ImageIcon 
} from "lucide-react";

interface TmsRecord { PH_VU: number; TOC_VU: number; TN_VU: number; TP_VU: number; SS_VU: number; }
interface WeatherData { RN_15m: number; }
interface AlertProps { latestValues: TmsRecord | null; latestWeather: WeatherData | null; }

export default function Row3Alerts({ latestValues, latestWeather }: AlertProps) {
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [showQuickMemo, setShowQuickMemo] = useState(false);
  const [memoContent, setMemoContent] = useState("");
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const phValue = latestValues?.PH_VU ?? 7.0;
  const tocValue = latestValues?.TOC_VU ?? 0;
  const tnValue = latestValues?.TN_VU ?? 0;
  const tpValue = latestValues?.TP_VU ?? 0;
  const ssValue = latestValues?.SS_VU ?? 0;
  const rainValue = latestWeather?.RN_15m ?? 0;

  const isPhDanger = phValue <= 5.8 || phValue >= 8.5;
  const isTmsDanger = tocValue >= 15 || tnValue >= 20 || tpValue >= 0.5 || ssValue >= 10;
  const isRainDanger = rainValue >= 10;

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
        method: "PUT", // 백엔드 @PutMapping 매칭
        headers: { "Authorization": `Bearer ${localStorage.getItem("accessToken")}` },
        body: formData
      });
      const result = await res.json();
      if (result.success) {
        alert("등록되었습니다.");
        setShowQuickMemo(false);
        setMemoContent("");
        setSelectedFile(null);
        window.dispatchEvent(new CustomEvent("refreshMemoList")); // 리스트 갱신 이벤트
      }
    } catch (e) { console.error(e); } finally { setIsSubmitting(false); }
  };

  const alerts = [
    { id: 1, title: "pH 이상관측", limit: "5.8~8.5", valueText: phValue.toFixed(2), status: isPhDanger ? "danger" : "normal", icon: <Activity className="w-4 h-4" />, actionContext: `pH ${phValue.toFixed(2)}` },
    { id: 2, title: "수질 기준 초과", limit: "TOC/TN/TP/SS", status: isTmsDanger ? "danger" : "normal", icon: <AlertTriangle className="w-4 h-4" />, actionContext: `수질초과`, 
      details: [{name:"TOC", val:tocValue, danger:tocValue>=15}, {name:"T-N", val:tnValue, danger:tnValue>=20}, {name:"T-P", val:tpValue, danger:tpValue>=0.5}, {name:"SS", val:ssValue, danger:ssValue>=10}] 
    },
    { id: 3, title: "강우 감지", limit: "기준 10mm", valueText: `${rainValue.toFixed(1)}mm`, status: isRainDanger ? "danger" : "normal", icon: <DatabaseZap className="w-4 h-4" />, actionContext: `${rainValue.toFixed(1)}mm` },
  ];

  return (
    <div className="bg-slate-800/40 p-4 rounded-3xl border border-white/10 h-full flex flex-col shadow-inner relative overflow-hidden">
      <div className="flex justify-between items-center mb-3 shrink-0">
        <h3 className="text-[10px] font-bold text-slate-500 uppercase tracking-widest">Event Detection</h3>
        <div className="flex items-center gap-2">
          <span className="w-1 h-1 rounded-full bg-red-500 animate-ping"></span>
          <span className="text-[9px] text-red-400 font-black">REAL-TIME</span>
        </div>
      </div>
      <div className="flex flex-col gap-2 flex-1 overflow-y-auto custom-scrollbar">
        {alerts.map((alert) => (
          <div key={alert.id} className={`px-3 py-2 rounded-xl border transition-all ${alert.status === 'danger' ? 'bg-red-500/10 border-red-500/30' : 'bg-slate-900/40 border-white/5'}`}>
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <div className={alert.status === 'danger' ? 'text-red-500' : 'text-emerald-500'}>
                  {alert.status === 'danger' ? alert.icon : <CheckCircle2 className="w-3.5 h-3.5" />}
                </div>
                <div className="flex flex-col">
                  <span className="text-[12px] font-bold text-slate-200 leading-tight">{alert.title}</span>
                  <span className="text-[8px] text-slate-500 font-medium uppercase tracking-tighter">{alert.limit}</span>
                </div>
              </div>
              <div className="flex items-center gap-2">
                {!alert.details && <span className={`text-[11px] font-mono font-bold ${alert.status === 'danger' ? 'text-red-400' : 'text-slate-500'}`}>{alert.valueText}</span>}
                {alert.status === 'danger' && (
                  <button onClick={() => handlePrepareAction(alert.title, alert.actionContext)} className="flex items-center gap-1 px-2 py-0.5 rounded-md bg-red-500 text-white text-[9px] font-black hover:bg-red-600 shadow-sm"><PencilLine className="w-2.5 h-2.5" /> 조치</button>
                )}
              </div>
            </div>
            {alert.details && (
              <div className="mt-1.5 grid grid-cols-4 gap-1 border-t border-white/5 pt-1.5">
                {alert.details.map((d) => (
                  <div key={d.name} className="flex flex-col items-center justify-center bg-black/20 rounded-md py-1">
                    <span className="text-[7px] text-slate-500 font-bold mb-0.5">{d.name}</span>
                    <span className={`text-[9px] font-mono font-bold ${d.danger ? 'text-red-400' : 'text-emerald-500/80'}`}>{d.val.toFixed(1)}</span>
                  </div>
                ))}
              </div>
            )}
          </div>
        ))}
      </div>

      {showQuickMemo && (
        <div className="absolute inset-0 bg-slate-950/95 backdrop-blur-md z-20 p-5 flex flex-col justify-center rounded-3xl">
          <div className="flex justify-between items-center mb-3">
            <h4 className="text-xs font-bold text-white flex items-center gap-2"><PencilLine className="w-3 h-3 text-blue-400" /> 조치 사항 보고</h4>
            <button onClick={() => setShowQuickMemo(false)} className="text-slate-400 hover:text-white"><X className="w-4 h-4"/></button>
          </div>
          <textarea className="w-full h-24 bg-white/5 border border-white/10 rounded-xl p-3 text-sm text-slate-200 focus:border-blue-500 outline-none resize-none" value={memoContent} onChange={(e) => setMemoContent(e.target.value)} autoFocus />
          <div className="mt-3 flex items-center justify-between">
            <div className="flex items-center gap-2">
              <input type="file" ref={fileInputRef} className="hidden" accept="image/*" onChange={(e) => setSelectedFile(e.target.files?.[0] || null)} />
              <button onClick={() => fileInputRef.current?.click()} className={`flex items-center gap-1 px-2 py-1.5 rounded-lg border text-[10px] font-bold ${selectedFile ? 'border-blue-500 text-blue-400' : 'border-white/10 text-slate-400'}`}>
                {selectedFile ? <ImageIcon className="w-3 h-3" /> : <Paperclip className="w-3 h-3" />} {selectedFile ? '변경' : '첨부'}
              </button>
            </div>
            <button onClick={handleSubmitMemo} disabled={isSubmitting} className="px-4 py-1.5 bg-blue-600 hover:bg-blue-500 text-white rounded-lg text-[11px] font-bold flex items-center gap-2">
              {isSubmitting ? <Loader2 className="w-3 h-3 animate-spin" /> : "등록 완료"}
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
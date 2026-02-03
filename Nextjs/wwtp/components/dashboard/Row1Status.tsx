export default function Row1Status() {
  const items = [
    { label: "유입유량", value: "1,200", status: "normal" },
    { label: "수위", value: "4.5", status: "warning" },
    { label: "TMS", value: "정상", status: "normal" },
    { label: "공정상태", value: "안정", status: "normal" },
    { label: "운영점수", value: "85", status: "danger" },
    { label: "데이터", value: "수신중", status: "normal" },
  ];

  return (
    <div className="grid grid-cols-3 gap-4">
      {items.map((item, i) => (
        <div key={i} className="bg-slate-800/40 p-4 rounded-2xl border border-white/5 flex flex-col items-center justify-center">
          <span className="text-slate-400 text-xs mb-1">{item.label}</span>
          <span className={`text-xl font-black ${
            item.status === 'warning' ? 'text-orange-400' : 
            item.status === 'danger' ? 'text-red-400' : 'text-emerald-400'
          }`}>{item.value}</span>
        </div>
      ))}
    </div>
  );
}
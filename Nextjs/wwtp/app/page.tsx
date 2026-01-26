import Image from 'next/image';

export default function LandingPage() {
  return (
    <main className="relative min-h-[200vh] w-full">
      <div className="fixed inset-0 w-full h-screen z-0">
        <video
          autoPlay
          loop
          muted
          playsInline
          className="w-full h-full object-cover"
        >
          <source src="/videos/홈페이지배경영상.mp4" type="video/mp4" />
        </video>
      
        <div className="absolute inset-0 bg-black/50" />
      </div>

      
      <section className="relative z-10 h-screen flex items-center justify-center">
        <div className="text-center">
          <button className="px-12 py-4 bg-white/20 hover:bg-white/40 text-white border border-white/50 backdrop-blur-md rounded-full text-xl transition-all">
            로그인
          </button>
        </div>
      </section>

      
      <section className="relative z-10 min-h-screen bg-white/10 backdrop-blur-lg text-white p-20">
  <div className="max-w-4xl mx-auto">
    <h2 className="text-4xl mb-10 font-bold">하수처리장 스마트 시스템 소개</h2>
    
    <div className="space-y-6 text-xl leading-loose mb-12">
      <p className="font-bold text-blue-300">
        안정적인 생물학적 처리를 위해 실시간 유입 부하를 상시 확인하십시오.<br />
        유량 변동은 미생물 침강성과 송풍기 효율에 직결됩니다.
      </p>
      
      <div className="grid gap-6 text-base opacity-90">
        <p><strong>1. 유량 계측기 유지관리:</strong> 초음파 유량계 센서 이물질 제거 및 영점 조절 주기 준수. (데이터 신뢰성 확보)</p>
        <p><strong>2. 강우 시 비상 운전:</strong> 유입량 급증 시 선행 침전지 수위 확인 및 바이패스 검토. (슬러지 유출 방지)</p>
        <p><strong>3. 계획 대비 유입 분석:</strong> 설계 용량 대비 현재 유입률 상시 분석. (시설 최적화 근거 마련)</p>
      </div>
    </div>

    {/* --- 이미지 삽입 섹션 --- */}
    <div className="mt-12 p-4 bg-white/5 rounded-2xl border border-white/10 shadow-2xl">
      <h3 className="text-2xl mb-6 font-semibold text-center">Sewage Treatment Process Flow</h3>
      <div className="relative w-full overflow-hidden rounded-lg">
        <Image
          src="/img/하수처리장.jpg" 
          alt="하수처리 공정 흐름도" 
          width={1000}
          height={600}
          className="rounded-lg shadow-lg"
        />
      </div>
      <p className="mt-4 text-center text-sm text-white/50">
        [ 이미지: 하수처리장의 주요 공정 단계별 흐름도 ]
      </p>
    </div>

    {/* 하단 여백 */}
    <div className="h-50" /> 
  </div>
</section>

      <section>
        <div className="relative z-10 min-h-screen bg-white/10 backdrop-blur-lg text-white p-20">
          <div className="max-w-4xl mx-auto">
            <h2 className="text-4xl mb-10">시스템 주요 기능</h2>
            <ul className="list-disc list-inside space-y-4 text-xl font-bold leading-loose">
              <li>실시간 유량 모니터링: 정확한 유입 부하 데이터 제공</li>
              <li>자동 경고 시스템: 이상 유입 시 즉각 알림</li>
              <li>데이터 분석 도구: 유입 패턴 및 트렌드 분석 지원</li>
              <li>사용자 친화적 인터페이스: 직관적인 대시보드 제공</li>
            </ul>
            <div className="h-125" /> 
          </div>
        </div>
      </section>
    </main>
  );
}
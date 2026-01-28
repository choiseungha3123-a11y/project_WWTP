"use client"; // 

import { useState } from 'react'; 
import { useRouter } from 'next/navigation';
import Image from 'next/image';
import { motion, AnimatePresence } from 'framer-motion'; 
import { jwtDecode } from 'jwt-decode';

export default function LandingPage() {
  const [isLoginOpen, setIsLoginOpen] = useState(false);
  const router = useRouter();

  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');

  const handleLogin = async (e: React.FormEvent) => {
    e.preventDefault();

  interface CustomJwtPayload {
  Userno: string;  
  Role?: string;
  Userid?: string;
  Username?: string;
  auth?: string;
  name?: string;
  sub?: string;
}
    console.log('Attempting login with:', { username, password });
    try {
      const response = await fetch('/api/member/login', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          userId: username,
          password: password,
        }),
      });

      if (response.ok) {
        const data = await response.json();
        if (!data.success) {
          alert(`로그인에 실패했습니다. 아이디와 비밀번호를 확인해주세요. (에러 코드: ${data.errorMsg})`);
        }
        else {
          const jwtToken = data.dataList[0];
          const decodedToken = jwtDecode<CustomJwtPayload>(jwtToken);
          console.log('Decoded JWT Token:', decodedToken);

          const userNo = decodedToken.Userno || "";
          const userId = decodedToken.Userid || "";
          const role = decodedToken.Role || ""; 
          const name = decodedToken.Username || "사용자"; 

          localStorage.setItem('accessToken', jwtToken);
          localStorage.setItem('userNo', String(userNo));
          localStorage.setItem('userId', userId);
          localStorage.setItem('userRole', role); 
          localStorage.setItem('userName', name);


          console.log('최종 저장 데이터:', { userNo, role, name }); 

          router.push('/dashboard');
        }
      } else {
        alert('로그인에 실패했습니다. 아이디와 비밀번호를 확인해주세요.');
      }
    } catch (error) {
      console.error('Error during login:', error);
      alert('로그인 중 오류가 발생했습니다. 다시 시도해주세요.');
    }
  };

  return (
    <main className="relative min-h-[200vh] w-full">
      
      <div className="fixed inset-0 w-full h-screen z-0">
        <video autoPlay loop muted playsInline className="w-full h-full object-cover">
          <source src="/videos/홈페이지배경영상.mp4" type="video/mp4" />
        </video>
        <div className="absolute inset-0 bg-black/50" />
      </div>

      
      <section className="relative z-10 h-screen flex items-center justify-center">
        <div className="text-center">
          
          <button 
            onClick={() => setIsLoginOpen(true)}
            className="px-12 py-4 bg-white/20 hover:bg-white/40 text-white border border-white/50 backdrop-blur-md rounded-full text-xl font-light tracking-widest transition-all duration-300 hover:scale-105 active:scale-95 cursor-pointer"
          >
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
      <AnimatePresence>
        {isLoginOpen && (
          <>
           
            <motion.div 
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              onClick={() => setIsLoginOpen(false)} 
              className="fixed inset-0 bg-black/40 backdrop-blur-sm z-100"
            />

            
            <motion.div 
              initial={{ x: "100%" }} 
              animate={{ x: 0 }}      
              exit={{ x: "100%" }}    
              transition={{ type: "spring", damping: 25, stiffness: 200 }}
              className="fixed top-0 right-0 h-full w-full md:w-112.5 bg-black/80 backdrop-blur-2xl z-101 p-12 flex flex-col justify-center border-l border-white/10 shadow-2xl"
            >
              <button 
                onClick={() => setIsLoginOpen(false)}
                className="absolute top-10 right-10 text-white/50 hover:text-white"
              >
                닫기 ✕
              </button>

              <h2 className="text-3xl font-bold text-white mb-2">시스템 로그인</h2>
              <p className="text-white/40 mb-10">스마트 시스템 접속 권한을 확인합니다.</p>

              <form className="space-y-6" onSubmit={handleLogin}> {/* 1. 여기에 onSubmit 추가 */}
                <div>
                  <label className="block text-sm text-white/60 mb-2">ID</label>
                  <input 
                    type="text" 
                    value={username} // 2. 상태(useState) 연결
                    onChange={(e) => setUsername(e.target.value)} // 3. 입력 시 값 업데이트
                    className="w-full p-4 bg-white/5 border border-white/20 rounded-xl text-white outline-none focus:border-blue-500 font-sans" 
                    placeholder="아이디 입력" 
                    required
                  />
                </div>
                <div>
                  <label className="block text-sm text-white/60 mb-2">Password</label>
                  <input 
                    type="password" 
                    value={password} // 4. 상태(useState) 연결
                    onChange={(e) => setPassword(e.target.value)} // 5. 입력 시 값 업데이트
                    className="w-full p-4 bg-white/5 border border-white/20 rounded-xl text-white outline-none focus:border-blue-500 font-sans" 
                    placeholder="비밀번호 입력" 
                    required
                  />
                </div>
                <button 
                  type="submit" // 6. 클릭 시 handleLogin 실행
                  className="w-full py-4 bg-blue-600 hover:bg-blue-500 text-white rounded-xl font-bold transition-all active:scale-95 shadow-lg"
                >
                  접속하기
                </button>
              </form>
            </motion.div>
          </>
        )}
      </AnimatePresence>
    </main>
  );
}
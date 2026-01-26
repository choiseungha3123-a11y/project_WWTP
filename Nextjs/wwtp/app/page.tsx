import Link from 'next/link';

export default function LandingPage() {
  return (
    <main className="relative h-screen w-full overflow-hidden flex items-center justify-center">
      <video
        autoPlay
        loop
        muted
        playsInline
        className="absolute inset-0 w-full h-full object-cover z-0"
      >
        <source src="/videos/배경영상.mp4" type="video/mp4" />
        브라우저가 동영상을 지원하지 않습니다.
      </video>

      <div className="absolute inset-0 bg-black/40 z-10" />

      <div className="relative z-20">
        <Link 
          href="/login" 
          className="px-12 py-4 bg-white/20 hover:bg-white/40 text-white border border-white/50 backdrop-blur-md rounded-full text-xl font-light tracking-widest transition-all duration-300 hover:scale-105 active:scale-95"
        >
          로그인
        </Link>
      </div>

    </main>
  );
}
import React, { useState, useEffect, useRef } from 'react';
import CameraFeed from './components/CameraFeed';
import { MobileControlBar } from './components/MobileControlBar';
import { toast } from 'sonner';
import { Toaster } from './components/ui/sonner';

export default function App() {
  const [isActive, setIsActive] = useState(false);
  const [predictedWord, setPredictedWord] = useState('');
  const [confidence, setConfidence] = useState(0);
  const [renderedImage, setRenderedImage] = useState<string>(''); // 실시간 뼈대 이미지 보관

  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const wsRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    let interval: NodeJS.Timeout;

    if (isActive) {
      // 1. 브라우저 웹캠 구동
      navigator.mediaDevices.getUserMedia({ video: { width: 480, height: 360 } })
        .then((stream) => {
          if (videoRef.current) {
            videoRef.current.srcObject = stream;
            videoRef.current.onloadedmetadata = () => {
              videoRef.current?.play().catch(e => console.error("Auto-play failed:", e));
            };
          }
        })
        .catch((err) => {
          console.error("Camera access failed:", err);
          toast.error('카메라 권한을 허용해주세요.');
          setIsActive(false);
        });

      // 2. Python FastAPI 서버와 웹소켓 연결
      // 아래 코드를 그대로 복사해서 붙여넣으세요!
      wsRef.current = new WebSocket('ws://localhost:8000/ws');
  

      

      wsRef.current.onopen = () => {
        console.log('FastAPI 웹소켓 연결 성공');
        
        // 3. 주기적으로 캔버스를 통해 비디오 프레임을 캡처하여 파이썬 서버로 실시간 전송
        interval = setInterval(() => {
          if (videoRef.current && canvasRef.current && wsRef.current?.readyState === WebSocket.OPEN) {
            const video = videoRef.current;
            const canvas = canvasRef.current;
            const ctx = canvas.getContext('2d');
            
            if (ctx && video.videoWidth > 0) {
              canvas.width = video.videoWidth;
              canvas.height = video.videoHeight;
              ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
              
              // 프레임을 JPEG 문자열로 인코딩하여 백엔드로 전달
              const base64Frame = canvas.toDataURL('image/jpeg', 0.5);
              wsRef.current.send(base64Frame);
            }
          }
        }, 100); // 100ms 마다 1프레임 전송 (10 FPS)
      };

      // 4. 백엔드(Python)가 보낸 실시간 연산 분석 결과 수신
      wsRef.current.onmessage = (event) => {
        try {
          const res = JSON.parse(event.data);
          if (res.word !== undefined) setPredictedWord(res.word);
          if (res.confidence !== undefined) setConfidence(res.confidence);
          if (res.image) {
            setRenderedImage(res.image); // 백엔드에서 특징점을 그린 이미지를 상태값으로 받아옴
          }
        } catch (e) {
          console.error("Data parsing error:", e);
        }
      };

      wsRef.current.onerror = (err) => {
        console.error('웹소켓 에러:', err);
      };

      wsRef.current.onclose = () => {
        console.log('웹소켓 연결 종료');
      };
    } else {
      // 카메라 OFF 시 리소스 해제 및 초기화
      if (wsRef.current) {
        wsRef.current.close();
      }
      if (videoRef.current && videoRef.current.srcObject) {
        const stream = videoRef.current.srcObject as MediaStream;
        stream.getTracks().forEach(track => track.stop());
        videoRef.current.srcObject = null;
      }
      setPredictedWord('');
      setConfidence(0);
      setRenderedImage('');
    }

    return () => {
      clearInterval(interval);
      if (wsRef.current) wsRef.current.close();
    };
  }, [isActive]);

  const handleToggleCamera = () => {
    setIsActive(prev => !prev);
  };

  return (
    <div className="flex flex-col h-screen w-full bg-slate-50 font-sans antialiased selection:bg-emerald-500/10">
      <Toaster position="top-center" richColors />
      
      {/* 헤더 바 */}
      <header className="flex items-center justify-between px-4 py-3 bg-white border-b border-gray-100 shrink-0 shadow-sm">
        <div className="flex items-center gap-2.5">
          <div className="w-8 h-8 rounded-xl bg-gradient-to-tr from-emerald-500 to-teal-600 flex items-center justify-center shadow-md shadow-emerald-500/20">
            <svg className="w-4 h-4 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2.5" d="M3 5h12M9 3v2m1.048 9.5A18.022 18.022 0 0113 12h2m2 0h-5m5 0a13.978 13.978 0 01-2.997 8.75M21 21l-3-3m1-1.5A7.5 7.5 0 1110.5 3 7.5 7.5 0 0119 10.5z" />
            </svg>
          </div>
          <div>
            <div className="text-sm font-bold text-gray-900">AI 수어 번역기</div>
            <div className="text-[11px] text-gray-400 leading-tight">Sign Language Translator</div>
          </div>
        </div>
        <div className="flex items-center gap-2 bg-gray-100 rounded-full px-3 py-1.5 border border-gray-200">
          <div className={`w-2 h-2 rounded-full ${isActive ? 'bg-emerald-500 animate-pulse' : 'bg-gray-300'}`} />
          <span className={`text-[11px] ${isActive ? 'text-emerald-600' : 'text-gray-400'}`}>{isActive ? 'LIVE' : 'OFF'}</span>
        </div>
      </header>

      {/* 비디오 및 결과 레이어 오버레이 영역 */}
      <div className="flex-1 relative mx-3 mt-3 mb-3 rounded-3xl overflow-hidden min-h-0 shadow-lg bg-black flex items-center justify-center">
        
        {/* 순정 카메라 비디오는 분석용으로만 쓰고 레이아웃 상에서는 숨김 처리 */}
        <video 
          ref={videoRef} 
          autoPlay 
          playsInline 
          muted 
          className="hidden" 
        />
        <canvas ref={canvasRef} className="hidden" />
        
        {/* 실시간 뼈대가 입혀진 이미지를 대신 노출 */}
        {isActive && renderedImage ? (
          <img 
            src={renderedImage} 
            alt="Realtime Sign Skeleton Tracking"
            className="w-full h-full object-cover"
          />
        ) : (
          <div className={isActive ? 'hidden' : 'text-gray-500 text-sm'}>카메라를 켜주세요.</div>
        )}
        
        <div className="absolute inset-0 z-10 flex">
          <CameraFeed 
            isActive={isActive} 
            predictedWord={predictedWord} 
            confidence={confidence} 
          />
        </div>
      </div>

      {/* 하단 모바일 컨트롤 바 조작계 */}
      <MobileControlBar 
        isActive={isActive} 
        onToggleCamera={handleToggleCamera} 
      />
    </div>
  );
}
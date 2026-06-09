import React, { useState, useEffect, useRef } from 'react';
import CameraFeed from './components/CameraFeed'; // 중괄호 제거 버전
import { MobileControlBar } from './components/MobileControlBar';
import { toast } from 'sonner';
import { Toaster } from './components/ui/sonner';

export default function App() {
  const [isActive, setIsActive] = useState(false);
  const [predictedWord, setPredictedWord] = useState('');
  const [confidence, setConfidence] = useState(0);

  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const wsRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    let interval: NodeJS.Timeout;

    if (isActive) {
      // 1. 브라우저 웹캠 구동
      navigator.mediaDevices.getUserMedia({ video: true })
        .then((stream) => {
          if (videoRef.current) {
            videoRef.current.srcObject = stream;
            // 비디오가 데이터를 받으면 즉시 재생되도록 유도
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
      wsRef.current = new WebSocket('ws://localhost:8000/ws');
      wsRef.current.onopen = () => console.log('WebSocket 연결 완료');
      
      wsRef.current.onmessage = (event) => {
        const data = JSON.parse(event.data);
        setPredictedWord(data.word);
        setConfidence(data.confidence);
      };

      // 3. 주기적으로 프레임 캡처 후 서버로 전송 (전송 부하 감소를 위해 320x240, 15fps 설정)
      interval = setInterval(() => {
        if (videoRef.current && canvasRef.current && wsRef.current?.readyState === WebSocket.OPEN) {
          const context = canvasRef.current.getContext('2d');
          if (context) {
            // 캔버스 크기를 축소하여 전송 데이터량 감소
            context.drawImage(videoRef.current, 0, 0, 320, 240);
            const base64Image = canvasRef.current.toDataURL('image/jpeg', 0.5); // 압축률 0.5로 낮춤
            if (base64Image && base64Image.startsWith('data:image/jpeg;base64,')) {
              wsRef.current.send(base64Image);
            }
          }
        }
      }, 66); // 15fps
    } else {
      // 카메라 종료 시 스트림 및 소켓 리셋
      setPredictedWord('');
      setConfidence(0);
      if (wsRef.current) wsRef.current.close();
      if (videoRef.current?.srcObject) {
        (videoRef.current.srcObject as MediaStream).getTracks().forEach(track => track.stop());
      }
    }

    return () => {
      clearInterval(interval);
      if (wsRef.current) wsRef.current.close();
      if (videoRef.current?.srcObject) {
        (videoRef.current.srcObject as MediaStream).getTracks().forEach(track => track.stop());
      }
    };
  }, [isActive]);

  const handleToggleCamera = () => {
    setIsActive(!isActive);
    if (!isActive) toast.success('카메라가 시작되었습니다');
    else toast.info('카메라가 종료되었습니다');
  };

  const handleReset = () => {
    setPredictedWord('');
    setConfidence(0);
    toast.info('초기화되었습니다');
  };

  return (
    <div className="h-dvh bg-gray-100 text-gray-900 flex flex-col overflow-hidden">
      <Toaster position="top-center" />

      {/* 상단 헤더 */}
      <header className="flex items-center justify-between px-5 pt-5 pb-3 shrink-0 bg-white border-b border-gray-200">
        <div className="flex items-center gap-2.5">
          <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-emerald-400 to-cyan-400 flex items-center justify-center shadow-md shadow-emerald-200">
            <span className="text-base">🤙</span>
          </div>
          <div>
            <div className="text-base leading-tight text-gray-900">AI 수어 번역기</div>
            <div className="text-[11px] text-gray-400 leading-tight">Sign Language Translator</div>
          </div>
        </div>
        <div className="flex items-center gap-2 bg-gray-100 rounded-full px-3 py-1.5 border border-gray-200">
          <div className={`w-2 h-2 rounded-full ${isActive ? 'bg-emerald-500 animate-pulse' : 'bg-gray-300'}`} />
          <span className={`text-[11px] ${isActive ? 'text-emerald-600' : 'text-gray-400'}`}>{isActive ? 'LIVE' : 'OFF'}</span>
        </div>
      </header>

      {/* 카메라 및 결과 오버레이 영역 */}
      <div className="flex-1 relative mx-3 mt-3 mb-3 rounded-3xl overflow-hidden min-h-0 shadow-lg bg-black flex items-center justify-center">
        {/* [치트키 1 적용] 인라인 스타일로 렌더링 강제 및playsInline 속성 강화 */}
        <video 
          ref={videoRef} 
          autoPlay 
          playsInline 
          muted 
          className={!isActive ? 'hidden' : ''} 
          style={{ width: '100%', height: '100%', objectFit: 'cover' }}
        />
        <canvas ref={canvasRef} width="320" height="240" className="hidden" />
        
        <div className="absolute inset-0 z-10 flex">
          <CameraFeed
            isActive={isActive}
            predictedWord={predictedWord}
            confidence={confidence}
          />
        </div>
      </div>

      {/* 하단 컨트롤 바 */}
      <MobileControlBar
        isActive={isActive}
        onToggleCamera={handleToggleCamera}
        onReset={handleReset}
      />
    </div>
  );
}
import React from 'react';

interface CameraFeedProps {
  isActive: boolean;
  predictedWord: string;
  confidence: number;
}

export default function CameraFeed({ isActive, predictedWord, confidence }: CameraFeedProps) {
  return (
    <div className="w-full h-full flex flex-col justify-between p-6 pointer-events-none z-10">
      {/* 상단 알림 영역 */}
      <div className="self-end">
        {!isActive && (
          <div className="bg-black/40 backdrop-blur-md text-white text-xs px-3 py-1.5 rounded-full border border-white/10">
            카메라가 꺼져 있습니다
          </div>
        )}
      </div>

      {/* 하단 실시간 수어 번역 결과 UI */}
      {isActive && predictedWord && (
        <div className="w-full bg-white/90 backdrop-blur-md rounded-2xl p-4 border border-gray-200/50 shadow-xl pointer-events-auto animate-fade-in">
          <div className="text-[11px] text-gray-400 font-medium uppercase tracking-wider mb-1">
            실시간 수어 번역 결과
          </div>
          <div className="flex items-end justify-between">
            <div className="text-2xl font-bold text-gray-950 tracking-tight">
              {predictedWord}
            </div>
            <div className="flex items-center gap-1.5 bg-emerald-50 px-2 py-1 rounded-lg border border-emerald-100">
              <span className="text-[10px] font-semibold text-emerald-600 uppercase tracking-wider">신뢰도</span>
              <span className="text-xs font-bold text-emerald-700">{confidence}%</span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
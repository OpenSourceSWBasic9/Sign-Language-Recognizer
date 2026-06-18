import React from 'react';
import { motion } from 'motion/react';
import { Video, VideoOff, RotateCcw } from 'lucide-react';

interface MobileControlBarProps {
  isActive: boolean;
  onToggleCamera: () => void;
  onReset: () => void;
}

export function MobileControlBar({ isActive, onToggleCamera, onReset }: MobileControlBarProps) {
  return (
    <div className="shrink-0 px-6 pt-4 pb-6 bg-white border-t border-gray-200">
      <div className="flex items-center justify-between">
        {/* 시퀀스 초기화 */}
        <button
          onClick={onReset}
          disabled={!isActive}
          className="flex flex-col items-center gap-1.5 disabled:opacity-30 transition-opacity"
        >
          <div className="w-12 h-12 rounded-2xl bg-gray-100 border border-gray-200 flex items-center justify-center">
            <RotateCcw className="w-5 h-5 text-gray-500" />
          </div>
          <span className="text-[11px] text-gray-400">초기화</span>
        </button>

        {/* 메인 카메라 버튼 */}
        <motion.button
          onClick={onToggleCamera}
          whileTap={{ scale: 0.94 }}
          whileHover={{ scale: 1.04 }}
          className="flex flex-col items-center gap-1.5"
        >
          <div className={`
            w-20 h-20 rounded-full flex items-center justify-center shadow-lg transition-all
            ${isActive
              ? 'bg-red-500 shadow-red-200'
              : 'bg-emerald-500 shadow-emerald-200'
            }
          `}>
            {isActive ? (
              <VideoOff className="w-8 h-8 text-white" />
            ) : (
              <Video className="w-8 h-8 text-white" />
            )}
          </div>
          <span className="text-[11px] text-gray-500">{isActive ? '종료' : '시작'}</span>
        </motion.button>

        {/* 빈 공간 (대칭을 위해) */}
        <div className="w-12"></div>
      </div>
    </div>
  );
}

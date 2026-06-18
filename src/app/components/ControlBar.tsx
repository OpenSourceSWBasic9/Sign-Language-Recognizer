import React from 'react';
import { Button } from './ui/button';
import { Video, VideoOff, RotateCcw, BookOpen } from 'lucide-react';

interface ControlBarProps {
  isActive: boolean;
  onToggleCamera: () => void;
  onReset: () => void;
  onShowGuide: () => void;
}

export function ControlBar({ isActive, onToggleCamera, onReset, onShowGuide }: ControlBarProps) {
  return (
    <div className="fixed bottom-0 left-0 right-0 bg-gray-950/95 backdrop-blur-lg border-t border-gray-800/50">
      <div className="max-w-7xl mx-auto px-6 py-4">
        <div className="flex items-center justify-center gap-4">
          <Button
            size="lg"
            variant={isActive ? 'destructive' : 'default'}
            onClick={onToggleCamera}
            className={`
              px-8 transition-all
              ${isActive 
                ? 'bg-red-600 hover:bg-red-700 text-white' 
                : 'bg-emerald-600 hover:bg-emerald-700 text-white'
              }
            `}
          >
            {isActive ? (
              <>
                <VideoOff className="size-5" />
                카메라 종료
              </>
            ) : (
              <>
                <Video className="size-5" />
                카메라 시작
              </>
            )}
          </Button>

          <Button
            size="lg"
            variant="outline"
            onClick={onReset}
            disabled={!isActive}
            className="px-8 border-gray-700 hover:bg-gray-800"
          >
            <RotateCcw className="size-5" />
            시퀀스 초기화
          </Button>

          <Button
            size="lg"
            variant="outline"
            onClick={onShowGuide}
            className="px-8 border-gray-700 hover:bg-gray-800"
          >
            <BookOpen className="size-5" />
            단어 사전 안내
          </Button>
        </div>
      </div>
    </div>
  );
}

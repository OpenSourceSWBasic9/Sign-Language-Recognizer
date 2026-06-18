import React from 'react';
import { motion, AnimatePresence } from 'motion/react';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Badge } from './ui/badge';
import { Sparkles } from 'lucide-react';

interface RecognitionPanelProps {
  predictedWord: string;
  confidence: number;
}

const availableWords = [
  { word: '안녕하세요', emoji: '👋' },
  { word: '고맙습니다', emoji: '🙏' },
  { word: '나', emoji: '🙋' },
  { word: '먹다', emoji: '🍽️' },
  { word: '사랑합니다', emoji: '❤️' },
];

export function RecognitionPanel({ predictedWord, confidence }: RecognitionPanelProps) {
  return (
    <div className="h-full flex flex-col space-y-4">
      {/* 예측 결과 */}
      <Card className="bg-gradient-to-br from-blue-950 to-gray-900 border-blue-800/30">
        <CardHeader>
          <CardTitle className="text-sm text-blue-300 flex items-center gap-2">
            <Sparkles className="size-4" />
            AI 예측 결과
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-3">
          <AnimatePresence mode="wait">
            <motion.div
              key={predictedWord}
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.9 }}
              transition={{ duration: 0.3 }}
              className="text-center"
            >
              <div className="text-4xl font-bold text-white mb-2">
                {predictedWord || '대기 중...'}
              </div>
              {predictedWord && (
                <motion.div
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  className="flex items-center justify-center gap-2"
                >
                  <span className="text-sm text-blue-300">신뢰도:</span>
                  <Badge className="bg-emerald-500/20 text-emerald-400 border-emerald-500/30">
                    {confidence}%
                  </Badge>
                </motion.div>
              )}
            </motion.div>
          </AnimatePresence>
        </CardContent>
      </Card>

      {/* 인식 가능한 단어 목록 */}
      <Card className="bg-gray-900/50 border-gray-800/50 flex-1">
        <CardHeader>
          <CardTitle className="text-sm text-gray-300">인식 가능한 단어</CardTitle>
        </CardHeader>
        <CardContent className="space-y-2">
          {availableWords.map((item, index) => (
            <motion.div
              key={item.word}
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: index * 0.1 }}
              className={`
                p-3 rounded-lg border transition-all cursor-pointer
                ${predictedWord === item.word 
                  ? 'bg-emerald-500/20 border-emerald-500/50 shadow-lg shadow-emerald-500/20' 
                  : 'bg-gray-800/50 border-gray-700/50 hover:bg-gray-800 hover:border-gray-600'
                }
              `}
            >
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-3">
                  <span className="text-2xl">{item.emoji}</span>
                  <span className={`font-medium ${predictedWord === item.word ? 'text-emerald-300' : 'text-gray-200'}`}>
                    {item.word}
                  </span>
                </div>
                {predictedWord === item.word && (
                  <motion.div
                    initial={{ scale: 0 }}
                    animate={{ scale: 1 }}
                    className="flex items-center"
                  >
                    <Badge className="bg-emerald-500 text-white border-0">
                      인식됨
                    </Badge>
                  </motion.div>
                )}
              </div>
            </motion.div>
          ))}
        </CardContent>
      </Card>

      {/* 통계 정보 */}
      <Card className="bg-gray-900/30 border-gray-800/30">
        <CardContent className="py-4">
          <div className="grid grid-cols-2 gap-4 text-center">
            <div>
              <div className="text-2xl font-bold text-blue-400">5</div>
              <div className="text-xs text-gray-400">학습된 단어</div>
            </div>
            <div>
              <div className="text-2xl font-bold text-emerald-400">98.5%</div>
              <div className="text-xs text-gray-400">평균 정확도</div>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}

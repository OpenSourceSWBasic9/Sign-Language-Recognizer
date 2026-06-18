import React from 'react';
import { motion } from 'motion/react';

interface WordChipsProps {
  predictedWord: string;
}

const availableWords = [
  { word: '안녕하세요', emoji: '👋' },
  { word: '고맙습니다', emoji: '🙏' },
  { word: '나', emoji: '🙋' },
  { word: '먹다', emoji: '🍽️' },
  { word: '사랑합니다', emoji: '❤️' },
];

export function WordChips({ predictedWord }: WordChipsProps) {
  return (
    <div>
      <div className="flex items-center justify-between mb-2 px-1">
        <span className="text-[11px] text-gray-400 uppercase tracking-wider">인식 가능한 단어</span>
        <span className="text-[11px] text-gray-400">{availableWords.length}개</span>
      </div>
      <div className="flex gap-2 overflow-x-auto pb-1 scrollbar-none">
        {availableWords.map((item, index) => {
          const isActive = predictedWord === item.word;
          return (
            <motion.div
              key={item.word}
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ delay: index * 0.05 }}
              className={`
                shrink-0 flex items-center gap-1.5 px-3 py-2 rounded-2xl border transition-all
                ${isActive
                  ? 'bg-emerald-50 border-emerald-300 shadow-sm shadow-emerald-100'
                  : 'bg-white border-gray-200'
                }
              `}
            >
              <span className="text-base leading-none">{item.emoji}</span>
              <span className={`text-[13px] whitespace-nowrap ${isActive ? 'text-emerald-700' : 'text-gray-600'}`}>
                {item.word}
              </span>
              {isActive && (
                <motion.div
                  initial={{ scale: 0 }}
                  animate={{ scale: 1 }}
                  className="w-1.5 h-1.5 rounded-full bg-emerald-500"
                />
              )}
            </motion.div>
          );
        })}
      </div>
    </div>
  );
}
